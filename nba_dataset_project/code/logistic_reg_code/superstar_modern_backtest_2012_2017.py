from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from superstar_logreg_label_tests import (
    OVERRIDE_FLOOR_BPM,
    OVERRIDE_FLOOR_PER,
    OVERRIDE_FLOOR_WS_PER_48,
    RISING_SUPERSTAR_MIN_VOTES,
    SUPERSTAR_MIN_VOTES,
    build_model,
    build_multiyear_test_features,
    build_multiyear_train_features,
    evaluate_option,
    label_map,
    preprocess_advanced_stats,
)


def parse_args() -> argparse.Namespace:
    """Read CLI arguments for this run."""
    parser = argparse.ArgumentParser(
        description="Leakage-safe walk-forward backtest with 3-tier superstar statuses for 2012-2017."
    )
    parser.add_argument("--stats-csv", type=Path, default=Path("NBA_Advanced_Stats_2002-2022.csv"))
    parser.add_argument("--all-nba-csv", type=Path, default=Path("all_nba_selections.csv"))
    parser.add_argument("--mvp-csv", type=Path, default=Path("mvp_votings.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eda_outputs"))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--labels", type=str, default="all")
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--target-positive-rate", type=float, default=0.15)
    parser.add_argument(
        "--threshold-override",
        type=float,
        default=None,
        help="Optional fixed threshold in [0.10, 0.90] for selected mode; if omitted, selected mode uses business threshold.",
    )
    parser.add_argument(
        "--single-threshold-only",
        action="store_true",
        help="Only report selected threshold mode (skip extra business_target row).",
    )
    return parser.parse_args()


def run_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtests and collect predictions/metrics."""
    df = preprocess_advanced_stats(args.stats_csv, args.all_nba_csv, args.mvp_csv)
    multi_x_full = build_multiyear_train_features(df)
    valid_mask = multi_x_full.notna().any(axis=1)
    df_valid = df.loc[valid_mask].copy()
    multi_x_valid = multi_x_full.loc[valid_mask].copy()

    available_models = ["logreg", "random_forest", "gradient_boosting"]
    selected_models = (
        available_models if args.models.strip().lower() == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    )
    invalid_models = [m for m in selected_models if m not in available_models]
    if invalid_models:
        raise ValueError(f"Unknown models: {invalid_models}")

    all_preds: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    for year in range(args.start_year, args.end_year + 1):
        train_mask = df_valid["year"] < year
        test_mask = df_valid["year"] == year
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        train_df = df_valid.loc[train_mask].copy()
        train_x = multi_x_valid.loc[train_mask].copy()
        test_df = df_valid.loc[test_mask].copy()
        hist_df = df_valid[df_valid["year"] < year].copy()
        test_x = build_multiyear_test_features(hist_df, test_df, list(train_x.columns))
        if test_x.empty or not test_x.notna().any(axis=1).any():
            continue

        option_to_y = label_map(train_df)
        selected_labels = (
            list(option_to_y.keys())
            if args.labels.strip().lower() == "all"
            else [s.strip() for s in args.labels.split(",") if s.strip()]
        )
        invalid_labels = [n for n in selected_labels if n not in option_to_y]
        if invalid_labels:
            raise ValueError(f"Unknown label options: {invalid_labels}")

        evaluated_models: dict[tuple[str, str], tuple[object, float]] = {}
        year_metric_rows: list[dict[str, float]] = []
        for name in selected_labels:
            y = option_to_y[name]
            for model_name in selected_models:
                metric_rows, model, threshold = evaluate_option(
                    train_df,
                    train_x,
                    y,
                    name,
                    model_name,
                    target_positive_rate=args.target_positive_rate,
                    threshold_override=args.threshold_override,
                    single_threshold_only=args.single_threshold_only,
                )
                if metric_rows is None or model is None or threshold is None:
                    continue
                for metric in metric_rows:
                    metric["backtest_year"] = year
                    year_metric_rows.append(metric)
                evaluated_models[(name, model_name)] = (model, threshold)

        if not evaluated_models:
            continue
        if year_metric_rows:
            all_metrics.append(pd.DataFrame(year_metric_rows))

        all_nba_gate = test_df["all_nba_1st_3x_gate"].to_numpy(dtype=int)
        recency_gate = pd.to_numeric(test_df["made_all_nba_prev3_seasons"], errors="coerce").fillna(0.0).to_numpy(dtype=int)
        season_all_nba_gate = pd.to_numeric(test_df["made_all_nba_that_season"], errors="coerce").fillna(0.0).to_numpy(dtype=int)
        test_bpm = pd.to_numeric(test_df["bpm"], errors="coerce").fillna(0.0).to_numpy()
        test_ws48 = pd.to_numeric(test_df["ws_per_48"], errors="coerce").fillna(0.0).to_numpy()
        test_per = pd.to_numeric(test_df["per"], errors="coerce").fillna(0.0).to_numpy()
        current_season_floor = (
            ((test_bpm >= OVERRIDE_FLOOR_BPM).astype(int))
            + ((test_ws48 >= OVERRIDE_FLOOR_WS_PER_48).astype(int))
            + ((test_per >= OVERRIDE_FLOOR_PER).astype(int))
            >= 2
        ).astype(int)

        pred_columns: dict[str, np.ndarray] = {}
        pct_superstar_override: dict[str, np.ndarray] = {}
        pct_rising_override: dict[str, np.ndarray] = {}
        for model_name in selected_models:
            key = ("percentile_multiyear_superstar", model_name)
            if key not in evaluated_models:
                continue
            pct_model, pct_threshold = evaluated_models[key]
            pct_proba = pct_model.predict_proba(test_x)[:, 1]
            pct_superstar_override[model_name] = (
                (pct_proba >= pct_threshold) & (all_nba_gate == 1) & (recency_gate == 1)
            ).astype(int)
            pct_rising_override[model_name] = (
                (pct_proba >= pct_threshold) & (all_nba_gate == 0) & (season_all_nba_gate == 1)
            ).astype(int)

        for (name, model_name), (model, threshold) in evaluated_models.items():
            proba = model.predict_proba(test_x)[:, 1]
            meets_threshold = (proba >= threshold).astype(int)
            pred = ((meets_threshold == 1) & (all_nba_gate == 1) & (recency_gate == 1)).astype(int)
            rising_pred = (
                (meets_threshold == 1) & (all_nba_gate == 0) & (season_all_nba_gate == 1)
            ).astype(int)
            if name != "percentile_multiyear_superstar" and model_name in pct_superstar_override:
                pred = np.maximum(pred, pct_superstar_override[model_name] & current_season_floor)
            if name != "percentile_multiyear_superstar" and model_name in pct_rising_override:
                rising_pred = np.maximum(rising_pred, pct_rising_override[model_name] & current_season_floor)

            base = f"{name}__{model_name}"
            pred_columns[f"prob_superstar__{base}"] = np.round(proba, 4)
            pred_columns[f"is_superstar__{base}"] = pred
            pred_columns[f"is_rising_superstar__{base}"] = rising_pred

        year_out = pd.DataFrame(
            {
                "player": test_df["player"].values,
                "year": test_df["year"].astype(int).values,
                "all_nba_1st_count_before_year": test_df["all_nba_1st_count_before_year"].astype(int).values,
                "made_all_nba_that_season": test_df["made_all_nba_that_season"].astype(int).values,
                "made_all_nba_prev3_seasons": test_df["made_all_nba_prev3_seasons"].astype(int).values,
                "all_nba_team_that_season": test_df["all_nba_team_that_season"].values,
                "mvp_winner_any_before_year": pd.to_numeric(
                    test_df.get("mvp_winner_any_before_year"), errors="coerce"
                ).fillna(0).astype(int).values,
                "mvp_top5_prev3_years": pd.to_numeric(
                    test_df.get("mvp_top5_prev3_years"), errors="coerce"
                ).fillna(0).astype(int).values,
                "bpm": pd.to_numeric(test_df["bpm"], errors="coerce").fillna(0.0).round(3).values,
                "ws_per_48": pd.to_numeric(test_df["ws_per_48"], errors="coerce").fillna(0.0).round(3).values,
                "per": pd.to_numeric(test_df["per"], errors="coerce").fillna(0.0).round(3).values,
            }
        )
        for col, vals in pred_columns.items():
            year_out[col] = vals
        all_preds.append(year_out)

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    metric_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    return pred_df, metric_df


def build_presentation(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Convert prediction outputs into final presentation statuses."""
    if pred_df.empty:
        return pred_df

    super_cols = [c for c in pred_df.columns if c.startswith("is_superstar__")]
    rising_cols = [c for c in pred_df.columns if c.startswith("is_rising_superstar__")]
    prob_cols = [c for c in pred_df.columns if c.startswith("prob_superstar__")]
    if not super_cols or not rising_cols:
        return pd.DataFrame()

    p = pred_df.copy()
    p["superstar_votes"] = p[super_cols].sum(axis=1)
    p["rising_superstar_votes"] = p[rising_cols].sum(axis=1)
    p["max_model_probability"] = p[prob_cols].max(axis=1).round(4)
    p["legacy_gate"] = (p["all_nba_1st_count_before_year"] >= 3).astype(int)
    p["recency_gate"] = (p["made_all_nba_prev3_seasons"] == 1).astype(int)
    p["current_impact_gate"] = (
        (
            (p["bpm"] >= OVERRIDE_FLOOR_BPM).astype(int)
            + (p["ws_per_48"] >= OVERRIDE_FLOOR_WS_PER_48).astype(int)
            + (p["per"] >= OVERRIDE_FLOOR_PER).astype(int)
        )
        >= 2
    ).astype(int)

    superstar_status = (
        (p["legacy_gate"] == 1)
        & (p["recency_gate"] == 1)
        & (p["current_impact_gate"] == 1)
        & (p["superstar_votes"] >= SUPERSTAR_MIN_VOTES)
    )
    former_status = (p["legacy_gate"] == 1) & (~superstar_status)
    rising_status = (
        (p["legacy_gate"] == 0)
        & (p["made_all_nba_that_season"] == 1)
        & (p["current_impact_gate"] == 1)
        & (p["rising_superstar_votes"] >= RISING_SUPERSTAR_MIN_VOTES)
    )
    mvp_override_gate = (
        (p["mvp_winner_any_before_year"] == 1) | (p["mvp_top5_prev3_years"] == 1)
    )
    rising_to_superstar = rising_status & mvp_override_gate
    superstar_status = superstar_status | rising_to_superstar
    rising_status = rising_status & (~rising_to_superstar)
    p["status"] = np.where(
        superstar_status,
        "superstar",
        np.where(former_status, "former_superstar", np.where(rising_status, "rising_superstar", "")),
    )
    # Retroactive promotion: if a player is superstar in a later season,
    # convert earlier former_superstar seasons to superstar.
    super_year_max = (
        p.loc[p["status"] == "superstar"]
        .groupby("player")["year"]
        .max()
    )
    if not super_year_max.empty:
        max_super_for_row = p["player"].map(super_year_max)
        retro_mask = (
            (p["status"] == "former_superstar")
            & max_super_for_row.notna()
            & (p["year"] < max_super_for_row)
        )
        p.loc[retro_mask, "status"] = "superstar"
    p = p[p["status"] != ""].copy()
    p = p[
        [
            "player",
            "year",
            "status",
            "all_nba_1st_count_before_year",
            "made_all_nba_prev3_seasons",
            "made_all_nba_that_season",
            "mvp_winner_any_before_year",
            "mvp_top5_prev3_years",
            "current_impact_gate",
            "superstar_votes",
            "rising_superstar_votes",
            "max_model_probability",
        ]
    ].sort_values(
        ["year", "status", "superstar_votes", "rising_superstar_votes", "max_model_probability", "player"],
        ascending=[True, True, False, False, False, True],
    )
    return p


def main() -> None:
    """Run the full script workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_df, metric_df = run_backtest(args)
    range_tag = f"{args.start_year}_{args.end_year}"
    pred_path = args.output_dir / f"superstar_backtest_{range_tag}_predictions.csv"
    metrics_path = args.output_dir / f"superstar_backtest_{range_tag}_metrics.csv"
    present_path = args.output_dir / f"superstar_backtest_{range_tag}_presentation.csv"

    if not pred_df.empty:
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved: {pred_path}")
        presentation = build_presentation(pred_df)
        presentation.to_csv(present_path, index=False)
        print(f"Saved: {present_path}")
    else:
        print("No prediction rows produced for requested years.")

    if not metric_df.empty:
        metric_df.to_csv(metrics_path, index=False)
        print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
