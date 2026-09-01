from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import superstar_logreg_label_tests as slt

try:
    import seaborn as sns

    HAS_SEABORN = True
except ModuleNotFoundError:
    HAS_SEABORN = False


CSV_RENAME_MAP = slt.CSV_RENAME_MAP
FEATURE_COLS = slt.FEATURE_COLS


def cleaning_audit(
    stats_csv: Path, min_year_end: int, max_year_end: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cleaning audit."""
    raw = pd.read_csv(stats_csv)
    steps: list[dict[str, float | int | str]] = []

    steps.append(
        {
            "step": "raw_load",
            "rows_before": int(len(raw)),
            "rows_after": int(len(raw)),
            "rows_removed": 0,
            "notes": "Initial file load.",
        }
    )

    df = raw.rename(columns=CSV_RENAME_MAP).copy()
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    before = len(df)
    if unnamed:
        df = df.drop(columns=unnamed)
    steps.append(
        {
            "step": "drop_unnamed_columns",
            "rows_before": int(before),
            "rows_after": int(len(df)),
            "rows_removed": int(before - len(df)),
            "notes": f"Dropped columns: {unnamed}" if unnamed else "No unnamed columns found.",
        }
    )

    before = len(df)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(min_year_end, max_year_end)].copy()
    steps.append(
        {
            "step": "filter_year_window",
            "rows_before": int(before),
            "rows_after": int(len(df)),
            "rows_removed": int(before - len(df)),
            "notes": f"Kept seasons where year in [{min_year_end}, {max_year_end}].",
        }
    )

    df["player"] = (
        df["year_name"].astype(str).str.replace(r"^\d{4}-", "", regex=True).str.strip()
    )

    numeric_cols = [c for c in FEATURE_COLS + ["year", "mp"] if c in df.columns]
    missing_before = df[numeric_cols].isna().sum().rename("missing_before")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_after = df[numeric_cols].isna().sum().rename("missing_after")
    coercion_report = pd.concat([missing_before, missing_after], axis=1)
    coercion_report["new_missing_from_numeric_cast"] = (
        coercion_report["missing_after"] - coercion_report["missing_before"]
    ).astype(int)
    coercion_report = coercion_report.sort_values(
        "new_missing_from_numeric_cast", ascending=False
    )

    before = len(df)
    dup_groups_before = int(df.duplicated(["player", "year"], keep=False).sum())
    df["_tot_pref"] = (df["tm"].astype(str).str.upper() == "TOT").astype(int)
    df = df.sort_values(
        ["player", "year", "_tot_pref", "mp"], ascending=[True, True, False, False]
    )
    df = df.drop_duplicates(["player", "year"], keep="first").drop(columns="_tot_pref")
    steps.append(
        {
            "step": "dedupe_player_season",
            "rows_before": int(before),
            "rows_after": int(len(df)),
            "rows_removed": int(before - len(df)),
            "notes": (
                "One row per player-season; keep TOT rows first, then highest minutes. "
                f"Rows in duplicate player-year groups before dedupe: {dup_groups_before}."
            ),
        }
    )

    missing_report = (
        df.isna().sum().sort_values(ascending=False).rename("missing_count").to_frame()
    )
    missing_report["missing_pct"] = (missing_report["missing_count"] / len(df)).round(4)

    return df, pd.DataFrame(steps), coercion_report, missing_report


def build_label_eda(
    df: pd.DataFrame, selected_labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build label eda."""
    label_dict = slt.label_map(df)
    missing = [l for l in selected_labels if l not in label_dict]
    if missing:
        raise ValueError(f"Unknown labels requested: {missing}")

    label_frame = pd.DataFrame({name: label_dict[name].astype(int) for name in selected_labels})
    prevalence = pd.DataFrame(
        {
            "label": selected_labels,
            "positive_count": [int(label_frame[c].sum()) for c in selected_labels],
            "positive_rate": [float(label_frame[c].mean()) for c in selected_labels],
            "n_rows": [int(len(df)) for _ in selected_labels],
        }
    ).sort_values("positive_rate", ascending=False)

    key_metrics = [c for c in ["bpm", "ws_per_48", "per", "usg_pct", "ts_pct"] if c in df.columns]
    summary_rows: list[dict[str, float | str]] = []
    corr_rows: list[dict[str, float | str]] = []

    for label_name in selected_labels:
        y = label_frame[label_name]
        for metric in key_metrics:
            x = pd.to_numeric(df[metric], errors="coerce")
            pos = x[y == 1]
            neg = x[y == 0]
            summary_rows.append(
                {
                    "label": label_name,
                    "metric": metric,
                    "mean_positive": float(pos.mean()) if len(pos) else np.nan,
                    "mean_negative": float(neg.mean()) if len(neg) else np.nan,
                    "median_positive": float(pos.median()) if len(pos) else np.nan,
                    "median_negative": float(neg.median()) if len(neg) else np.nan,
                    "mean_gap_pos_minus_neg": float(pos.mean() - neg.mean())
                    if len(pos) and len(neg)
                    else np.nan,
                }
            )

            valid = x.notna() & y.notna()
            corr_val = x[valid].corr(y[valid]) if valid.sum() > 2 else np.nan
            corr_rows.append(
                {"label": label_name, "metric": metric, "pearson_corr_with_label": corr_val}
            )

    feature_summary = pd.DataFrame(summary_rows).sort_values(
        ["label", "mean_gap_pos_minus_neg"], ascending=[True, False]
    )
    corr_summary = pd.DataFrame(corr_rows).sort_values(
        ["label", "pearson_corr_with_label"], ascending=[True, False]
    )
    return prevalence, feature_summary, corr_summary


def save_plots(df: pd.DataFrame, prevalence: pd.DataFrame, output_dir: Path) -> None:
    """Save plots."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(prevalence["label"], prevalence["positive_rate"], color="#1f77b4")
    ax.set_ylabel("Positive Rate")
    ax.set_title("Label Prevalence")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_label_prevalence.png", dpi=180)
    plt.close(fig)

    plot_metrics = [c for c in ["bpm", "ws_per_48", "per"] if c in df.columns]
    if not plot_metrics:
        return

    first_label = str(prevalence.iloc[0]["label"])
    y = slt.label_map(df)[first_label].astype(int)
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5.2 * len(plot_metrics), 4.4))
    axes = np.array(axes).reshape(-1)
    for i, metric in enumerate(plot_metrics):
        frame = pd.DataFrame({metric: pd.to_numeric(df[metric], errors="coerce"), "label": y})
        frame = frame.dropna(subset=[metric])
        if HAS_SEABORN:
            sns.kdeplot(
                data=frame,
                x=metric,
                hue="label",
                common_norm=False,
                fill=True,
                alpha=0.3,
                ax=axes[i],
            )
        else:
            axes[i].hist(
                frame.loc[frame["label"] == 0, metric],
                bins=30,
                alpha=0.6,
                label="label=0",
                color="#808080",
            )
            axes[i].hist(
                frame.loc[frame["label"] == 1, metric],
                bins=30,
                alpha=0.6,
                label="label=1",
                color="#1f77b4",
            )
            axes[i].legend()
        axes[i].set_title(f"{metric} by {first_label}")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_metric_distributions_by_label.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the full script workflow."""
    parser = argparse.ArgumentParser(
        description="Cleaning audit + EDA report generator for the superstar project."
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=Path("NBA_Advanced_Stats_2002-2022.csv"),
    )
    parser.add_argument(
        "--all-nba-csv",
        type=Path,
        default=Path("all_nba_selections.csv"),
    )
    parser.add_argument(
        "--mvp-csv",
        type=Path,
        default=Path("mvp_votings.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eda_outputs"),
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="balanced_multiyear_2of3_or_breakout,elite_bpm_ws",
        help="Comma-separated labels to summarize.",
    )
    parser.add_argument("--min-year", type=int, default=2003)
    parser.add_argument("--max-year", type=int, default=2017)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, steps, coercion_report, missing_report = cleaning_audit(
        args.stats_csv, args.min_year, args.max_year
    )
    steps.to_csv(args.output_dir / "eda_cleaning_steps_summary.csv", index=False)
    coercion_report.to_csv(args.output_dir / "eda_numeric_cast_missing_delta.csv")
    missing_report.to_csv(args.output_dir / "eda_missingness_after_cleaning.csv")

    model_df = slt.preprocess_advanced_stats(
        args.stats_csv,
        args.all_nba_csv,
        args.mvp_csv,
        min_year_end=args.min_year,
        max_year_end=args.max_year,
    )
    multi_X = slt.build_multiyear_train_features(model_df)
    valid_mask = multi_X.notna().any(axis=1)
    model_df = model_df.loc[valid_mask].copy()

    selected_labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    prevalence, feature_summary, corr_summary = build_label_eda(model_df, selected_labels)
    prevalence.to_csv(args.output_dir / "eda_label_prevalence.csv", index=False)
    feature_summary.to_csv(args.output_dir / "eda_feature_summary_by_label.csv", index=False)
    corr_summary.to_csv(args.output_dir / "eda_feature_label_correlations.csv", index=False)

    save_plots(model_df, prevalence, args.output_dir)

    print("Saved EDA report artifacts:")
    print("- eda_cleaning_steps_summary.csv")
    print("- eda_numeric_cast_missing_delta.csv")
    print("- eda_missingness_after_cleaning.csv")
    print("- eda_label_prevalence.csv")
    print("- eda_feature_summary_by_label.csv")
    print("- eda_feature_label_correlations.csv")
    print("- eda_label_prevalence.png")
    print("- eda_metric_distributions_by_label.png")


if __name__ == "__main__":
    main()
