from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "per",
    "ts_pct",
    "usg_pct",
    "bpm",
    "ws_per_48",
    "ws",
    "vorp",
    "g",
    "mp",
    "obpm",
    "dbpm",
    "ast_pct",
    "trb_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "ftr",
    "fg3a_rate",
    "ows",
    "dws",
]

# Current-season floor to prevent legacy-only overrides for declining players.
OVERRIDE_FLOOR_BPM = 5.5
OVERRIDE_FLOOR_WS_PER_48 = 0.18
OVERRIDE_FLOOR_PER = 20.0
SUPERSTAR_MIN_VOTES = 2
RISING_SUPERSTAR_MIN_VOTES = 1

CSV_RENAME_MAP = {
    "year-name": "year_name",
    "Pos": "pos",
    "Age": "age",
    "Tm": "tm",
    "G": "g",
    "MP": "mp",
    "PER": "per",
    "TS%": "ts_pct",
    "3PAr": "fg3a_rate",
    "FTr": "ftr",
    "ORB%": "orb_pct",
    "DRB%": "drb_pct",
    "TRB%": "trb_pct",
    "AST%": "ast_pct",
    "STL%": "stl_pct",
    "BLK%": "blk_pct",
    "TOV%": "tov_pct",
    "USG%": "usg_pct",
    "OWS": "ows",
    "DWS": "dws",
    "WS": "ws",
    "WS/48": "ws_per_48",
    "OBPM": "obpm",
    "DBPM": "dbpm",
    "BPM": "bpm",
    "VORP": "vorp",
}


def normalize_player_name(name: str) -> str:
    """Normalize player name."""
    text = str(name).strip()
    if any(token in text for token in ("Ã", "Ä", "Å", "â")):
        try:
            text = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s*\(.*?\)\s*", " ", text)
    text = re.sub(r"\s+[CFG]$", "", text.strip())
    text = text.replace(".", "")
    text = re.sub(r"[^A-Za-z0-9'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def season_start_series(df: pd.DataFrame) -> pd.Series:
    """Season start series."""
    if "season" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df["season"].astype(str).str.slice(0, 4), errors="coerce")


def year_to_season_str(year_end: pd.Series) -> pd.Series:
    """Year to season str."""
    year_end_int = pd.to_numeric(year_end, errors="coerce").astype("Int64")
    start = (year_end_int - 1).astype("Int64").astype(str)
    end = (year_end_int % 100).astype("Int64").astype(str).str.zfill(2)
    return start + "-" + end


def preprocess_advanced_stats(
    stats_csv: Path,
    all_nba_csv: Path,
    mvp_csv: Path | None = None,
    min_year_end: int = 2003,
    max_year_end: int = 2017,
) -> pd.DataFrame:
    """Load and clean advanced stats, then add join features."""
    df = pd.read_csv(stats_csv).rename(columns=CSV_RENAME_MAP)
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    needed = {"year_name", "year", "tm", "mp"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {stats_csv.name}: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(min_year_end, max_year_end)].copy()
    df["player"] = df["year_name"].astype(str).str.replace(r"^\d{4}-", "", regex=True).str.strip()

    for col in FEATURE_COLS + ["year", "mp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # One row per player-season. Prefer aggregate TOT row for traded players.
    df["_tot_pref"] = (df["tm"].astype(str).str.upper() == "TOT").astype(int)
    df = df.sort_values(
        ["player", "year", "_tot_pref", "mp"], ascending=[True, True, False, False]
    )
    df = df.drop_duplicates(["player", "year"], keep="first").drop(columns="_tot_pref")

    df["season"] = year_to_season_str(df["year"])
    df["award_share"] = 0.0  # Not available in this data source; keep threshold logic compatible.

    first_team_years = build_all_nba_first_team_years(
        all_nba_csv, min_year_end=min_year_end, max_year_end=max_year_end
    )
    season_team_map = build_all_nba_season_team_map(
        all_nba_csv, min_year_end=min_year_end, max_year_end=max_year_end
    )
    all_nba_any_team_years = build_all_nba_any_team_years(
        all_nba_csv, min_year_end=min_year_end, max_year_end=max_year_end
    )
    df["player_norm"] = df["player"].map(normalize_player_name)
    df["all_nba_1st_count_before_year"] = [
        int(sum(1 for yr in first_team_years.get(player_norm, []) if yr < int(year_end)))
        for year_end, player_norm in zip(df["year"], df["player_norm"])
    ]
    df["all_nba_team_that_season"] = [
        season_team_map.get((int(year_end), player_norm), "")
        for year_end, player_norm in zip(df["year"], df["player_norm"])
    ]
    df["made_all_nba_that_season"] = (df["all_nba_team_that_season"] != "").astype(int)
    df["made_all_nba_prev3_seasons"] = [
        int(any((int(year_end) - 3) <= yr < int(year_end) for yr in all_nba_any_team_years.get(player_norm, [])))
        for year_end, player_norm in zip(df["year"], df["player_norm"])
    ]
    df["all_nba_1st_3x_gate"] = (df["all_nba_1st_count_before_year"] >= 3).astype(int)
    df = add_mvp_history_flags(df, mvp_csv)
    return df


def add_mvp_history_flags(df: pd.DataFrame, mvp_csv: Path | None) -> pd.DataFrame:
    """Add mvp history flags."""
    out = df.copy()
    out["mvp_winner_any_before_year"] = 0
    out["mvp_top5_prev3_years"] = 0
    if mvp_csv is None or not mvp_csv.exists():
        return out

    mvp = pd.read_csv(mvp_csv)
    unnamed = [c for c in mvp.columns if c.lower().startswith("unnamed")]
    if unnamed:
        mvp = mvp.drop(columns=unnamed)

    required_cols = {"season", "player"}
    if not required_cols.issubset(mvp.columns):
        return out

    mvp = mvp.copy()
    mvp["player_norm"] = mvp["player"].map(normalize_player_name)
    mvp["season_end"] = pd.to_numeric(mvp["season"].astype(str).str.slice(0, 4), errors="coerce") + 1
    mvp = mvp[mvp["season_end"].notna() & (mvp["player_norm"] != "")]
    if mvp.empty:
        return out
    mvp["season_end"] = mvp["season_end"].astype(int)

    if {"points_won", "points_max"}.issubset(mvp.columns):
        points_won = pd.to_numeric(mvp["points_won"], errors="coerce").fillna(0.0)
        points_max = pd.to_numeric(mvp["points_max"], errors="coerce")
        mvp["award_share"] = np.where(points_max > 0, points_won / points_max, 0.0)
    elif "award_share" in mvp.columns:
        mvp["award_share"] = pd.to_numeric(mvp["award_share"], errors="coerce").fillna(0.0)
    else:
        mvp["award_share"] = 0.0

    if "points_won" in mvp.columns:
        rank_metric = pd.to_numeric(mvp["points_won"], errors="coerce").fillna(0.0)
    else:
        rank_metric = mvp["award_share"]
    mvp["rank_metric"] = rank_metric
    mvp["mvp_rank"] = mvp.groupby("season_end")["rank_metric"].rank(method="min", ascending=False)
    mvp["is_top5"] = (mvp["mvp_rank"] <= 5).astype(int)

    mvp["is_winner"] = (
        (mvp["award_share"] > 0)
        & (mvp["award_share"] == mvp.groupby("season_end")["award_share"].transform("max"))
    ).astype(int)

    winner_years_by_player = (
        mvp[mvp["is_winner"] == 1]
        .groupby("player_norm")["season_end"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .to_dict()
    )
    top5_years_by_player = (
        mvp[mvp["is_top5"] == 1]
        .groupby("player_norm")["season_end"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .to_dict()
    )

    out["mvp_winner_any_before_year"] = [
        int(any(yr <= int(year_end) for yr in winner_years_by_player.get(player_norm, [])))
        for year_end, player_norm in zip(out["year"], out["player_norm"])
    ]
    out["mvp_top5_prev3_years"] = [
        int(any((int(year_end) - 3) <= yr < int(year_end) for yr in top5_years_by_player.get(player_norm, [])))
        for year_end, player_norm in zip(out["year"], out["player_norm"])
    ]
    return out


def build_all_nba_first_team_years(
    all_nba_csv: Path, min_year_end: int, max_year_end: int
) -> dict[str, list[int]]:
    """Build all nba first team years."""
    seasons_by_player: dict[str, list[int]] = {}
    with all_nba_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            season = row[0].strip()
            league = row[1].strip()
            team = row[2].strip()
            if league != "NBA" or team != "1st":
                continue

            match = re.match(r"^(\d{4})-(\d{2})$", season)
            if not match:
                continue
            season_start = int(match.group(1))
            season_end = season_start + 1
            if season_end < min_year_end or season_end > max_year_end:
                continue

            for player_field in row[4:9]:
                player_name = normalize_player_name(player_field)
                if player_name:
                    seasons_by_player.setdefault(player_name, []).append(season_end)

    for player_name, seasons in seasons_by_player.items():
        seasons_by_player[player_name] = sorted(seasons)
    return seasons_by_player


def build_all_nba_season_team_map(
    all_nba_csv: Path, min_year_end: int, max_year_end: int
) -> dict[tuple[int, str], str]:
    """Build all nba season team map."""
    season_team: dict[tuple[int, str], str] = {}
    valid_teams = {"1st", "2nd", "3rd"}
    with all_nba_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            season = row[0].strip()
            league = row[1].strip()
            team = row[2].strip()
            if league != "NBA" or team not in valid_teams:
                continue

            match = re.match(r"^(\d{4})-(\d{2})$", season)
            if not match:
                continue
            season_start = int(match.group(1))
            season_end = season_start + 1
            if season_end < min_year_end or season_end > max_year_end:
                continue

            for player_field in row[4:9]:
                player_name = normalize_player_name(player_field)
                if player_name:
                    season_team[(season_end, player_name)] = team

    return season_team


def build_all_nba_any_team_years(
    all_nba_csv: Path, min_year_end: int, max_year_end: int
) -> dict[str, list[int]]:
    """Build all nba any team years."""
    years_by_player: dict[str, list[int]] = {}
    valid_teams = {"1st", "2nd", "3rd"}
    with all_nba_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            season = row[0].strip()
            league = row[1].strip()
            team = row[2].strip()
            if league != "NBA" or team not in valid_teams:
                continue

            match = re.match(r"^(\d{4})-(\d{2})$", season)
            if not match:
                continue
            season_start = int(match.group(1))
            season_end = season_start + 1
            if season_end < min_year_end or season_end > max_year_end:
                continue

            for player_field in row[4:9]:
                player_name = normalize_player_name(player_field)
                if player_name:
                    years_by_player.setdefault(player_name, []).append(season_end)

    for player_name, years in years_by_player.items():
        years_by_player[player_name] = sorted(set(years))
    return years_by_player


def build_multiyear_train_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe historical features for training rows."""
    if "player" not in df.columns or "season" not in df.columns:
        raise ValueError("Multi-year features require 'player' and 'season' columns in training data.")

    tmp = df[["player", "season"] + FEATURE_COLS].copy()
    tmp["season_start"] = season_start_series(tmp)
    tmp["orig_idx"] = tmp.index
    tmp = tmp.sort_values(["player", "season_start", "orig_idx"], kind="mergesort")

    multi_cols: list[str] = []
    grouped = tmp.groupby("player", sort=False)
    for col in FEATURE_COLS:
        col_prev2 = f"{col}_prev2_mean"
        col_prev3 = f"{col}_prev3_mean"
        tmp[col_prev2] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window=2, min_periods=2).mean()
        )
        tmp[col_prev3] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=3).mean()
        )
        multi_cols.extend([col_prev2, col_prev3])

    return tmp[multi_cols].reindex(df.index)


def build_multiyear_test_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, multi_feature_cols: list[str]
) -> pd.DataFrame:
    """Create matching historical features for test rows only."""
    if "player" not in test_df.columns or "player" not in train_df.columns:
        return pd.DataFrame(index=test_df.index, columns=multi_feature_cols, dtype=float)

    hist = train_df[["player", "season"] + FEATURE_COLS].copy()
    hist["season_start"] = season_start_series(hist)
    hist["orig_idx"] = hist.index
    hist = hist.sort_values(["player", "season_start", "orig_idx"], kind="mergesort")

    player_to_vals: dict[str, dict[str, float]] = {}
    for player, grp in hist.groupby("player", sort=False):
        row_vals: dict[str, float] = {}
        for col in FEATURE_COLS:
            s = grp[col].dropna()
            row_vals[f"{col}_prev2_mean"] = float(s.tail(2).mean()) if len(s) >= 2 else np.nan
            row_vals[f"{col}_prev3_mean"] = float(s.tail(3).mean()) if len(s) >= 3 else np.nan
        player_to_vals[player] = row_vals

    out = pd.DataFrame(index=test_df.index, columns=multi_feature_cols, dtype=float)
    for idx, player in test_df["player"].items():
        vals = player_to_vals.get(player)
        if vals is None:
            continue
        for c in multi_feature_cols:
            out.at[idx, c] = vals[c]

    return out


def safe_numeric_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Safe numeric col."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def label_map(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build all label options from the prepared stats table."""
    labels: dict[str, pd.Series] = {}

    award_share = safe_numeric_col(df, "award_share", default=0.0)
    bpm = safe_numeric_col(df, "bpm", default=0.0)
    ws_per_48 = safe_numeric_col(df, "ws_per_48", default=0.0)
    usg_pct = safe_numeric_col(df, "usg_pct", default=0.0)
    per = safe_numeric_col(df, "per", default=0.0)
    ts_pct = safe_numeric_col(df, "ts_pct", default=0.0)

    labels["mvp_share_0_10"] = (award_share >= 0.10).astype(int)
    labels["mvp_share_0_20"] = (award_share >= 0.20).astype(int)
    labels["mvp_share_0_30"] = (award_share >= 0.30).astype(int)

    labels["elite_bpm_ws"] = ((bpm >= 6.0) & (ws_per_48 >= 0.200)).astype(int)
    labels["usage_impact_combo"] = (
        (usg_pct >= 30.0) & (ws_per_48 >= 0.200) & (bpm >= 6.0)
    ).astype(int)

    labels["hybrid_share_or_impact"] = (
        (award_share >= 0.15) | ((bpm >= 7.0) & (ws_per_48 >= 0.220))
    ).astype(int)

    balanced_gate = (award_share >= 0.20) | (
        (bpm >= 7.0) & (ws_per_48 >= 0.220) & (per >= 24.0)
    )
    labels["balanced_gate_single_season"] = balanced_gate.astype(int)

    if "player" in df.columns and "season" in df.columns:
        tmp = df[["player", "season"]].copy()
        tmp["balanced_gate"] = balanced_gate.astype(int)
        tmp["orig_idx"] = df.index
        tmp["season_start"] = pd.to_numeric(
            tmp["season"].astype(str).str.slice(0, 4), errors="coerce"
        )
        tmp = tmp.sort_values(["player", "season_start", "orig_idx"])

        rolling_counts = (
            tmp.groupby("player", sort=False)["balanced_gate"]
            .rolling(window=3, min_periods=3)
            .sum()
            .reset_index(level=0, drop=True)
        )
        tmp["balanced_2_of_last_3"] = (rolling_counts.fillna(0) >= 2).astype(int)
        multi_year = ((tmp["balanced_2_of_last_3"] == 1) | (award_share.loc[tmp.index] >= 0.35)).astype(int)
        labels["balanced_multiyear_2of3_or_breakout"] = (
            pd.Series(multi_year, index=tmp.index).reindex(df.index).fillna(0).astype(int)
        )
    else:
        labels["balanced_multiyear_2of3_or_breakout"] = balanced_gate.astype(int)

    if "season" in df.columns and "player" in df.columns:
        tmp = df.copy()
        tmp["season_start"] = pd.to_numeric(
            tmp["season"].astype(str).str.slice(0, 4), errors="coerce"
        )
        tmp = tmp.sort_values(["player", "season_start", "season"], kind="mergesort")

        metrics = ["bpm", "ws_per_48", "ts_pct", "usg_pct"]
        for col in metrics:
            if col not in tmp.columns:
                raise ValueError(f"Missing required column for percentile label: {col}")
            tmp[f"{col}_p90"] = tmp.groupby("season_start")[col].transform(lambda s: s.quantile(0.90))
            tmp[f"elite_{col}"] = (tmp[col] >= tmp[f"{col}_p90"]).astype(int)

        tmp["elite_season_percentile"] = (
            (tmp["elite_bpm"] == 1)
            & (tmp["elite_ws_per_48"] == 1)
            & ((tmp["elite_ts_pct"] == 1) | (tmp["elite_usg_pct"] == 1))
        ).astype(int)

        rolling_counts = (
            tmp.groupby("player", sort=False)["elite_season_percentile"]
            .rolling(window=3, min_periods=3)
            .sum()
            .reset_index(level=0, drop=True)
        )
        tmp["elite_2_of_last_3"] = (rolling_counts.fillna(0) >= 2).astype(int)
        tmp["breakout_override"] = (award_share.reindex(tmp.index).fillna(0) >= 0.20).astype(int)
        labels["percentile_multiyear_superstar"] = (
            ((tmp["elite_2_of_last_3"] == 1) | (tmp["breakout_override"] == 1))
            .reindex(df.index)
            .fillna(0)
            .astype(int)
        )

    all_nba_gate = (safe_numeric_col(df, "all_nba_1st_count_before_year", default=0.0) >= 3).astype(int)
    recency_gate = (safe_numeric_col(df, "made_all_nba_prev3_seasons", default=0.0) >= 1).astype(int)
    labels["all_nba_1st_3x_gate"] = (all_nba_gate & recency_gate).astype(int)

    # Superstar requires threshold label + prior 3x 1st-team gate + recent All-NBA signal.
    for name in list(labels.keys()):
        if name == "all_nba_1st_3x_gate":
            continue
        labels[name] = (labels[name].astype(int) & all_nba_gate & recency_gate).astype(int)

    return labels


def build_model(model_name: str) -> Pipeline:
    """Build model."""
    base_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]

    if model_name == "logreg":
        base_steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    elif model_name == "random_forest":
        base_steps.append(
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )
    elif model_name == "gradient_boosting":
        base_steps.append(
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            )
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(steps=base_steps)


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Best threshold."""
    candidates = np.round(np.arange(0.20, 0.81, 0.02), 2)
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        pred = (proba >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def threshold_for_target_positive_rate(proba: np.ndarray, target_rate: float) -> float:
    """Threshold for target positive rate."""
    target = float(np.clip(target_rate, 0.01, 0.50))
    q = float(np.quantile(proba, 1.0 - target))
    return float(np.clip(np.round(q, 2), 0.10, 0.90))


def safe_stratify_target(y: pd.Series) -> pd.Series | None:
    """Safe stratify target."""
    counts = y.value_counts(dropna=False)
    if counts.empty or counts.min() < 2:
        return None
    return y


def stratified_cv_summary(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict[str, float]:
    """Estimate cross-validated performance for one model setup."""
    counts = y.value_counts(dropna=False)
    if y.nunique() < 2 or counts.min() < 2:
        return {
            "cv_f1_mean": np.nan,
            "cv_f1_std": np.nan,
            "cv_roc_auc_mean": np.nan,
            "cv_roc_auc_std": np.nan,
            "cv_precision_mean": np.nan,
            "cv_recall_mean": np.nan,
            "cv_accuracy_mean": np.nan,
        }

    n_splits = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = build_model(model_name)
    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "f1": "f1",
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "accuracy": "accuracy",
        },
        n_jobs=-1,
    )

    return {
        "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
        "cv_f1_std": float(np.std(cv_scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
        "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
    }


def bootstrap_ci(
    y_true: pd.Series, proba: np.ndarray, threshold: float, n_boot: int = 500, seed: int = 42
) -> dict[str, float]:
    """Bootstrap ci."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true)
    n = len(y_arr)
    f1_vals: list[float] = []
    auc_vals: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_arr[idx]
        p_b = proba[idx]
        pred_b = (p_b >= threshold).astype(int)
        f1_vals.append(float(f1_score(y_b, pred_b, zero_division=0)))
        if np.unique(y_b).size >= 2:
            auc_vals.append(float(roc_auc_score(y_b, p_b)))

    out = {
        "f1_ci_low": float(np.quantile(f1_vals, 0.025)),
        "f1_ci_high": float(np.quantile(f1_vals, 0.975)),
    }
    if auc_vals:
        out["roc_auc_ci_low"] = float(np.quantile(auc_vals, 0.025))
        out["roc_auc_ci_high"] = float(np.quantile(auc_vals, 0.975))
    else:
        out["roc_auc_ci_low"] = np.nan
        out["roc_auc_ci_high"] = np.nan
    return out


def threshold_sweep(y_true: pd.Series, proba: np.ndarray, center_t: float) -> dict[str, float]:
    """Threshold sweep."""
    low = max(0.10, center_t - 0.10)
    high = min(0.90, center_t + 0.10)
    thresholds = np.round(np.arange(low, high + 1e-9, 0.02), 2)
    f1_vals = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds]
    return {
        "sweep_t_low": float(low),
        "sweep_t_high": float(high),
        "sweep_f1_min": float(np.min(f1_vals)),
        "sweep_f1_max": float(np.max(f1_vals)),
        "sweep_f1_range": float(np.max(f1_vals) - np.min(f1_vals)),
    }


def temporal_train_val_test_split(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Temporal train val test split."""
    if "season" not in df.columns:
        raise ValueError("Temporal split requires a 'season' column.")

    season_start = pd.to_numeric(df["season"].astype(str).str.slice(0, 4), errors="coerce")
    valid_mask = season_start.notna()
    if valid_mask.sum() < len(df):
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        season_start = season_start.loc[valid_mask]

    unique_seasons = np.array(sorted(season_start.unique()))
    if len(unique_seasons) < 3:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=safe_stratify_target(y),
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=42,
            stratify=safe_stratify_target(y_train_val),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    test_start = unique_seasons[-1]
    val_start = unique_seasons[-2]
    train_mask = season_start < val_start
    val_mask = season_start == val_start
    test_mask = season_start >= test_start

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    if (
        len(X_train) == 0
        or len(X_val) == 0
        or len(X_test) == 0
        or y_train.nunique() < 2
        or y_val.nunique() < 2
        or y_test.nunique() < 2
    ):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=safe_stratify_target(y),
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=42,
            stratify=safe_stratify_target(y_train_val),
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_option(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    option_name: str,
    model_name: str,
    target_positive_rate: float,
    threshold_override: float | None = None,
    single_threshold_only: bool = False,
) -> tuple[list[dict[str, float]], Pipeline, float] | tuple[None, None, None]:
    """Fit one model/label option and return thresholded metrics."""
    if y.nunique() < 2:
        return None, None, None

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(df, X, y)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return None, None, None

    model = build_model(model_name)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    f1_threshold = best_threshold(y_val, val_proba)
    business_threshold = threshold_for_target_positive_rate(val_proba, target_positive_rate)
    selected_threshold = (
        float(np.clip(threshold_override, 0.10, 0.90))
        if threshold_override is not None
        else business_threshold
    )

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    model.fit(X_train_val, y_train_val)
    proba = model.predict_proba(X_test)[:, 1]
    def build_row(threshold: float, threshold_mode: str) -> dict[str, float]:
        """Build row."""
        pred = (proba >= threshold).astype(int)
        out = {
            "label_option": option_name,
            "model": model_name,
            "threshold_mode": threshold_mode,
            "n_rows": float(len(X)),
            "positive_rate": float(y.mean()),
            "target_positive_rate": float(target_positive_rate),
            "threshold": threshold,
            "threshold_business": business_threshold,
            "threshold_selected": selected_threshold,
            "threshold_f1_opt": f1_threshold,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() >= 2 else np.nan,
        }
        out.update(stratified_cv_summary(X, y, model_name))
        out.update(threshold_sweep(y_test, proba, threshold))
        out.update(bootstrap_ci(y_test, proba, threshold))
        return out

    rows = [build_row(selected_threshold, "selected")]
    if (not single_threshold_only) and (abs(selected_threshold - business_threshold) > 1e-9):
        rows.append(build_row(business_threshold, "business_target"))
    return rows, model, selected_threshold


def parse_args() -> argparse.Namespace:
    """Read CLI arguments for this run."""
    parser = argparse.ArgumentParser(
        description="Compare models for superstar label definitions on 2003-2017 advanced stats."
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=Path("NBA_Advanced_Stats_2002-2022.csv"),
        help="Path to NBA advanced stats CSV.",
    )
    parser.add_argument(
        "--all-nba-csv",
        type=Path,
        default=Path("all_nba_selections.csv"),
        help="Path to All-NBA selections CSV.",
    )
    parser.add_argument(
        "--mvp-csv",
        type=Path,
        default=Path("mvp_votings.csv"),
        help="Path to MVP voting CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eda_outputs"),
        help="Directory to write evaluation and prediction CSV files.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="all",
        help="Comma-separated label options to run, or 'all'.",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="Print available label options and exit.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names to run (logreg,random_forest,gradient_boosting) or 'all'.",
    )
    parser.add_argument(
        "--target-positive-rate",
        type=float,
        default=0.15,
        help="Business threshold target: fraction of samples predicted positive (default 0.15).",
    )
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


def main() -> None:
    """Run the full script workflow."""
    args = parse_args()

    train_df = preprocess_advanced_stats(args.stats_csv, args.all_nba_csv, args.mvp_csv)
    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in {args.stats_csv.name}: {missing}")

    train_multi_X = build_multiyear_train_features(train_df)
    train_valid_mask = train_multi_X.notna().any(axis=1)
    train_multi_X = train_multi_X.loc[train_valid_mask].copy()
    train_eval_df = train_df.loc[train_valid_mask].copy()

    option_to_y = label_map(train_eval_df)

    if args.list_options:
        print("Available label options:")
        for name in option_to_y:
            print(f"- {name}")
        return

    if args.labels.strip().lower() == "all":
        selected = list(option_to_y.keys())
    else:
        selected = [s.strip() for s in args.labels.split(",") if s.strip()]

    invalid = [name for name in selected if name not in option_to_y]
    if invalid:
        raise ValueError(f"Unknown label options: {invalid}")

    available_models = ["logreg", "random_forest", "gradient_boosting"]
    if args.models.strip().lower() == "all":
        selected_models = available_models
    else:
        selected_models = [s.strip() for s in args.models.split(",") if s.strip()]

    invalid_models = [m for m in selected_models if m not in available_models]
    if invalid_models:
        raise ValueError(f"Unknown models: {invalid_models}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    pred_columns: dict[str, np.ndarray] = {}
    evaluated_models: dict[tuple[str, str], tuple[Pipeline, float]] = {}

    latest_year = int(train_df["year"].max())
    test_df = train_df[train_df["year"] == latest_year].copy()
    hist_df = train_df[train_df["year"] < latest_year].copy()
    test_X = build_multiyear_test_features(hist_df, test_df, list(train_multi_X.columns))
    has_test_features = not test_X.empty and test_X.notna().any(axis=1).any()

    for name in selected:
        y = option_to_y[name]
        for model_name in selected_models:
            metric_rows, model, threshold = evaluate_option(
                train_eval_df,
                train_multi_X,
                y,
                name,
                model_name,
                target_positive_rate=args.target_positive_rate,
                threshold_override=args.threshold_override,
                single_threshold_only=args.single_threshold_only,
            )
            if metric_rows is None or model is None or threshold is None:
                print(f"Skipping {name}/{model_name}: label split is degenerate.")
                continue

            rows.extend(metric_rows)
            evaluated_models[(name, model_name)] = (model, threshold)

    if not rows:
        raise RuntimeError("No valid label options were evaluated.")

    results = pd.DataFrame(rows).sort_values(by=["f1", "roc_auc"], ascending=False)
    results_path = args.output_dir / "superstar_label_option_results.csv"
    results.to_csv(results_path, index=False)

    print("\nEvaluation summary (sorted by F1 then ROC-AUC):")
    print(results.to_string(index=False))
    print(f"\nSaved: {results_path}")

    if has_test_features and evaluated_models:
        all_nba_gate = test_df["all_nba_1st_3x_gate"].to_numpy(dtype=int)
        recency_gate = pd.to_numeric(
            test_df.get("made_all_nba_prev3_seasons"), errors="coerce"
        ).fillna(0.0).to_numpy(dtype=int)
        season_all_nba_gate = pd.to_numeric(
            test_df.get("made_all_nba_that_season"), errors="coerce"
        ).fillna(0.0).to_numpy(dtype=int)
        test_bpm = pd.to_numeric(test_df.get("bpm"), errors="coerce").fillna(0.0).to_numpy()
        test_ws48 = pd.to_numeric(test_df.get("ws_per_48"), errors="coerce").fillna(0.0).to_numpy()
        test_per = pd.to_numeric(test_df.get("per"), errors="coerce").fillna(0.0).to_numpy()
        current_season_floor = (
            ((test_bpm >= OVERRIDE_FLOOR_BPM).astype(int))
            + ((test_ws48 >= OVERRIDE_FLOOR_WS_PER_48).astype(int))
            + ((test_per >= OVERRIDE_FLOOR_PER).astype(int))
            >= 2
        ).astype(int)
        percentile_override_superstar: dict[str, np.ndarray] = {}
        percentile_override_rising: dict[str, np.ndarray] = {}

        for model_name in selected_models:
            key = ("percentile_multiyear_superstar", model_name)
            if key not in evaluated_models:
                continue
            pct_model, pct_threshold = evaluated_models[key]
            pct_proba = pct_model.predict_proba(test_X)[:, 1]
            percentile_override_superstar[model_name] = (
                (pct_proba >= pct_threshold) & (all_nba_gate == 1) & (recency_gate == 1)
            ).astype(int)
            percentile_override_rising[model_name] = (
                (pct_proba >= pct_threshold) & (all_nba_gate == 0) & (season_all_nba_gate == 1)
            ).astype(int)

        for (name, model_name), (model, threshold) in evaluated_models.items():
            proba = model.predict_proba(test_X)[:, 1]
            meets_threshold = (proba >= threshold).astype(int)
            pred = ((meets_threshold == 1) & (all_nba_gate == 1) & (recency_gate == 1)).astype(int)
            rising_pred = (
                (meets_threshold == 1) & (all_nba_gate == 0) & (season_all_nba_gate == 1)
            ).astype(int)
            if name != "percentile_multiyear_superstar" and model_name in percentile_override_superstar:
                pred = np.maximum(pred, percentile_override_superstar[model_name] & current_season_floor)
            if name != "percentile_multiyear_superstar" and model_name in percentile_override_rising:
                rising_pred = np.maximum(
                    rising_pred, percentile_override_rising[model_name] & current_season_floor
                )

            col_base = f"{name}__{model_name}"
            pred_columns[f"prob_superstar__{col_base}"] = np.round(proba, 4)
            pred_columns[f"is_superstar__{col_base}"] = pred
            pred_columns[f"is_rising_superstar__{col_base}"] = rising_pred

    pred_out = pd.DataFrame()
    pred_out["player"] = test_df["player"].values
    pred_out["year"] = test_df["year"].astype(int).values
    pred_out["all_nba_1st_count_before_year"] = test_df["all_nba_1st_count_before_year"].values
    pred_out["made_all_nba_that_season"] = test_df["made_all_nba_that_season"].astype(int).values
    pred_out["made_all_nba_prev3_seasons"] = test_df["made_all_nba_prev3_seasons"].astype(int).values
    pred_out["all_nba_team_that_season"] = test_df["all_nba_team_that_season"].values
    pred_out["mvp_winner_any_before_year"] = (
        pd.to_numeric(test_df.get("mvp_winner_any_before_year"), errors="coerce").fillna(0).astype(int).values
    )
    pred_out["mvp_top5_prev3_years"] = (
        pd.to_numeric(test_df.get("mvp_top5_prev3_years"), errors="coerce").fillna(0).astype(int).values
    )
    pred_out["bpm"] = pd.to_numeric(test_df.get("bpm"), errors="coerce").fillna(0.0).round(3).values
    pred_out["ws_per_48"] = (
        pd.to_numeric(test_df.get("ws_per_48"), errors="coerce").fillna(0.0).round(3).values
    )
    pred_out["per"] = pd.to_numeric(test_df.get("per"), errors="coerce").fillna(0.0).round(3).values
    for col, values in pred_columns.items():
        pred_out[col] = values

    if not pred_out.empty and pred_columns:
        pred_path = args.output_dir / "superstar_test_predictions.csv"
        pred_out.to_csv(pred_path, index=False)
        print(f"Saved: {pred_path}")

        super_cols = [c for c in pred_out.columns if c.startswith("is_superstar__")]
        rising_cols = [c for c in pred_out.columns if c.startswith("is_rising_superstar__")]
        prob_cols = [c for c in pred_out.columns if c.startswith("prob_superstar__")]
        if super_cols and rising_cols:
            presentation = pred_out.copy()
            presentation["superstar_votes"] = presentation[super_cols].sum(axis=1)
            presentation["rising_superstar_votes"] = presentation[rising_cols].sum(axis=1)
            presentation["max_model_probability"] = presentation[prob_cols].max(axis=1).round(4)
            presentation["legacy_gate"] = (presentation["all_nba_1st_count_before_year"] >= 3).astype(int)
            presentation["recency_gate"] = (presentation["made_all_nba_prev3_seasons"] == 1).astype(int)
            presentation["current_impact_gate"] = (
                (
                    (presentation["bpm"] >= OVERRIDE_FLOOR_BPM).astype(int)
                    + (presentation["ws_per_48"] >= OVERRIDE_FLOOR_WS_PER_48).astype(int)
                    + (presentation["per"] >= OVERRIDE_FLOOR_PER).astype(int)
                )
                >= 2
            ).astype(int)
            superstar_status = (
                (presentation["legacy_gate"] == 1)
                & (presentation["recency_gate"] == 1)
                & (presentation["current_impact_gate"] == 1)
                & (presentation["superstar_votes"] >= SUPERSTAR_MIN_VOTES)
            )
            former_status = (presentation["legacy_gate"] == 1) & (~superstar_status)
            rising_status = (
                (presentation["legacy_gate"] == 0)
                & (presentation["made_all_nba_that_season"] == 1)
                & (presentation["current_impact_gate"] == 1)
                & (presentation["rising_superstar_votes"] >= RISING_SUPERSTAR_MIN_VOTES)
            )
            mvp_override_gate = (
                (presentation["mvp_winner_any_before_year"] == 1)
                | (presentation["mvp_top5_prev3_years"] == 1)
            )
            rising_to_superstar = rising_status & mvp_override_gate
            superstar_status = superstar_status | rising_to_superstar
            rising_status = rising_status & (~rising_to_superstar)
            presentation["status"] = np.where(
                superstar_status,
                "superstar",
                np.where(former_status, "former_superstar", np.where(rising_status, "rising_superstar", "")),
            )
            # Retroactive promotion: if a player is superstar in a later season,
            # convert earlier former_superstar seasons to superstar.
            super_year_max = (
                presentation.loc[presentation["status"] == "superstar"]
                .groupby("player")["year"]
                .max()
            )
            if not super_year_max.empty:
                max_super_for_row = presentation["player"].map(super_year_max)
                retro_mask = (
                    (presentation["status"] == "former_superstar")
                    & max_super_for_row.notna()
                    & (presentation["year"] < max_super_for_row)
                )
                presentation.loc[retro_mask, "status"] = "superstar"
            presentation = presentation[presentation["status"] != ""].copy()
            presentation = presentation[
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
                ["status", "superstar_votes", "rising_superstar_votes", "max_model_probability", "player"],
                ascending=[True, False, False, False, True],
            )

            present_path = args.output_dir / "superstar_rising_for_presentation.csv"
            presentation.to_csv(present_path, index=False)
            print(f"Saved: {present_path}")


if __name__ == "__main__":
    main()
