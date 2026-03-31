import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREFERRED_SHEETS = ["RAW", "Raw", "raw", "Sheet1"]
DATE_HINTS = [
    "date",
    "snapshot",
    "created",
    "closed",
    "due",
    "release",
    "month",
    "year",
]
FLAG_VALUES_TRUE = {"yes", "y", "true", "1", "x"}
FLAG_VALUES_FALSE = {"no", "n", "false", "0", ""}


def auto_find_dataset(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    here = Path(__file__).resolve().parent
    preferred_names = [
        "ISRM Report (Primary Data Source).xlsx",
        "ISRM Report (Primary Data Source).csv",
    ]

    for name in preferred_names:
        candidate = here / name
        if candidate.exists():
            return candidate

    for pattern in ("*.xlsx", "*.xlsm", "*.xls", "*.csv"):
        files = sorted(here.glob(pattern))
        if files:
            return files[0]

    raise FileNotFoundError(
        "No dataset found in this folder. Put the Excel/CSV file beside the script or pass the file path."
    )


def choose_sheet(xls: pd.ExcelFile) -> str:
    for preferred in PREFERRED_SHEETS:
        if preferred in xls.sheet_names:
            return preferred
    return xls.sheet_names[0]


def load_dataset(path: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        return df, "CSV", ["CSV"]

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet_name = choose_sheet(xls)
        df = pd.read_excel(path, sheet_name=sheet_name)
        return df, sheet_name, xls.sheet_names

    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    for col in df.columns:
        col_str = str(col).strip().replace("\n", " ").replace("\r", " ")
        while "  " in col_str:
            col_str = col_str.replace("  ", " ")
        cleaned.append(col_str)
    df.columns = cleaned
    return df


def maybe_parse_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    parsed_cols = []

    for col in df.columns:
        col_lower = col.lower()
        should_try = any(hint in col_lower for hint in DATE_HINTS)

        if not should_try:
            continue

        parsed = pd.to_datetime(df[col], errors="coerce")
        non_null_original = df[col].notna().sum()
        non_null_parsed = parsed.notna().sum()

        if non_null_original > 0 and non_null_parsed >= max(3, int(non_null_original * 0.4)):
            df[col] = parsed
            parsed_cols.append(col)

    return df, parsed_cols


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = []
    datetime_cols = []
    categorical_cols = []
    flag_like_cols = []
    text_heavy_cols = []

    row_count = max(len(df), 1)

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_bool_dtype(series):
            flag_like_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        non_null = series.dropna().astype(str).str.strip()
        unique_count = non_null.nunique()

        lowered = set(non_null.str.lower().unique().tolist())
        if lowered and lowered.issubset(FLAG_VALUES_TRUE.union(FLAG_VALUES_FALSE)):
            flag_like_cols.append(col)
            continue

        avg_len = non_null.str.len().mean() if len(non_null) else 0
        unique_ratio = unique_count / row_count

        if avg_len > 40 and unique_ratio > 0.5:
            text_heavy_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "datetime": datetime_cols,
        "categorical": categorical_cols,
        "flag_like": flag_like_cols,
        "text_heavy": text_heavy_cols,
    }


def safe_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce").dropna()


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for col in df.columns:
        series = df[col]
        non_null = series.notna().sum()
        missing = series.isna().sum()
        missing_pct = round((missing / max(len(df), 1)) * 100, 2)
        nunique = series.nunique(dropna=True)

        sample_values = series.dropna().astype(str).head(5).tolist()

        row = {
            "column": col,
            "dtype": str(series.dtype),
            "non_null_count": int(non_null),
            "missing_count": int(missing),
            "missing_pct": missing_pct,
            "unique_count": int(nunique),
            "sample_values": " | ".join(sample_values),
            "min": "",
            "max": "",
            "mean": "",
            "median": "",
        }

        if pd.api.types.is_bool_dtype(series):
            pass
        elif pd.api.types.is_numeric_dtype(series):
            numeric = safe_numeric_series(series)
            if not numeric.empty:
                row["min"] = numeric.min()
                row["max"] = numeric.max()
                row["mean"] = numeric.mean()
                row["median"] = numeric.median()
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null_dates = series.dropna()
            if not non_null_dates.empty:
                row["min"] = non_null_dates.min()
                row["max"] = non_null_dates.max()

        records.append(row)

    return pd.DataFrame(records)


def build_missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = max(len(df), 1)

    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing,
                "missing_pct": round((missing / total_rows) * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values(["missing_pct", "missing_count"], ascending=False)


def build_categorical_profile(df: pd.DataFrame, cat_cols: List[str], top_n: int = 15) -> pd.DataFrame:
    records = []

    for col in cat_cols:
        value_counts = df[col].fillna("<<MISSING>>").astype(str).value_counts().head(top_n)
        for value, count in value_counts.items():
            records.append({"column": col, "value": value, "count": int(count)})

    return pd.DataFrame(records)


def build_flag_profile(df: pd.DataFrame, flag_cols: List[str]) -> pd.DataFrame:
    records = []

    for col in flag_cols:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            counts = series.fillna(False).astype(str).str.lower().value_counts(dropna=False)
        else:
            counts = (
                series.fillna("<<MISSING>>")
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts(dropna=False)
            )

        for value, count in counts.items():
            records.append({"column": col, "value": value, "count": int(count)})

    return pd.DataFrame(records)


def build_numeric_profile(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    records = []

    for col in numeric_cols:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            continue

        numeric = safe_numeric_series(series)

        if numeric.empty:
            records.append(
                {
                    "column": col,
                    "non_null_count": 0,
                    "min": "",
                    "p25": "",
                    "median": "",
                    "p75": "",
                    "max": "",
                    "mean": "",
                    "std_dev": "",
                }
            )
            continue

        records.append(
            {
                "column": col,
                "non_null_count": int(numeric.notna().sum()),
                "min": numeric.min(),
                "p25": numeric.quantile(0.25),
                "median": numeric.median(),
                "p75": numeric.quantile(0.75),
                "max": numeric.max(),
                "mean": numeric.mean(),
                "std_dev": numeric.std(),
            }
        )

    return pd.DataFrame(records)


def build_datetime_profile(df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
    records = []

    for col in dt_cols:
        s = pd.to_datetime(df[col], errors="coerce").dropna()

        if s.empty:
            records.append(
                {
                    "column": col,
                    "non_null_count": 0,
                    "min_date": "",
                    "max_date": "",
                    "year_min": "",
                    "year_max": "",
                    "distinct_years": 0,
                    "distinct_months": 0,
                }
            )
            continue

        years = s.dt.year.dropna()
        months = s.dt.to_period("M").astype(str).replace("NaT", np.nan).dropna()

        records.append(
            {
                "column": col,
                "non_null_count": int(s.notna().sum()),
                "min_date": s.min(),
                "max_date": s.max(),
                "year_min": years.min() if not years.empty else "",
                "year_max": years.max() if not years.empty else "",
                "distinct_years": int(years.nunique()) if not years.empty else 0,
                "distinct_months": int(months.nunique()) if not months.empty else 0,
            }
        )

    return pd.DataFrame(records)


def save_text_summary(
    output_path: Path,
    dataset_path: Path,
    sheet_used: str,
    all_sheets: List[str],
    df: pd.DataFrame,
    parsed_dates: List[str],
    classified: Dict[str, List[str]],
) -> None:
    duplicate_rows = int(df.duplicated().sum())

    lines = [
        "ISRM DATA PROFILE SUMMARY",
        "=" * 80,
        f"Dataset: {dataset_path}",
        f"Sheet used: {sheet_used}",
        f"Available sheets: {', '.join(all_sheets)}",
        "",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Duplicate rows: {duplicate_rows}",
        "",
        "Parsed date columns:",
        ", ".join(parsed_dates) if parsed_dates else "None",
        "",
        f"Numeric columns ({len(classified['numeric'])}):",
        ", ".join(classified["numeric"]) if classified["numeric"] else "None",
        "",
        f"Datetime columns ({len(classified['datetime'])}):",
        ", ".join(classified["datetime"]) if classified["datetime"] else "None",
        "",
        f"Flag-like columns ({len(classified['flag_like'])}):",
        ", ".join(classified["flag_like"]) if classified["flag_like"] else "None",
        "",
        f"Categorical columns ({len(classified['categorical'])}):",
        ", ".join(classified["categorical"]) if classified["categorical"] else "None",
        "",
        f"Text-heavy columns ({len(classified['text_heavy'])}):",
        ", ".join(classified["text_heavy"]) if classified["text_heavy"] else "None",
        "",
        "First 20 columns:",
        ", ".join(df.columns[:20].tolist()),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chart_missingness(missing_df: pd.DataFrame, outdir: Path) -> None:
    top = missing_df.head(20).copy()
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.35)))
    ax.barh(top["column"], top["missing_pct"])
    ax.invert_yaxis()
    ax.set_title("Top Columns by Missing Percentage")
    ax.set_xlabel("Missing %")
    ax.set_ylabel("Column")
    plt.tight_layout()
    fig.savefig(outdir / "01_missingness_top_columns.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def chart_top_categoricals(df: pd.DataFrame, cat_cols: List[str], outdir: Path) -> None:
    plotted = 0
    for col in cat_cols:
        counts = df[col].fillna("<<MISSING>>").astype(str).value_counts().head(12)
        if counts.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(counts.index.astype(str), counts.values)
        ax.invert_yaxis()
        ax.set_title(f"Top Values: {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        plt.tight_layout()

        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in col)[:60]
        fig.savefig(outdir / f"cat_{safe_name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        plotted += 1
        if plotted >= 8:
            break


def chart_top_flags(df: pd.DataFrame, flag_cols: List[str], outdir: Path) -> None:
    for col in flag_cols[:8]:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            counts = series.astype(str).str.lower().value_counts(dropna=False)
        else:
            counts = series.fillna("<<MISSING>>").astype(str).str.strip().str.lower().value_counts(dropna=False)

        if counts.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Flag Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.tight_layout()

        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in col)[:60]
        fig.savefig(outdir / f"flag_{safe_name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def chart_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str], outdir: Path) -> None:
    plotted = 0
    for col in numeric_cols:
        if pd.api.types.is_bool_dtype(df[col]):
            continue

        s = safe_numeric_series(df[col])
        if len(s) < 5:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(s, bins=20)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in col)[:60]
        fig.savefig(outdir / f"num_{safe_name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        plotted += 1
        if plotted >= 8:
            break


def chart_datetime_timelines(df: pd.DataFrame, dt_cols: List[str], outdir: Path) -> None:
    plotted = 0
    for col in dt_cols:
        s = pd.to_datetime(df[col], errors="coerce").dropna()
        if len(s) < 5:
            continue

        counts = s.dt.to_period("M").value_counts().sort_index()
        if counts.empty:
            continue

        labels = [str(x) for x in counts.index]
        values = counts.values

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(labels, values, marker="o")
        ax.set_title(f"Monthly Volume: {col}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in col)[:60]
        fig.savefig(outdir / f"dt_{safe_name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        plotted += 1
        if plotted >= 6:
            break


def save_profile_workbook(
    output_file: Path,
    overview_df: pd.DataFrame,
    column_profile_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    numeric_profile_df: pd.DataFrame,
    datetime_profile_df: pd.DataFrame,
    categorical_profile_df: pd.DataFrame,
    flag_profile_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="overview", index=False)
        column_profile_df.to_excel(writer, sheet_name="columns", index=False)
        missing_df.to_excel(writer, sheet_name="missingness", index=False)
        numeric_profile_df.to_excel(writer, sheet_name="numeric_profile", index=False)
        datetime_profile_df.to_excel(writer, sheet_name="datetime_profile", index=False)
        categorical_profile_df.to_excel(writer, sheet_name="categorical_top_values", index=False)
        flag_profile_df.to_excel(writer, sheet_name="flag_profile", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ISRM data before building better charts.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional path to the CSV/XLSX dataset. If omitted, the script searches the current folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="isrm_data_profile",
        help="Folder where the profiling outputs will be saved.",
    )
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "diagnostic_charts"

    ensure_folder(output_dir)
    ensure_folder(charts_dir)

    df_raw, sheet_used, all_sheets = load_dataset(dataset_path)
    df = normalize_headers(df_raw)
    df, parsed_dates = maybe_parse_dates(df)

    classified = classify_columns(df)
    duplicate_rows = int(df.duplicated().sum())

    overview_df = pd.DataFrame(
        [
            {"metric": "dataset_path", "value": str(dataset_path)},
            {"metric": "sheet_used", "value": sheet_used},
            {"metric": "available_sheets", "value": ", ".join(all_sheets)},
            {"metric": "row_count", "value": len(df)},
            {"metric": "column_count", "value": len(df.columns)},
            {"metric": "duplicate_rows", "value": duplicate_rows},
            {"metric": "parsed_date_columns", "value": ", ".join(parsed_dates)},
            {"metric": "numeric_columns", "value": ", ".join(classified["numeric"])},
            {"metric": "datetime_columns", "value": ", ".join(classified["datetime"])},
            {"metric": "flag_like_columns", "value": ", ".join(classified["flag_like"])},
            {"metric": "categorical_columns", "value": ", ".join(classified["categorical"])},
            {"metric": "text_heavy_columns", "value": ", ".join(classified["text_heavy"])},
        ]
    )

    column_profile_df = build_column_profile(df)
    missing_df = build_missingness_profile(df)
    numeric_profile_df = build_numeric_profile(df, classified["numeric"])
    datetime_profile_df = build_datetime_profile(df, classified["datetime"])
    categorical_profile_df = build_categorical_profile(df, classified["categorical"])
    flag_profile_df = build_flag_profile(df, classified["flag_like"])

    save_profile_workbook(
        output_dir / "isrm_profile_report.xlsx",
        overview_df,
        column_profile_df,
        missing_df,
        numeric_profile_df,
        datetime_profile_df,
        categorical_profile_df,
        flag_profile_df,
    )

    save_text_summary(
        output_dir / "isrm_profile_summary.txt",
        dataset_path,
        sheet_used,
        all_sheets,
        df,
        parsed_dates,
        classified,
    )

    chart_missingness(missing_df, charts_dir)
    chart_top_categoricals(df, classified["categorical"], charts_dir)
    chart_top_flags(df, classified["flag_like"], charts_dir)
    chart_numeric_histograms(df, classified["numeric"], charts_dir)
    chart_datetime_timelines(df, classified["datetime"], charts_dir)

    print("Profiling complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Sheet used: {sheet_used}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate rows: {duplicate_rows}")
    print("")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print(f"Workbook: {(output_dir / 'isrm_profile_report.xlsx').resolve()}")
    print(f"Summary: {(output_dir / 'isrm_profile_summary.txt').resolve()}")
    print(f"Charts: {charts_dir.resolve()}")


if __name__ == "__main__":
    main()
