import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
PREFERRED_SHEETS = ["RAW", "Raw", "raw", "Sheet1"]

DATE_CANDIDATES = [
    "Issue_Creation_Date",
    "Issue_Closed_Date",
    "Expected_Due_Date",
    "Revised_Due_Date",
    "Initial_Agreed_Remediation_Date",
    "Final_Month_Snapshot",
]

COLUMN_ALIASES = {
    "Record_ID": ["Record_ID", "ID", "Issue_ID"],
    "Issue_Finding_Type": ["Issue_Finding_Type", "Finding", "Issue Finding Type"],
    "Issue_Finding_State": ["Issue_Finding_State", "Issue State", "Status"],
    "Issue_Finding_State_Subaction": ["Issue_Finding_State_Subaction"],
    "Issue_Creation_Date": ["Issue_Creation_Date"],
    "Issue_Closed_Date": ["Issue_Closed_Date"],
    "Expected_Due_Date": ["Expected_Due_Date", "Due_Date_Sld"],
    "Revised_Due_Date": ["Revised_Due_Date"],
    "Final_Month_Snapshot": ["Final_Month_Snapshot"],
    "Snapshot_Year": ["Snapshot_Year"],
    "Days_Old": ["Days_Old"],
    "Days_Until_Due": ["Days_Until_Due"],
    "Overdue_Flag": ["Overdue_Flag", "Flag_3_Open_and_Overdue", "Open_And_Overdue_Flag"],
    "Repeat_Finding": ["Repeat_Finding"],
    "Rating": ["Rating", "Alert_1_Rating_Verbal", "Severity"],
    "Global_Control_Reference": ["Global_Control_Reference", "Global_Control_Reference_Number", "ControlId"],
    "Control_Name": ["Control_Name"],
    "Control_Category": ["Control_Category"],
    "Manager_Issue": [
        "Manager_Issue",
        "Business_Contact_Issue",
        "Accountable_Contact_Issue",
        "MRC_Company_Contact",
    ],
    "MRC_ID": ["MRC_ID", "Impacted_MRC"],
    "Impacted_Sector": ["Impacted_Sector", "IT_Asset_Accountable_Sector"],
    "Impacted_Region": ["Impacted_Region"],
    "SOX_Flag": ["In_Scope_For_SOX_Testing", "GS_MRC_Flag", "SOX_Certificate_Issue_Flag"],
    "Root_Cause_Description": ["Root_Cause_Description"],
    "Root_Cause_Insights": ["Root_Cause_Insights"],
    "Management_Response": ["Management_Response"],
    "Recommendation_Title": ["Recommendation_Title"],
    "Recommendation_State": ["Recommendation_State", "Remediation_Status"],
    "Released": ["Released"],
    "Open_Critical_Issues_Sld_Flag": ["Open_Critical_Issues_Sld_Flag"],
    "Open_Major_Issues_Sld": ["Open_Major_Issues_Sld"],
}

REPORT_FOLDERS = [
    "01_executive_summary",
    "02_issue_volume_trends",
    "03_aging_due_overdue",
    "04_control_hotspots",
    "05_owner_accountability",
    "06_sox_compliance",
    "07_data_quality_appendix",
]


# -----------------------------
# Helpers
# -----------------------------
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

    raise FileNotFoundError("No dataset found in the script folder.")


def choose_sheet(xls: pd.ExcelFile) -> str:
    for sheet in PREFERRED_SHEETS:
        if sheet in xls.sheet_names:
            return sheet
    return xls.sheet_names[0]


def load_dataset(path: Path) -> Tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        return df, "CSV"

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet_name = choose_sheet(xls)
        df = pd.read_excel(path, sheet_name=sheet_name)
        return df, sheet_name

    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
    return df


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    rename_map = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.strip().lower()
            if key in lower_map:
                rename_map[lower_map[key]] = canonical
                break

    df = df.rename(columns=rename_map)

    if "Record_ID" not in df.columns:
        df["Record_ID"] = range(1, len(df) + 1)

    for col in DATE_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Snapshot_Year" not in df.columns and "Final_Month_Snapshot" in df.columns:
        df["Snapshot_Year"] = pd.to_datetime(df["Final_Month_Snapshot"], errors="coerce").dt.year

    return df


def clean_flag_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.replace(
        {
            "yes": "YES",
            "y": "YES",
            "true": "YES",
            "1": "YES",
            "x": "YES",
            "no": "NO",
            "n": "NO",
            "false": "NO",
            "0": "NO",
            "nan": "NO",
            "": "NO",
        }
    )


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for folder in REPORT_FOLDERS:
        path = base_dir / folder
        path.mkdir(parents=True, exist_ok=True)
        out[folder] = path
    return out


def save_barh(series: pd.Series, title: str, xlabel: str, output: Path, max_items: int = 12) -> None:
    s = series.head(max_items)
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(s) * 0.45)))
    ax.barh(s.index.astype(str), s.values)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_vertical_bar(series: pd.Series, title: str, ylabel: str, output: Path) -> None:
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(series.index.astype(str), series.values)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(series.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_line(series: pd.Series, title: str, ylabel: str, output: Path) -> None:
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(series.index.astype(str), series.values, marker="o", linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stacked_bar(df_plot: pd.DataFrame, title: str, ylabel: str, output: Path) -> None:
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    df_plot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_table_image(df_table: pd.DataFrame, title: str, output: Path) -> None:
    if df_table.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, df_table.shape[1] * 1.8), max(3, df_table.shape[0] * 0.55)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=12)

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E78")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#F3F6FA")
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_summary_metric(records: List[Dict[str, object]], metric: str, value: object) -> None:
    records.append({"Metric": metric, "Value": value})


# -----------------------------
# Report sections
# -----------------------------
def create_executive_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    metrics = []

    add_summary_metric(metrics, "Total records", len(df))
    add_summary_metric(metrics, "Columns", len(df.columns))

    if "Issue_Finding_State" in df.columns:
        open_like = df["Issue_Finding_State"].astype(str).str.lower().str.contains("open", na=False).sum()
        closed_like = df["Issue_Finding_State"].astype(str).str.lower().str.contains("closed", na=False).sum()
        add_summary_metric(metrics, "Open-like issues", int(open_like))
        add_summary_metric(metrics, "Closed-like issues", int(closed_like))

        status_counts = df["Issue_Finding_State"].fillna("Unknown").astype(str).value_counts()
        save_barh(
            status_counts,
            "Issue Count by Status",
            "Issue Count",
            outdir / "executive_issue_count_by_status.png",
            max_items=12,
        )

    if "Rating" in df.columns:
        rating_counts = df["Rating"].fillna("Unknown").astype(str).value_counts()
        add_summary_metric(metrics, "Distinct ratings", int(rating_counts.shape[0]))
        save_vertical_bar(
            rating_counts,
            "Issue Count by Rating",
            "Issue Count",
            outdir / "executive_issue_count_by_rating.png",
        )

    if "Repeat_Finding" in df.columns:
        repeat_counts = clean_flag_series(df["Repeat_Finding"]).value_counts()
        repeat_yes = int(repeat_counts.get("YES", 0))
        add_summary_metric(metrics, "Repeat findings", repeat_yes)

    if "Overdue_Flag" in df.columns:
        overdue_counts = clean_flag_series(df["Overdue_Flag"]).value_counts()
        overdue_yes = int(overdue_counts.get("YES", 0))
        add_summary_metric(metrics, "Overdue issues", overdue_yes)

    summary_df = pd.DataFrame(metrics)
    summary_df.to_excel(outdir / "executive_summary_metrics.xlsx", index=False)
    save_table_image(summary_df, "Executive Summary Metrics", outdir / "executive_summary_metrics.png")
    return summary_df


def create_issue_volume_trends(df: pd.DataFrame, outdir: Path) -> None:
    if "Issue_Creation_Date" in df.columns:
        created = pd.to_datetime(df["Issue_Creation_Date"], errors="coerce").dropna()
        if not created.empty:
            created_monthly = created.dt.to_period("M").value_counts().sort_index()
            created_monthly.index = created_monthly.index.astype(str)
            save_line(
                created_monthly,
                "Issues Created Over Time",
                "Issue Count",
                outdir / "issues_created_over_time.png",
            )

    if "Issue_Closed_Date" in df.columns:
        closed = pd.to_datetime(df["Issue_Closed_Date"], errors="coerce").dropna()
        if not closed.empty:
            closed_monthly = closed.dt.to_period("M").value_counts().sort_index()
            closed_monthly.index = closed_monthly.index.astype(str)
            save_line(
                closed_monthly,
                "Issues Closed Over Time",
                "Issue Count",
                outdir / "issues_closed_over_time.png",
            )

    if "Issue_Finding_Type" in df.columns:
        finding_counts = df["Issue_Finding_Type"].fillna("Unknown").astype(str).value_counts()
        save_barh(
            finding_counts,
            "Issue Count by Finding Type",
            "Issue Count",
            outdir / "issue_count_by_finding_type.png",
            max_items=15,
        )

    if "Snapshot_Year" in df.columns and "Rating" in df.columns:
        yearly = df.dropna(subset=["Snapshot_Year"]).copy()
        if not yearly.empty:
            pivot = yearly.pivot_table(
                index="Snapshot_Year",
                columns="Rating",
                values="Record_ID",
                aggfunc="count",
                fill_value=0,
            )
            if not pivot.empty:
                save_stacked_bar(
                    pivot,
                    "Issue Volume by Year and Rating",
                    "Issue Count",
                    outdir / "issue_volume_by_year_and_rating.png",
                )


def create_aging_due_overdue(df: pd.DataFrame, outdir: Path) -> None:
    if "Days_Old" in df.columns:
        days_old = pd.to_numeric(df["Days_Old"], errors="coerce").dropna()
        if not days_old.empty:
            bins = [-np.inf, 30, 60, 90, 180, np.inf]
            labels = ["0-30", "31-60", "61-90", "91-180", "181+"]
            aging = pd.cut(days_old, bins=bins, labels=labels).value_counts().sort_index()
            save_vertical_bar(
                aging,
                "Issue Aging Buckets",
                "Issue Count",
                outdir / "issue_aging_buckets.png",
            )

    if "Days_Until_Due" in df.columns:
        days_until_due = pd.to_numeric(df["Days_Until_Due"], errors="coerce").dropna()
        if not days_until_due.empty:
            bins = [-np.inf, 0, 7, 30, 60, np.inf]
            labels = ["Overdue", "Due in 1-7", "Due in 8-30", "Due in 31-60", "60+"]
            due_buckets = pd.cut(days_until_due, bins=bins, labels=labels).value_counts().sort_index()
            save_vertical_bar(
                due_buckets,
                "Due Date Buckets",
                "Issue Count",
                outdir / "due_date_buckets.png",
            )

    if "Overdue_Flag" in df.columns:
        overdue = clean_flag_series(df["Overdue_Flag"]).value_counts()
        save_vertical_bar(
            overdue,
            "Overdue vs Not Overdue",
            "Issue Count",
            outdir / "overdue_flag_distribution.png",
        )

    if "Expected_Due_Date" in df.columns:
        due = pd.to_datetime(df["Expected_Due_Date"], errors="coerce").dropna()
        if not due.empty:
            due_monthly = due.dt.to_period("M").value_counts().sort_index()
            due_monthly.index = due_monthly.index.astype(str)
            save_line(
                due_monthly,
                "Expected Due Dates by Month",
                "Issue Count",
                outdir / "expected_due_dates_by_month.png",
            )


def create_control_hotspots(df: pd.DataFrame, outdir: Path) -> None:
    if "Global_Control_Reference" in df.columns:
        control_counts = df["Global_Control_Reference"].fillna("Unknown").astype(str).value_counts()
        save_barh(
            control_counts,
            "Top Control References by Issue Count",
            "Issue Count",
            outdir / "top_control_references.png",
            max_items=15,
        )

    if "Control_Category" in df.columns:
        category_counts = df["Control_Category"].fillna("Unknown").astype(str).value_counts()
        save_barh(
            category_counts,
            "Issue Count by Control Category",
            "Issue Count",
            outdir / "issue_count_by_control_category.png",
            max_items=12,
        )

    if "Global_Control_Reference" in df.columns and "Rating" in df.columns:
        top_controls = (
            df["Global_Control_Reference"].fillna("Unknown").astype(str).value_counts().head(10).index.tolist()
        )
        subset = df[df["Global_Control_Reference"].fillna("Unknown").astype(str).isin(top_controls)].copy()
        pivot = subset.pivot_table(
            index="Global_Control_Reference",
            columns="Rating",
            values="Record_ID",
            aggfunc="count",
            fill_value=0,
        )
        if not pivot.empty:
            save_stacked_bar(
                pivot,
                "Top Control References by Rating",
                "Issue Count",
                outdir / "top_control_references_by_rating.png",
            )


def create_owner_accountability(df: pd.DataFrame, outdir: Path) -> None:
    if "Manager_Issue" in df.columns:
        owner_counts = df["Manager_Issue"].fillna("Unknown").astype(str).value_counts()
        save_barh(
            owner_counts,
            "Top Owners by Issue Count",
            "Issue Count",
            outdir / "top_owners_by_issue_count.png",
            max_items=15,
        )

    if "Manager_Issue" in df.columns and "Overdue_Flag" in df.columns:
        overdue_mask = clean_flag_series(df["Overdue_Flag"]) == "YES"
        overdue_by_owner = (
            df.loc[overdue_mask, "Manager_Issue"].fillna("Unknown").astype(str).value_counts()
        )
        save_barh(
            overdue_by_owner,
            "Top Owners by Overdue Issues",
            "Overdue Issue Count",
            outdir / "top_owners_by_overdue_issues.png",
            max_items=15,
        )

    if "Manager_Issue" in df.columns and "Repeat_Finding" in df.columns:
        repeat_mask = clean_flag_series(df["Repeat_Finding"]) == "YES"
        repeat_by_owner = (
            df.loc[repeat_mask, "Manager_Issue"].fillna("Unknown").astype(str).value_counts()
        )
        save_barh(
            repeat_by_owner,
            "Top Owners by Repeat Findings",
            "Repeat Finding Count",
            outdir / "top_owners_by_repeat_findings.png",
            max_items=15,
        )


def create_sox_compliance(df: pd.DataFrame, outdir: Path) -> None:
    if "SOX_Flag" in df.columns:
        sox_counts = clean_flag_series(df["SOX_Flag"]).value_counts()
        save_vertical_bar(
            sox_counts,
            "SOX Scope Distribution",
            "Issue Count",
            outdir / "sox_scope_distribution.png",
        )

    if "SOX_Flag" in df.columns and "Rating" in df.columns:
        subset = df.copy()
        subset["SOX_Flag"] = clean_flag_series(subset["SOX_Flag"])
        pivot = subset.pivot_table(
            index="SOX_Flag",
            columns="Rating",
            values="Record_ID",
            aggfunc="count",
            fill_value=0,
        )
        if not pivot.empty:
            save_stacked_bar(
                pivot,
                "SOX Scope by Rating",
                "Issue Count",
                outdir / "sox_scope_by_rating.png",
            )

    if "SOX_Flag" in df.columns and "Snapshot_Year" in df.columns:
        subset = df.copy()
        subset["SOX_Flag"] = clean_flag_series(subset["SOX_Flag"])
        subset = subset.dropna(subset=["Snapshot_Year"])
        if not subset.empty:
            pivot = subset.pivot_table(
                index="Snapshot_Year",
                columns="SOX_Flag",
                values="Record_ID",
                aggfunc="count",
                fill_value=0,
            )
            if not pivot.empty:
                save_stacked_bar(
                    pivot,
                    "SOX Scope Trend by Year",
                    "Issue Count",
                    outdir / "sox_scope_trend_by_year.png",
                )


def create_data_quality_appendix(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    total_rows = max(len(df), 1)

    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "Column": col,
                "Missing Count": missing,
                "Missing %": round((missing / total_rows) * 100, 2),
                "Distinct Values": int(df[col].nunique(dropna=True)),
            }
        )

    dq = pd.DataFrame(rows).sort_values(["Missing %", "Missing Count"], ascending=False)
    dq.to_excel(outdir / "data_quality_profile.xlsx", index=False)

    save_table_image(
        dq.head(20),
        "Top 20 Columns by Missingness",
        outdir / "top_20_columns_by_missingness.png",
    )

    missing_series = dq.set_index("Column")["Missing %"].head(15)
    save_barh(
        missing_series,
        "Top Missing Columns",
        "Missing %",
        outdir / "top_missing_columns_bar.png",
        max_items=15,
    )


# -----------------------------
# Workbook summary
# -----------------------------
def write_master_workbook(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    base_dir: Path,
) -> None:
    workbook_path = base_dir / "ISRM_Professional_Report_Pack.xlsx"

    sheets = {
        "executive_summary": summary_df,
    }

    if "Issue_Finding_State" in df.columns:
        sheets["status_counts"] = (
            df["Issue_Finding_State"].fillna("Unknown").astype(str).value_counts().rename_axis("Status").reset_index(name="Issue Count")
        )

    if "Rating" in df.columns:
        sheets["rating_counts"] = (
            df["Rating"].fillna("Unknown").astype(str).value_counts().rename_axis("Rating").reset_index(name="Issue Count")
        )

    if "Global_Control_Reference" in df.columns:
        sheets["control_counts"] = (
            df["Global_Control_Reference"].fillna("Unknown").astype(str).value_counts().rename_axis("Global_Control_Reference").reset_index(name="Issue Count")
        )

    if "Manager_Issue" in df.columns:
        sheets["owner_counts"] = (
            df["Manager_Issue"].fillna("Unknown").astype(str).value_counts().rename_axis("Owner").reset_index(name="Issue Count")
        )

    if "Issue_Creation_Date" in df.columns:
        created = pd.to_datetime(df["Issue_Creation_Date"], errors="coerce").dropna()
        if not created.empty:
            sheets["created_monthly"] = (
                created.dt.to_period("M").value_counts().sort_index().rename_axis("Month").reset_index(name="Issue Count")
            )

    if "Days_Old" in df.columns:
        days_old = pd.to_numeric(df["Days_Old"], errors="coerce").dropna()
        if not days_old.empty:
            bins = [-np.inf, 30, 60, 90, 180, np.inf]
            labels = ["0-30", "31-60", "61-90", "91-180", "181+"]
            aging = pd.cut(days_old, bins=bins, labels=labels).value_counts().sort_index()
            sheets["aging_buckets"] = aging.rename_axis("Aging Bucket").reset_index(name="Issue Count")

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            data.to_excel(writer, sheet_name=sheet_name[:31], index=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a professional ISRM report pack.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional path to the dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="isrm_professional_report_pack",
        help="Output folder for report pack.",
    )
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    report_dirs = ensure_dirs(output_dir)

    df_raw, sheet_used = load_dataset(dataset_path)
    df = normalize_headers(df_raw)
    df = apply_aliases(df)

    summary_df = create_executive_summary(df, report_dirs["01_executive_summary"])
    create_issue_volume_trends(df, report_dirs["02_issue_volume_trends"])
    create
