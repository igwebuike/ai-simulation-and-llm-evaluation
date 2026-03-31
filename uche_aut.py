import argparse
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10


PREFERRED_SHEETS = ["RAW", "Raw", "raw", "Sheet1"]

COLUMN_ALIASES = {
    "record_id": ["Record_ID", "ID", "Issue_ID"],
    "issue_finding_type": ["Issue_Finding_Type", "Finding", "Issue Finding Type"],
    "manager_issue": [
        "Manager_Issue",
        "Business_Contact_Issue",
        "MRC_Company_Contact",
        "Accountable_Contact_Issue",
        "Issue Owner",
        "Control Owner",
    ],
    "global_control_reference": [
        "Global_Control_Reference",
        "Global_Control_Reference_Number",
        "ControlId",
        "Global Control Reference",
    ],
    "impacted_mrc": ["Impacted_MRC", "MRC_ID", "MRC"],
    "rating": ["Rating", "Severity", "Issue Rating", "Alert_1_Rating_Verbal"],
    "repeat_finding": ["Repeat_Finding", "Repeat Finding", "Repeat_Finding_Flag"],
    "sox_flag": ["In_Scope_For_SOX_Testing", "SOX_Certificate_Issue_Flag", "GS_MRC_Flag"],
    "issue_creation_date": ["Issue_Creation_Date", "Create_Date", "Release_Date"],
    "issue_closed_date": ["Issue_Closed_Date"],
    "expected_due_date": ["Expected_Due_Date", "Due_Date_Sld", "Expected_Closure_Date"],
    "revised_due_date": ["Revised_Due_Date", "Revised_Remediation_Date"],
    "final_month_snapshot": ["Final_Month_Snapshot"],
    "snapshot_year": ["Snapshot_Year", "Year", "Snapshot Year"],
    "issue_finding_state": ["Issue_Finding_State"],
    "issue_finding_state_subaction": ["Issue_Finding_State_Subaction"],
    "recommendation_state": ["Recommendation_State"],
    "overdue_flag": ["Overdue_Flag", "Flag_3_Open_and_Overdue", "Open_And_Overdue_Flag"],
    "days_old": ["Days_Old"],
    "days_until_due": ["Days_Until_Due"],
    "root_cause_description": ["Root_Cause_Description"],
    "control_category": ["Control_Category"],
    "risk_category": ["Risk_Category"],
    "objective_title": ["Objective_Title"],
}


YES_VALUES = {"yes", "y", "true", "1", "x"}
NO_VALUES = {"no", "n", "false", "0"}


def auto_find_dataset(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p

    here = Path(__file__).resolve().parent
    preferred = [
        "ISRM Report (Primary Data Source).xlsx",
        "ISRM Report (Primary Data Source).csv",
    ]
    for name in preferred:
        p = here / name
        if p.exists():
            return p

    for pattern in ("*.xlsx", "*.xlsm", "*.xls", "*.csv"):
        files = sorted(here.glob(pattern))
        if files:
            return files[0]

    raise FileNotFoundError("No dataset found in the script folder.")


def choose_sheet(xls: pd.ExcelFile) -> str:
    for s in PREFERRED_SHEETS:
        if s in xls.sheet_names:
            return s
    return xls.sheet_names[0]


def load_dataset(path: Path) -> Tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path), "CSV"

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet = choose_sheet(xls)
        return pd.read_excel(path, sheet_name=sheet), sheet

    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    for c in df.columns:
        s = str(c).strip().replace("\n", " ").replace("\r", " ")
        while "  " in s:
            s = s.replace("  ", " ")
        cleaned.append(s)
    df.columns = cleaned
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

    if "record_id" not in df.columns:
        df["record_id"] = range(1, len(df) + 1)

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "issue_creation_date",
        "issue_closed_date",
        "expected_due_date",
        "revised_due_date",
        "final_month_snapshot",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "snapshot_year" not in df.columns:
        if "final_month_snapshot" in df.columns:
            df["snapshot_year"] = df["final_month_snapshot"].dt.year
        elif "issue_creation_date" in df.columns:
            df["snapshot_year"] = df["issue_creation_date"].dt.year

    return df


def normalize_yes_no(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.replace(
        {
            "true": "yes",
            "false": "no",
            "1": "yes",
            "0": "no",
            "y": "yes",
            "n": "no",
            "x": "yes",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text_series(series: pd.Series, missing_label: str = "Unknown") -> pd.Series:
    return (
        series.fillna(missing_label)
        .astype(str)
        .str.strip()
        .replace("", missing_label)
    )


def top_counts(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    s = clean_text_series(df[col])
    return (
        s.value_counts()
        .head(top_n)
        .rename_axis(col)
        .reset_index(name="Count")
    )


def monthly_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=["Month", "Count"])
    counts = s.dt.to_period("M").value_counts().sort_index()
    return pd.DataFrame({"Month": [str(x) for x in counts.index], "Count": counts.values})


def save_barh(data: pd.DataFrame, category_col: str, value_col: str, title: str, path: Path) -> None:
    if data.empty:
        raise ValueError("No data to plot.")

    fig_h = max(5, len(data) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    bars = ax.barh(data[category_col].astype(str), data[value_col].astype(float))
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.set_xlabel(value_col)
    ax.set_ylabel(category_col)

    max_val = float(data[value_col].max())
    offset = max(1, max_val * 0.01)

    for bar, value in zip(bars, data[value_col]):
        ax.text(
            float(value) + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}" if float(value).is_integer() else f"{value:,.1f}",
            va="center",
        )

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_line(data: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    if data.empty:
        raise ValueError("No data to plot.")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data[x_col], data[y_col], marker="o", linewidth=2)
    ax.set_title(title, loc="left")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_stacked_bar(data: pd.DataFrame, title: str, path: Path, xlabel: str, ylabel: str) -> None:
    if data.empty:
        raise ValueError("No data to plot.")

    fig, ax = plt.subplots(figsize=(11, 6))
    data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(rows: List[Tuple[str, str]], path: Path, title: str) -> None:
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(df) * 0.55)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def report_overview(df: pd.DataFrame, outdir: Path) -> None:
    rows = [
        ("Total rows", f"{len(df):,}"),
        ("Total columns", f"{len(df.columns):,}"),
        ("Unique issues", f"{df['record_id'].nunique():,}" if "record_id" in df.columns else f"{len(df):,}"),
    ]

    if "rating" in df.columns:
        rows.append(("Unique ratings", f"{clean_text_series(df['rating']).nunique():,}"))
    if "issue_finding_type" in df.columns:
        rows.append(("Unique finding types", f"{clean_text_series(df['issue_finding_type']).nunique():,}"))
    if "global_control_reference" in df.columns:
        rows.append(("Unique control references", f"{clean_text_series(df['global_control_reference']).nunique():,}"))
    if "manager_issue" in df.columns:
        rows.append(("Unique owners/contacts", f"{clean_text_series(df['manager_issue']).nunique():,}"))
    if "issue_creation_date" in df.columns:
        dt = pd.to_datetime(df["issue_creation_date"], errors="coerce").dropna()
        if not dt.empty:
            rows.append(("First creation date", str(dt.min().date())))
            rows.append(("Last creation date", str(dt.max().date())))

    save_summary_table(rows, outdir / "01_overview_summary.png", "ISRM Overview Summary")


def report_issue_mix(df: pd.DataFrame, outdir: Path) -> None:
    if "issue_finding_type" in df.columns:
        data = top_counts(df, "issue_finding_type", 12)
        save_barh(data, "issue_finding_type", "Count", "Issue Mix by Finding Type", outdir / "02_issue_mix_by_type.png")

    if "rating" in df.columns:
        data = top_counts(df, "rating", 10)
        save_barh(data, "rating", "Count", "Issue Mix by Rating", outdir / "03_issue_mix_by_rating.png")


def report_ownership(df: pd.DataFrame, outdir: Path) -> None:
    if "manager_issue" in df.columns:
        data = top_counts(df, "manager_issue", 15)
        save_barh(data, "manager_issue", "Count", "Top Owners / Contacts by Issue Count", outdir / "04_top_owners_issue_count.png")

    if "global_control_reference" in df.columns:
        data = top_counts(df, "global_control_reference", 15)
        save_barh(
            data,
            "global_control_reference",
            "Count",
            "Top Control References by Issue Count",
            outdir / "05_top_control_references.png",
        )

    if "impacted_mrc" in df.columns:
        data = top_counts(df, "impacted_mrc", 12)
        save_barh(data, "impacted_mrc", "Count", "Top Impacted MRCs by Issue Count", outdir / "06_top_impacted_mrcs.png")


def report_trends(df: pd.DataFrame, outdir: Path) -> None:
    if "issue_creation_date" in df.columns:
        data = monthly_counts(df, "issue_creation_date")
        save_line(data, "Month", "Count", "Issue Creation Trend by Month", outdir / "07_creation_trend_monthly.png")

    if "issue_closed_date" in df.columns:
        data = monthly_counts(df, "issue_closed_date")
        save_line(data, "Month", "Count", "Issue Closure Trend by Month", outdir / "08_closure_trend_monthly.png")

    if "final_month_snapshot" in df.columns and "rating" in df.columns:
        temp = df.copy()
        temp["Month"] = pd.to_datetime(temp["final_month_snapshot"], errors="coerce").dt.to_period("M").astype(str)
        temp["rating"] = clean_text_series(temp["rating"])
        temp = temp[temp["Month"] != "NaT"]

        pivot = temp.pivot_table(
            index="Month",
            columns="rating",
            values="record_id",
            aggfunc="count",
            fill_value=0,
        )

        if not pivot.empty:
            pivot = pivot.sort_index()
            save_stacked_bar(
                pivot.tail(12),
                "Monthly Snapshot Trend by Rating",
                outdir / "09_snapshot_trend_by_rating.png",
                "Month",
                "Issue Count",
            )


def report_flags(df: pd.DataFrame, outdir: Path) -> None:
    if "repeat_finding" in df.columns:
        s = normalize_yes_no(df["repeat_finding"])
        data = s.fillna("unknown").value_counts().rename_axis("repeat_finding").reset_index(name="Count")
        save_barh(data, "repeat_finding", "Count", "Repeat Finding Distribution", outdir / "10_repeat_finding_distribution.png")

    if "sox_flag" in df.columns:
        s = normalize_yes_no(df["sox_flag"])
        data = s.fillna("unknown").value_counts().rename_axis("sox_flag").reset_index(name="Count")
        save_barh(data, "sox_flag", "Count", "SOX / In-Scope Distribution", outdir / "11_sox_distribution.png")

    if "overdue_flag" in df.columns:
        s = normalize_yes_no(df["overdue_flag"])
        data = s.fillna("unknown").value_counts().rename_axis("overdue_flag").reset_index(name="Count")
        save_barh(data, "overdue_flag", "Count", "Overdue Distribution", outdir / "12_overdue_distribution.png")


def report_state_views(df: pd.DataFrame, outdir: Path) -> None:
    if "issue_finding_state" in df.columns:
        data = top_counts(df, "issue_finding_state", 12)
        save_barh(data, "issue_finding_state", "Count", "Issue Count by Finding State", outdir / "13_issue_state_distribution.png")

    if "recommendation_state" in df.columns:
        data = top_counts(df, "recommendation_state", 12)
        save_barh(data, "recommendation_state", "Count", "Issue Count by Recommendation State", outdir / "14_recommendation_state_distribution.png")

    if "issue_finding_state_subaction" in df.columns:
        data = top_counts(df, "issue_finding_state_subaction", 12)
        save_barh(
            data,
            "issue_finding_state_subaction",
            "Count",
            "Issue Count by Finding State Subaction",
            outdir / "15_state_subaction_distribution.png",
        )


def report_aging(df: pd.DataFrame, outdir: Path) -> None:
    if "days_old" in df.columns:
        s = pd.to_numeric(df["days_old"], errors="coerce").dropna()
        if not s.empty:
            bins = [-np.inf, 30, 60, 90, 180, np.inf]
            labels = ["0-30", "31-60", "61-90", "91-180", "181+"]
            bucketed = pd.cut(s, bins=bins, labels=labels)
            data = bucketed.value_counts(sort=False).rename_axis("Age Bucket").reset_index(name="Count")
            save_barh(data, "Age Bucket", "Count", "Issue Aging Distribution", outdir / "16_issue_aging_distribution.png")

    if "days_until_due" in df.columns:
        s = pd.to_numeric(df["days_until_due"], errors="coerce").dropna()
        if not s.empty:
            bins = [-np.inf, 0, 15, 30, 60, np.inf]
            labels = ["Overdue", "1-15", "16-30", "31-60", "61+"]
            bucketed = pd.cut(s, bins=bins, labels=labels)
            data = bucketed.value_counts(sort=False).rename_axis("Days Until Due Bucket").reset_index(name="Count")
            save_barh(data, "Days Until Due Bucket", "Count", "Days Until Due Distribution", outdir / "17_due_bucket_distribution.png")


def report_deep_dive(df: pd.DataFrame, outdir: Path) -> None:
    if "rating" in df.columns and "issue_finding_type" in df.columns:
        temp = df.copy()
        temp["rating"] = clean_text_series(temp["rating"])
        temp["issue_finding_type"] = clean_text_series(temp["issue_finding_type"])

        pivot = temp.pivot_table(
            index="issue_finding_type",
            columns="rating",
            values="record_id",
            aggfunc="count",
            fill_value=0,
        )

        if not pivot.empty:
            save_stacked_bar(
                pivot,
                "Finding Type by Rating",
                outdir / "18_finding_type_by_rating.png",
                "Finding Type",
                "Issue Count",
            )

    if "manager_issue" in df.columns and "rating" in df.columns:
        temp = df.copy()
        temp["manager_issue"] = clean_text_series(temp["manager_issue"])
        temp["rating"] = clean_text_series(temp["rating"])

        top_owners = temp["manager_issue"].value_counts().head(10).index.tolist()
        temp = temp[temp["manager_issue"].isin(top_owners)]

        pivot = temp.pivot_table(
            index="manager_issue",
            columns="rating",
            values="record_id",
            aggfunc="count",
            fill_value=0,
        )

        if not pivot.empty:
            save_stacked_bar(
                pivot,
                "Top Owners by Rating Mix",
                outdir / "19_top_owners_by_rating_mix.png",
                "Owner / Contact",
                "Issue Count",
            )


def safe_run(name: str, func: Callable[[pd.DataFrame, Path], None], df: pd.DataFrame, outdir: Path) -> str:
    try:
        func(df, outdir)
        return f"[OK] {name}"
    except Exception as e:
        return f"[SKIPPED] {name}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple polished ISRM reports.")
    parser.add_argument("dataset", nargs="?", default=None, help="Optional path to dataset")
    parser.add_argument("--output-dir", default="isrm_professional_reports", help="Output directory")
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    raw_df, sheet_used = load_dataset(dataset_path)
    df = clean_headers(raw_df)
    df = apply_aliases(df)
    df = parse_dates(df)

    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    report_dirs = {
        "01_overview": outdir / "01_overview",
        "02_issue_mix": outdir / "02_issue_mix",
        "03_ownership": outdir / "03_ownership",
        "04_trends": outdir / "04_trends",
        "05_flags": outdir / "05_flags",
        "06_states": outdir / "06_states",
        "07_aging": outdir / "07_aging",
        "08_deep_dive": outdir / "08_deep_dive",
    }

    for p in report_dirs.values():
        ensure_dir(p)

    tasks = [
        ("Overview Report", report_overview, report_dirs["01_overview"]),
        ("Issue Mix Report", report_issue_mix, report_dirs["02_issue_mix"]),
        ("Ownership Report", report_ownership, report_dirs["03_ownership"]),
        ("Trend Report", report_trends, report_dirs["04_trends"]),
        ("Flags Report", report_flags, report_dirs["05_flags"]),
        ("State Report", report_state_views, report_dirs["06_states"]),
        ("Aging Report", report_aging, report_dirs["07_aging"]),
        ("Deep Dive Report", report_deep_dive, report_dirs["08_deep_dive"]),
    ]

    log_lines = [
        f"Dataset: {dataset_path}",
        f"Sheet used: {sheet_used}",
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns):,}",
        "",
    ]

    for name, func, folder in tasks:
        log_lines.append(safe_run(name, func, df, folder))

    (outdir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("Professional report generation complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Sheet used: {sheet_used}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("")
    print(f"Output folder: {outdir.resolve()}")
    print(f"Run log: {(outdir / 'run_log.txt').resolve()}")


if __name__ == "__main__":
    main()
