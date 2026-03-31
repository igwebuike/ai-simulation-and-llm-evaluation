import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Callable

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
    "record_id": ["Record_ID", "ID", "Issue_ID", "IssueId"],
    "issue_finding_type": ["Issue_Finding_Type", "Finding", "Issue Finding Type"],
    "business_contact": [
        "Business_Contact_Issue",
        "Accountable_Contact_Issue",
        "MRC_Company_Contact",
    ],
    "manager_issue": ["Manager_Issue"],
    "global_control_reference": [
        "Global_Control_Reference",
        "Global_Control_Reference_Number",
        "ControlId",
        "Controlid",
        "Global Control Reference",
    ],
    "control_name": ["Control_Name"],
    "control_category": ["Control_Category"],
    "risk_category": ["Risk_Category"],
    "impacted_mrc": ["Impacted_MRC", "MRC_ID", "MRC"],
    "rating": ["Rating", "Severity", "Issue Rating", "Alert_1_Rating_Verbal"],
    "repeat_finding": ["Repeat_Finding", "Repeat Finding", "Repeat_Finding_Flag"],
    "sox_flag": ["In_Scope_For_SOX_Testing", "SOX_Certificate_Issue_Flag", "GS_MRC_Flag"],
    "sox_reportable": ["SOX_Reportable", "SOX_Reportable_Classification"],
    "issue_creation_date": ["Issue_Creation_Date", "Create_Date", "Release_Date"],
    "issue_closed_date": ["Issue_Closed_Date"],
    "expected_due_date": ["Expected_Due_Date", "Due_Date_Sld", "Expected_Closure_Date"],
    "revised_due_date": ["Revised_Due_Date", "Revised_Remediation_Date"],
    "final_month_snapshot": ["Final_Month_Snapshot"],
    "snapshot_year": ["Snapshot_Year", "Year", "Snapshot Year", "Repeat_Year"],
    "issue_finding_state": ["Issue_Finding_State"],
    "issue_finding_state_subaction": ["Issue_Finding_State_Subaction"],
    "recommendation_state": ["Recommendation_State"],
    "recommendations_subaction": ["RecommendationsStateSubaction"],
    "overdue_flag": ["Overdue_Flag", "Flag_3_Open_and_Overdue", "Open_And_Overdue_Flag"],
    "days_old": ["Days_Old"],
    "days_until_due": ["Days_Until_Due"],
    "root_cause_description": ["Root_Cause_Description"],
    "common_insights": ["Common_Insights"],
    "entity": ["Entity"],
    "entity_region": ["Entity_Region", "Impacted_Region"],
    "entity_sector": ["Entity_Sector", "Impacted_Sector"],
    "issue_status_sox_cert": ["Issue_Status_SOX_Cert"],
    "issue_status_sox_short": ["Issue_Status_SOX_Short"],
}

YES_VALUES = {"yes", "y", "true", "1", "x", "in scope", "sox", "reportable"}
MISSING_LABEL = "Missing / Not Provided"

SEVERITY_COLOR_MAP = {
    "critical": "#d62728",              # red
    "major": "#ff7f0e",                 # orange
    "minor": "#f2c94c",                 # gold
    "medium": "#ff7f0e",                # orange
    "moderate": "#ff7f0e",              # orange
    "low": "#2ca02c",                   # green
    "missing / not provided": "#9e9e9e",
    "unknown": "#9e9e9e",
    "nan": "#9e9e9e",
}

DEFAULT_BLUE = "#4e79a7"


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text_series(series: pd.Series, missing_label: str = MISSING_LABEL) -> pd.Series:
    return (
        series.fillna(missing_label)
        .astype(str)
        .str.strip()
        .replace("", missing_label)
        .replace("nan", missing_label)
        .replace("NaN", missing_label)
        .replace("<NA>", missing_label)
    )


def normalize_yes_no(series: pd.Series) -> pd.Series:
    s = clean_text_series(series).str.lower()
    s = s.replace(
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
    return s


def top_counts(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    s = clean_text_series(df[col])
    return s.value_counts().head(top_n).rename_axis(col).reset_index(name="Count")


def monthly_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=["Month", "Count"])
    counts = s.dt.to_period("M").value_counts().sort_index()
    return pd.DataFrame({"Month": [str(x) for x in counts.index], "Count": counts.values})


def get_series_colors(labels) -> List[str]:
    colors = []
    for label in labels:
        key = str(label).strip().lower()
        colors.append(SEVERITY_COLOR_MAP.get(key, DEFAULT_BLUE))
    return colors


def get_stacked_colors(columns) -> List[str]:
    return [SEVERITY_COLOR_MAP.get(str(col).strip().lower(), DEFAULT_BLUE) for col in columns]


def reorder_severity_columns(cols: List[str]) -> List[str]:
    preferred = ["Critical", "Major", "Minor", "Medium", "Moderate", "Low", MISSING_LABEL]
    found = [c for c in preferred if c in cols]
    remainder = [c for c in cols if c not in found]
    return found + remainder


def save_barh(data: pd.DataFrame, category_col: str, value_col: str, title: str, path: Path) -> None:
    if data.empty:
        raise ValueError("No data to plot.")

    fig_h = max(5, len(data) * 0.45)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    colors = get_series_colors(data[category_col].tolist())
    bars = ax.barh(
        data[category_col].astype(str),
        data[value_col].astype(float),
        color=colors,
    )

    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.20)

    max_val = float(data[value_col].max())
    offset = max(1, max_val * 0.01)

    for bar, value in zip(bars, data[value_col]):
        label = f"{int(value):,}" if float(value).is_integer() else f"{value:,.1f}"
        ax.text(float(value) + offset, bar.get_y() + bar.get_height() / 2, label, va="center")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_line(data: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    if data.empty:
        raise ValueError("No data to plot.")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data[x_col], data[y_col], marker="o", linewidth=2, color=DEFAULT_BLUE)
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

    data = data.copy()
    ordered_cols = reorder_severity_columns(list(data.columns))
    data = data[ordered_cols]
    colors = get_stacked_colors(data.columns)

    fig, ax = plt.subplots(figsize=(11, 6))
    data.plot(kind="bar", stacked=True, ax=ax, color=colors)

    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.20)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(rows: List[Tuple[str, str]], path: Path, title: str) -> None:
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, len(df) * 0.55)))
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
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scope_label(year_scope: str, sox_only: bool) -> str:
    parts = []
    if year_scope == "2026_only":
        parts.append("2026 Only")
    elif year_scope == "2025_2026":
        parts.append("2025-2026")
    else:
        parts.append("All Years")

    parts.append("SOX Only" if sox_only else "All Issues")
    return " | ".join(parts)


def filter_by_scope(df: pd.DataFrame, year_scope: str, sox_only: bool) -> pd.DataFrame:
    out = df.copy()

    if year_scope != "all":
        year_col = None
        if "issue_creation_date" in out.columns:
            year_col = out["issue_creation_date"].dt.year
        elif "snapshot_year" in out.columns:
            year_col = pd.to_numeric(out["snapshot_year"], errors="coerce")

        if year_col is not None:
            if year_scope == "2026_only":
                out = out[year_col == 2026]
            elif year_scope == "2025_2026":
                out = out[year_col.isin([2025, 2026])]

    if sox_only and "sox_flag" in out.columns:
        s = normalize_yes_no(out["sox_flag"])
        out = out[s.isin(YES_VALUES)]

    return out.copy()


def get_business_contact_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["business_contact", "manager_issue"]:
        if col in df.columns:
            return col
    return None


def report_overview(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    rows = [
        ("Scope", scope_text),
        ("Total issues", f"{len(df):,}"),
        ("Unique issues", f"{df['record_id'].nunique():,}"),
    ]

    if "rating" in df.columns:
        rows.append(("Unique ratings", f"{clean_text_series(df['rating']).nunique():,}"))

    if "global_control_reference" in df.columns:
        rows.append(("Unique control references", f"{clean_text_series(df['global_control_reference']).nunique():,}"))

    business_col = get_business_contact_col(df)
    if business_col:
        rows.append(("Unique business contacts", f"{clean_text_series(df[business_col]).nunique():,}"))

    if "issue_creation_date" in df.columns:
        dt = pd.to_datetime(df["issue_creation_date"], errors="coerce").dropna()
        if not dt.empty:
            rows.append(("First creation date", str(dt.min().date())))
            rows.append(("Last creation date", str(dt.max().date())))

    if "issue_closed_date" in df.columns:
        dt = pd.to_datetime(df["issue_closed_date"], errors="coerce").dropna()
        if not dt.empty:
            rows.append(("First closed date", str(dt.min().date())))
            rows.append(("Last closed date", str(dt.max().date())))

    save_summary_table(rows, outdir / "01_overview_summary.png", f"ISRM Overview Summary ({scope_text})")


def report_issue_mix(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    if "issue_finding_type" in df.columns:
        data = top_counts(df, "issue_finding_type", 12)
        save_barh(data, "issue_finding_type", "Count", f"Issue Mix by Finding Type ({scope_text})", outdir / "02_issue_mix_by_type.png")

    if "rating" in df.columns:
        data = top_counts(df, "rating", 10)
        ordered = reorder_severity_columns(data["rating"].tolist())
        data["rating"] = pd.Categorical(data["rating"], categories=ordered, ordered=True)
        data = data.sort_values("rating")
        save_barh(data, "rating", "Count", f"Issue Mix by Rating ({scope_text})", outdir / "03_issue_mix_by_rating.png")


def report_business_ownership(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    business_col = get_business_contact_col(df)
    if business_col:
        data = top_counts(df, business_col, 15)
        save_barh(data, business_col, "Count", f"Top Business Contacts by Issue Count ({scope_text})", outdir / "04_top_business_contacts.png")

    if "impacted_mrc" in df.columns:
        data = top_counts(df, "impacted_mrc", 12)
        save_barh(data, "impacted_mrc", "Count", f"Top Impacted MRCs by Issue Count ({scope_text})", outdir / "05_top_impacted_mrcs.png")


def report_controls(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    if "control_name" in df.columns:
        data = top_counts(df, "control_name", 12)
        save_barh(data, "control_name", "Count", f"Top Controls by Issue Count ({scope_text})", outdir / "06_top_controls.png")

    if "global_control_reference" in df.columns:
        data = top_counts(df, "global_control_reference", 12)
        save_barh(data, "global_control_reference", "Count", f"Top Control References by Issue Count ({scope_text})", outdir / "07_top_control_refs.png")

    if "control_category" in df.columns:
        data = top_counts(df, "control_category", 10)
        save_barh(data, "control_category", "Count", f"Control Category Distribution ({scope_text})", outdir / "08_control_category_distribution.png")

    if "risk_category" in df.columns:
        data = top_counts(df, "risk_category", 10)
        save_barh(data, "risk_category", "Count", f"Risk Category Distribution ({scope_text})", outdir / "09_risk_category_distribution.png")


def report_trends(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    if "issue_creation_date" in df.columns:
        data = monthly_counts(df, "issue_creation_date")
        save_line(data, "Month", "Count", f"Issue Creation Trend by Month ({scope_text})", outdir / "10_creation_trend_monthly.png")

    if "issue_closed_date" in df.columns:
        data = monthly_counts(df, "issue_closed_date")
        save_line(data, "Month", "Count", f"Issue Closure Trend by Month ({scope_text})", outdir / "11_closure_trend_monthly.png")


def report_2026_focus(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    if "issue_closed_date" in df.columns:
        closed_2026 = df[df["issue_closed_date"].dt.year == 2026].copy()
        rows = [("Scope", scope_text), ("Issues closed in 2026", f"{len(closed_2026):,}")]
        save_summary_table(rows, outdir / "12_closed_2026_summary.png", "2026 Closure Summary")

        if not closed_2026.empty:
            data = monthly_counts(closed_2026, "issue_closed_date")
            save_line(data, "Month", "Count", "Issues Closed in 2026 by Month", outdir / "13_closed_2026_monthly.png")

    if "issue_creation_date" in df.columns:
        created_2026 = df[df["issue_creation_date"].dt.year == 2026].copy()
        rows = [("Scope", scope_text), ("Issues created in 2026", f"{len(created_2026):,}")]
        save_summary_table(rows, outdir / "14_created_2026_summary.png", "2026 Creation Summary")

        if not created_2026.empty:
            data = monthly_counts(created_2026, "issue_creation_date")
            save_line(data, "Month", "Count", "Issues Created in 2026 by Month", outdir / "15_created_2026_monthly.png")


def report_flags(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    if "repeat_finding" in df.columns:
        s = normalize_yes_no(df["repeat_finding"])
        data = s.value_counts().rename_axis("repeat_finding").reset_index(name="Count")
        data["repeat_finding"] = data["repeat_finding"].replace("nan", MISSING_LABEL)
        save_barh(data, "repeat_finding", "Count", f"Repeat Finding Distribution ({scope_text})", outdir / "16_repeat_finding_distribution.png")

    if "sox_flag" in df.columns:
        s = clean_text_series(df["sox_flag"])
        data = s.value_counts().rename_axis("sox_flag").reset_index(name="Count")
        save_barh(data, "sox_flag", "Count", f"SOX / In-Scope Distribution ({scope_text})", outdir / "17_sox_distribution.png")

    if "overdue_flag" in df.columns:
        s = normalize_yes_no(df["overdue_flag"])
        data = s.value_counts().rename_axis("overdue_flag").reset_index(name="Count")
        data["overdue_flag"] = data["overdue_flag"].replace("nan", MISSING_LABEL)
        save_barh(data, "overdue_flag", "Count", f"Overdue Distribution ({scope_text})", outdir / "18_overdue_distribution.png")


def report_states(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    for col, fname, title in [
        ("issue_finding_state", "19_issue_state_distribution.png", "Issue Count by Finding State"),
        ("recommendation_state", "20_recommendation_state_distribution.png", "Issue Count by Recommendation State"),
        ("issue_finding_state_subaction", "21_state_subaction_distribution.png", "Issue Count by Finding State Subaction"),
        ("recommendations_subaction", "22_recommendations_subaction_distribution.png", "Issue Count by Recommendation Subaction"),
    ]:
        if col in df.columns:
            data = top_counts(df, col, 12)
            save_barh(data, col, "Count", f"{title} ({scope_text})", outdir / fname)


def report_aging(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    if "days_old" in df.columns:
        s = pd.to_numeric(df["days_old"], errors="coerce").dropna()
        if not s.empty:
            bins = [-np.inf, 30, 60, 90, 180, np.inf]
            labels = ["0-30", "31-60", "61-90", "91-180", "181+"]
            bucketed = pd.cut(s, bins=bins, labels=labels)
            data = bucketed.value_counts(sort=False).rename_axis("Age Bucket").reset_index(name="Count")
            save_barh(data, "Age Bucket", "Count", f"Issue Aging Distribution ({scope_text})", outdir / "23_issue_aging_distribution.png")

    if "days_until_due" in df.columns:
        s = pd.to_numeric(df["days_until_due"], errors="coerce").dropna()
        if not s.empty:
            bins = [-np.inf, 0, 15, 30, 60, np.inf]
            labels = ["Overdue", "1-15", "16-30", "31-60", "61+"]
            bucketed = pd.cut(s, bins=bins, labels=labels)
            data = bucketed.value_counts(sort=False).rename_axis("Days Until Due Bucket").reset_index(name="Count")
            save_barh(data, "Days Until Due Bucket", "Count", f"Days Until Due Distribution ({scope_text})", outdir / "24_due_bucket_distribution.png")


def report_deep_dive(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
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
            pivot = pivot[reorder_severity_columns(list(pivot.columns))]
            save_stacked_bar(
                pivot,
                f"Finding Type by Rating ({scope_text})",
                outdir / "25_finding_type_by_rating.png",
                "Finding Type",
                "Issue Count",
            )

    business_col = get_business_contact_col(df)
    if business_col and "rating" in df.columns:
        temp = df.copy()
        temp[business_col] = clean_text_series(temp[business_col])
        temp["rating"] = clean_text_series(temp["rating"])
        top_items = temp[business_col].value_counts().head(10).index.tolist()
        temp = temp[temp[business_col].isin(top_items)]

        pivot = temp.pivot_table(
            index=business_col,
            columns="rating",
            values="record_id",
            aggfunc="count",
            fill_value=0,
        )

        if not pivot.empty:
            pivot = pivot[reorder_severity_columns(list(pivot.columns))]
            save_stacked_bar(
                pivot,
                f"Top Business Contacts by Rating Mix ({scope_text})",
                outdir / "26_business_contacts_by_rating_mix.png",
                "Business Contact",
                "Issue Count",
            )


def report_notes(df: pd.DataFrame, outdir: Path, scope_text: str) -> None:
    rows: List[Tuple[str, str]] = [("Scope", scope_text)]

    if "control_name" in df.columns:
        top_controls = top_counts(df, "control_name", 3)
        if not top_controls.empty:
            rows.append(("Top controls driving issues", "; ".join(top_controls["control_name"].astype(str).tolist())))

    if "control_category" in df.columns:
        top_cats = top_counts(df, "control_category", 3)
        if not top_cats.empty:
            rows.append(("Top control categories", "; ".join(top_cats["control_category"].astype(str).tolist())))

    if "risk_category" in df.columns:
        top_risk = top_counts(df, "risk_category", 3)
        if not top_risk.empty:
            rows.append(("Top risk categories", "; ".join(top_risk["risk_category"].astype(str).tolist())))

    business_col = get_business_contact_col(df)
    if business_col:
        top_contacts = top_counts(df, business_col, 3)
        if not top_contacts.empty:
            rows.append(("Top business contacts", "; ".join(top_contacts[business_col].astype(str).tolist())))

    save_summary_table(rows, outdir / "27_notes_and_context.png", "Notes / Control Context")


def safe_run(name: str, func: Callable[[pd.DataFrame, Path, str], None], df: pd.DataFrame, outdir: Path, scope_text: str) -> str:
    try:
        func(df, outdir, scope_text)
        return f"[OK] {name}"
    except Exception as e:
        return f"[SKIPPED] {name}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate upgraded ISRM business-facing reports.")
    parser.add_argument("dataset", nargs="?", default=None, help="Optional path to dataset")
    parser.add_argument("--output-dir", default="isrm_professional_reports_v2", help="Output directory")
    parser.add_argument("--year-scope", choices=["all", "2025_2026", "2026_only"], default="all")
    parser.add_argument("--sox-only", action="store_true")
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    raw_df, sheet_used = load_dataset(dataset_path)
    df = clean_headers(raw_df)
    df = apply_aliases(df)
    df = parse_dates(df)
    df = filter_by_scope(df, args.year_scope, args.sox_only)

    scope_text = scope_label(args.year_scope, args.sox_only)

    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    report_dirs = {
        "01_overview": outdir / "01_overview",
        "02_issue_mix": outdir / "02_issue_mix",
        "03_business_ownership": outdir / "03_business_ownership",
        "04_controls": outdir / "04_controls",
        "05_trends": outdir / "05_trends",
        "06_2026_focus": outdir / "06_2026_focus",
        "07_flags": outdir / "07_flags",
        "08_states": outdir / "08_states",
        "09_aging": outdir / "09_aging",
        "10_deep_dive": outdir / "10_deep_dive",
        "11_notes": outdir / "11_notes",
    }

    for p in report_dirs.values():
        ensure_dir(p)

    tasks = [
        ("Overview Report", report_overview, report_dirs["01_overview"]),
        ("Issue Mix Report", report_issue_mix, report_dirs["02_issue_mix"]),
        ("Business Ownership Report", report_business_ownership, report_dirs["03_business_ownership"]),
        ("Controls Report", report_controls, report_dirs["04_controls"]),
        ("Trend Report", report_trends, report_dirs["05_trends"]),
        ("2026 Focus Report", report_2026_focus, report_dirs["06_2026_focus"]),
        ("Flags Report", report_flags, report_dirs["07_flags"]),
        ("State Report", report_states, report_dirs["08_states"]),
        ("Aging Report", report_aging, report_dirs["09_aging"]),
        ("Deep Dive Report", report_deep_dive, report_dirs["10_deep_dive"]),
        ("Notes / Context Report", report_notes, report_dirs["11_notes"]),
    ]

    log_lines = [
        f"Dataset: {dataset_path}",
        f"Sheet used: {sheet_used}",
        f"Rows after filter: {len(df):,}",
        f"Columns: {len(df.columns):,}",
        f"Scope: {scope_text}",
        "",
    ]

    for name, func, folder in tasks:
        log_lines.append(safe_run(name, func, df, folder, scope_text))

    (outdir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("Upgraded report generation complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Sheet used: {sheet_used}")
    print(f"Rows after filter: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Scope: {scope_text}")
    print("")
    print(f"Output folder: {outdir.resolve()}")
    print(f"Run log: {(outdir / 'run_log.txt').resolve()}")


if __name__ == "__main__":
    main()
