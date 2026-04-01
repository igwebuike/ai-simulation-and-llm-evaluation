import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# GLOBAL STYLE
# =========================
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["font.family"] = "DejaVu Sans"

COLOR_RED = "#C00000"
COLOR_CRITICAL = "#D62728"
COLOR_MAJOR = "#F4B400"
COLOR_MINOR = "#FFF200"
COLOR_HEADER_TEXT = "white"
COLOR_GRID = "#D9D9D9"
COLOR_ROW_A = "#F7F2F2"
COLOR_ROW_B = "#FFFFFF"
COLOR_BG = "#FFFFFF"
MISSING_LABEL = "Missing / Not Provided"


# =========================
# CONFIG
# =========================
PREFERRED_SHEETS = ["RAW", "Raw", "raw", "Sheet1"]

COLUMN_ALIASES = {
    "record_id": ["Record_ID", "ID", "Issue_ID", "IssueId"],
    "finding_summary": [
        "Finding Summary",
        "Summary_Finding_Exec_Summary",
        "Issue_Finding_Title",
        "Finding",
    ],
    "root_cause": ["Root_Cause_Description", "Finding Root Cause", "Root Cause", "Root_Cause_Insights"],
    "control_owner": ["Business_Contact_Recommendations", "Control Owner", "Business_Contact_Issue"],
    "sector_responsible": [
        "Sector Responsible",
        "Impacted_Sector",
        "Entity_Sector",
        "IT_Asset_Accountable_Sector",
    ],
    "system_area": [
        "System & Control / Area",
        "System & Control",
        "Area",
        "Control_Category",
        "Risk_Category",
    ],
    "release_date": ["Release_Date"],
    "issue_creation_date": ["Issue_Creation_Date", "Create_Date"],
    "issue_closed_date": ["Issue_Closed_Date"],
    "remediation_date": [
        "Initial_Agreed_Remediation_Date",
        "Expected_Due_Date",
        "Revised_Due_Date",
        "Revised_Remediation_Date",
        "Due_Date_Sld",
    ],
    "status": [
        "Issue_Status_Sld",
        "Issue_Status_SOX_Short",
        "Issue_Status_SOX_Cert",
        "Issue_Finding_State",
        "Issue_Status",
    ],
    "actual_status": [
        "Recommendation_State",
        "RecommendationsStateSubaction",
        "Issue_Finding_State_Subaction",
        "Remediation Status",
    ],
    "sox_type": ["SOX_Type"],
    "rating": ["Rating", "Severity", "Issue Rating", "Alert_1_Rating_Verbal"],
    "repeat_finding": ["Repeat_Finding", "Repeat Finding", "Repeat_Finding_Flag"],
    "year": ["Audit_Year", "Snapshot_Year", "Repeat_Year"],
    "open_flag": ["Open_Issues_Sld_Flag", "Issue_Status_SOX_Short", "Issue_Status_Sld"],
}


# =========================
# FILE / DATA LOAD
# =========================
def auto_find_dataset(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p

    here = Path(__file__).resolve().parent
    preferred = [
        "ISRM Report (Primary Data Source).xlsx",
        "ISRM Report (Primary Data Source).xlsm",
        "ISRM Report (Primary Data Source).xls",
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
        df = pd.read_csv(path)
        return df, "CSV"

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet = choose_sheet(xls)
        df = pd.read_excel(path, sheet_name=sheet)
        return df, sheet

    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    for c in df.columns:
        s = str(c).strip().replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s)
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

    for col in ["release_date", "issue_creation_date", "issue_closed_date", "remediation_date", "year"]:
        if col in df.columns:
            if col == "year":
                parsed = pd.to_datetime(df[col], errors="coerce")
                year_num = pd.to_numeric(df[col], errors="coerce")
                df[col] = np.where(parsed.notna(), parsed.dt.year, year_num)
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    if "year" not in df.columns:
        if "issue_creation_date" in df.columns:
            df["year"] = df["issue_creation_date"].dt.year
        elif "release_date" in df.columns:
            df["year"] = df["release_date"].dt.year
        elif "issue_closed_date" in df.columns:
            df["year"] = df["issue_closed_date"].dt.year

    return df


# =========================
# CLEAN / STANDARDIZE
# =========================
def clean_text(value, missing_label=MISSING_LABEL) -> str:
    if pd.isna(value):
        return missing_label
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return missing_label
    return s


def clean_series(series: pd.Series, missing_label=MISSING_LABEL) -> pd.Series:
    return series.apply(lambda x: clean_text(x, missing_label))


def yes_no_normalize(series: pd.Series) -> pd.Series:
    s = clean_series(series).str.lower()
    mapping = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "x": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
    }
    return s.map(lambda x: mapping.get(x, x))


def is_open_status(value: str) -> bool:
    s = clean_text(value, "").lower()
    return any(k in s for k in ["open", "in progress", "not started", "overdue", "pending"])


def wrap_text(value: str, width: int) -> str:
    s = clean_text(value, "")
    if not s:
        return ""
    return textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=False)


def clean_finding_text(text: str) -> str:
    s = clean_text(text, "")
    if not s:
        return ""

    replacements = {
        "Insufficient ": "Lack of ",
        "Incomplete ": "Missing ",
        "Inadequate ": "Weak ",
        "Temporary Elevated Access Control Design": "Privileged Access Controls",
        "Segregation of Duties": "SoD Controls",
        "Risk Control Matrix": "RCM Definition",
        "Network Segmentation": "Network Segmentation Controls",
        "Supply Chain Business Continuity Plan": "BCP Controls",
        "Process owner": "Control owner",
        "service organization": "service provider",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_date(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return pd.to_datetime(value).strftime("%m/%d/%Y")
    except Exception:
        return str(value)


def ensure_columns(df: pd.DataFrame, needed: List[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def coalesce_columns(df: pd.DataFrame, target_col: str, candidates: List[str]) -> pd.DataFrame:
    df = df.copy()
    if target_col in df.columns:
        return df

    for c in candidates:
        if c in df.columns:
            df[target_col] = df[c]
            return df

    df[target_col] = ""
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Make sure business contact recommendations becomes control owner
    if "control_owner" not in df.columns:
        df["control_owner"] = ""

    for fallback in ["Business_Contact_Recommendations", "Business_Contact_Issue", "control_owner"]:
        if fallback in df.columns:
            df["control_owner"] = np.where(
                clean_series(df["control_owner"], "").eq(""),
                df[fallback],
                df["control_owner"],
            )

    # clean important columns
    for col in [
        "finding_summary",
        "root_cause",
        "control_owner",
        "sector_responsible",
        "system_area",
        "status",
        "actual_status",
        "sox_type",
        "rating",
        "repeat_finding",
    ]:
        if col in df.columns:
            df[col] = clean_series(df[col])

    if "finding_summary" in df.columns:
        df["finding_summary"] = df["finding_summary"].apply(clean_finding_text)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "release_date" in df.columns:
        df["release_date_fmt"] = df["release_date"].apply(format_date)
    else:
        df["release_date_fmt"] = ""

    if "remediation_date" in df.columns:
        df["remediation_date_fmt"] = df["remediation_date"].apply(format_date)
    else:
        df["remediation_date_fmt"] = ""

    if "status" not in df.columns:
        df["status"] = ""

    if "actual_status" not in df.columns:
        df["actual_status"] = ""

    if "sox_type" not in df.columns:
        df["sox_type"] = MISSING_LABEL

    return df


# =========================
# FILTERS
# =========================
def filter_years(df: pd.DataFrame, years=(2025, 2026)) -> pd.DataFrame:
    if "year" not in df.columns:
        return df.copy()
    return df[df["year"].isin(years)].copy()


def filter_open_issues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "status" in df.columns:
        mask = df["status"].apply(is_open_status)
        return df[mask].copy()
    return df.copy()


def filter_repeat_findings(df: pd.DataFrame) -> pd.DataFrame:
    if "repeat_finding" not in df.columns:
        return df.iloc[0:0].copy()
    s = yes_no_normalize(df["repeat_finding"])
    return df[s == "yes"].copy()


def filter_critical_findings(df: pd.DataFrame) -> pd.DataFrame:
    if "rating" not in df.columns:
        return df.iloc[0:0].copy()
    s = clean_series(df["rating"]).str.lower()
    return df[s == "critical"].copy()


def filter_sox_open(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_open_issues(df)
    if "sox_type" in df.columns:
        s = clean_series(df["sox_type"]).str.lower()
        # remove type 3 completely
        df = df[~s.eq("type 3")].copy()
    return df


# =========================
# OUTPUT UTILS
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wrapped_title(title: str, width: int = 55) -> str:
    return textwrap.fill(title, width=width, break_long_words=False, break_on_hyphens=False)


def add_title(ax, title: str) -> None:
    ax.set_title(wrapped_title(title), loc="left", pad=26, color=COLOR_RED, fontsize=18, fontweight="bold")


def severity_color(label: str) -> str:
    s = clean_text(label, "").lower()
    if s == "critical":
        return COLOR_CRITICAL
    if s == "major":
        return COLOR_MAJOR
    if s == "minor":
        return COLOR_MINOR
    return "#BFBFBF"


def table_row_colors(n_rows: int) -> List[str]:
    colors = []
    for i in range(n_rows):
        colors.append(COLOR_ROW_A if i % 2 == 0 else COLOR_ROW_B)
    return colors


# =========================
# TABLE RENDERING
# =========================
def render_table_image(
    df: pd.DataFrame,
    title: str,
    path: Path,
    col_widths: Optional[List[float]] = None,
    font_size: int = 10,
    row_height_scale: float = 1.65,
    title_wrap_width: int = 55,
    wrap_widths: Optional[Dict[str, int]] = None,
    figsize: Tuple[float, float] = (20, 10),
):
    if df.empty:
        return

    display_df = df.copy()

    if wrap_widths:
        for col, width in wrap_widths.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: wrap_text(x, width))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLOR_BG)
    ax.axis("off")

    ax.set_title(
        textwrap.fill(title, width=title_wrap_width, break_long_words=False, break_on_hyphens=False),
        loc="left",
        pad=18,
        color=COLOR_RED,
        fontsize=18,
        fontweight="bold",
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        colLoc="left",
        cellLoc="left",
        loc="upper left",
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_height_scale)

    nrows = display_df.shape[0]
    ncols = display_df.shape[1]
    body_colors = table_row_colors(nrows)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.8)
        cell.get_text().set_wrap(True)

        if r == 0:
            cell.set_facecolor(COLOR_RED)
            cell.get_text().set_color(COLOR_HEADER_TEXT)
            cell.get_text().set_weight("bold")
            cell.set_height(cell.get_height() * 1.15)
        else:
            cell.set_facecolor(body_colors[r - 1])

    plt.subplots_adjust(top=0.88, left=0.01, right=0.99, bottom=0.02)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def paginate_table(
    df: pd.DataFrame,
    title: str,
    outdir: Path,
    filename_prefix: str,
    rows_per_page: int,
    col_widths: List[float],
    font_size: int,
    row_height_scale: float,
    wrap_widths: Dict[str, int],
    figsize: Tuple[float, float],
):
    if df.empty:
        return

    total_pages = math.ceil(len(df) / rows_per_page)

    for page_num in range(total_pages):
        start = page_num * rows_per_page
        end = start + rows_per_page
        chunk = df.iloc[start:end].copy()

        page_title = title if total_pages == 1 else f"{title} (Page {page_num + 1} of {total_pages})"
        outfile = outdir / f"{filename_prefix}_page_{page_num + 1}.png"

        render_table_image(
            chunk,
            page_title,
            outfile,
            col_widths=col_widths,
            font_size=font_size,
            row_height_scale=row_height_scale,
            wrap_widths=wrap_widths,
            figsize=figsize,
        )


# =========================
# CHARTS
# =========================
def chart_open_findings_by_sector_risk(df: pd.DataFrame, year: int, outdir: Path):
    year_df = filter_sox_open(df)
    year_df = year_df[year_df["year"] == year].copy()

    ensure_columns(year_df, ["sector_responsible", "rating", "record_id"])

    if year_df.empty:
        return

    year_df["sector_responsible"] = clean_series(year_df["sector_responsible"])
    year_df["rating"] = clean_series(year_df["rating"])

    pivot = year_df.pivot_table(
        index="system_area" if "system_area" in year_df.columns else "sector_responsible",
        columns="rating",
        values="record_id",
        aggfunc="count",
        fill_value=0,
    )

    ordered_cols = [c for c in ["Critical", "Major", "Minor"] if c in pivot.columns]
    if not ordered_cols:
        return
    pivot = pivot[ordered_cols]

    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).drop(columns=["Total"])

    fig_h = max(6, len(pivot) * 0.75)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor(COLOR_BG)

    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))

    for col in pivot.columns:
        vals = pivot[col].values
        bars = ax.barh(y, vals, left=left, color=severity_color(col), label=col)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0:
                ax.text(left[i] + v / 2, bar.get_y() + bar.get_height() / 2, f"{int(v)}", ha="center", va="center", fontsize=11)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index.tolist(), fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", color=COLOR_GRID, alpha=0.8)
    ax.set_axisbelow(True)
    add_title(ax, f"{year} YTD IT SOX Open Findings Risk Area & Trends")
    ax.legend(title="Severity", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=11, title_fontsize=12)

    plt.subplots_adjust(top=0.88, left=0.22, right=0.86, bottom=0.06)
    fig.savefig(outdir / f"{year}_open_findings_risk_area_trends.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_sox_open_type_breakdown(df: pd.DataFrame, year: int, outdir: Path):
    year_df = filter_sox_open(df)
    year_df = year_df[year_df["year"] == year].copy()

    ensure_columns(year_df, ["sector_responsible", "sox_type", "record_id"])

    if year_df.empty:
        return

    year_df["sector_responsible"] = clean_series(year_df["sector_responsible"])
    year_df["sox_type"] = clean_series(year_df["sox_type"])

    # REMOVE TYPE 3 COMPLETELY
    year_df = year_df[~year_df["sox_type"].str.lower().eq("type 3")].copy()

    pivot = year_df.pivot_table(
        index="sector_responsible",
        columns="sox_type",
        values="record_id",
        aggfunc="count",
        fill_value=0,
    )

    ordered_cols = ["Missing / Not Provided", "Type 1", "Type 2"]
    pivot = pivot[[c for c in ordered_cols if c in pivot.columns]]

    pivot["Grand Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Grand Total", ascending=False)

    grand = pivot.sum(numeric_only=True).to_frame().T
    grand.index = ["Grand Total"]
    pivot = pd.concat([pivot, grand], axis=0)

    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"sector_responsible": "Sector Responsible"})

    render_table_image(
        pivot,
        f"{year} YTD IT SOX Open Findings by Sector - Type Breakdown",
        outdir / f"{year}_sox_open_findings_type_breakdown.png",
        col_widths=[0.18, 0.18, 0.18, 0.18, 0.18],
        font_size=12,
        row_height_scale=1.9,
        wrap_widths={"Sector Responsible": 22},
        figsize=(16, 7),
    )


# =========================
# DETAIL TABLES
# =========================
def report_critical_findings(df: pd.DataFrame, outdir: Path):
    crit = filter_critical_findings(filter_years(df))
    crit = crit[crit["year"].isin([2025, 2026])].copy()

    if crit.empty:
        return

    cols = [
        "year",
        "finding_summary",
        "system_area",
        "sector_responsible",
        "control_owner",
        "root_cause",
        "release_date_fmt",
        "remediation_date_fmt",
        "status",
    ]
    for c in cols:
        if c not in crit.columns:
            crit[c] = ""

    report = crit[cols].copy()
    report.columns = [
        "Year",
        "Finding Summary",
        "System & Control / Area",
        "Sector Responsible",
        "Control Owner",
        "Finding Root Cause",
        "Release Date",
        "Action Plan Remediation Date",
        "Status",
    ]

    report["Year"] = report["Year"].fillna("").apply(lambda x: "" if x == "" else str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x))

    report = report.sort_values(by=["Year", "Release Date", "Finding Summary"], ascending=[True, True, True])

    paginate_table(
        report,
        "Critical IT SOX Findings – Includes Release Date and Control Owner",
        outdir,
        "critical_it_sox_findings",
        rows_per_page=9,
        col_widths=[0.06, 0.19, 0.14, 0.13, 0.14, 0.20, 0.10, 0.12, 0.10],
        font_size=9.5,
        row_height_scale=3.0,
        wrap_widths={
            "Finding Summary": 30,
            "System & Control / Area": 22,
            "Sector Responsible": 18,
            "Control Owner": 18,
            "Finding Root Cause": 32,
            "Status": 18,
        },
        figsize=(24, 10),
    )


def report_repeat_findings(df: pd.DataFrame, outdir: Path):
    rpt = filter_repeat_findings(filter_years(df))
    rpt = rpt[~clean_series(rpt["sox_type"]).str.lower().eq("type 3")] if "sox_type" in rpt.columns else rpt

    if rpt.empty:
        return

    cols = [
        "year",
        "finding_summary",
        "system_area",
        "sector_responsible",
        "control_owner",
        "root_cause",
        "actual_status",
        "remediation_date_fmt",
        "sox_type",
    ]
    for c in cols:
        if c not in rpt.columns:
            rpt[c] = ""

    report = rpt[cols].copy()
    report.columns = [
        "Year",
        "Finding Summary",
        "System & Control",
        "Sector Responsible",
        "Control Owner",
        "Finding Root Cause",
        "Actual Status",
        "Remediation Date",
        "SOX Type",
    ]

    report["Year"] = report["Year"].fillna("").apply(lambda x: "" if x == "" else str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x))
    report = report.sort_values(by=["Year", "SOX Type", "Finding Summary"], ascending=[True, True, True])

    paginate_table(
        report,
        "Current Repeat IT SOX Findings – Updated with Actual Status",
        outdir,
        "current_repeat_it_sox_findings",
        rows_per_page=16,
        col_widths=[0.05, 0.17, 0.11, 0.10, 0.11, 0.20, 0.10, 0.09, 0.07],
        font_size=8.5,
        row_height_scale=1.9,
        wrap_widths={
            "Finding Summary": 28,
            "System & Control": 18,
            "Sector Responsible": 15,
            "Control Owner": 16,
            "Finding Root Cause": 30,
            "Actual Status": 18,
            "SOX Type": 10,
        },
        figsize=(22, 11),
    )


# =========================
# SUMMARY METRICS
# =========================
def report_metrics_summary(df: pd.DataFrame, outdir: Path):
    data = filter_years(df)
    open_data = filter_open_issues(data)
    repeat_data = filter_repeat_findings(data)
    crit_data = filter_critical_findings(data)
    overdue_data = data.iloc[0:0].copy()

    if "status" in data.columns:
        overdue_mask = clean_series(data["status"]).str.lower().str.contains("overdue", na=False)
        overdue_data = data[overdue_mask].copy()

    rows = []
    for year in [2025, 2026]:
        rows.append({
            "Year": year,
            "Current Repeat Findings": len(repeat_data[repeat_data["year"] == year]),
            "Current Critical SOX Findings": len(crit_data[crit_data["year"] == year]),
            "Current Open SOX Findings": len(open_data[open_data["year"] == year]),
            "Current Overdue Findings": len(overdue_data[overdue_data["year"] == year]),
        })

    metrics_df = pd.DataFrame(rows)

    render_table_image(
        metrics_df,
        "Metrics – Current Open SOX / Audit Findings",
        outdir / "metrics_current_open_findings.png",
        col_widths=[0.12, 0.22, 0.22, 0.22, 0.22],
        font_size=13,
        row_height_scale=2.4,
        figsize=(16, 4.8),
    )


# =========================
# MAIN REPORT RUNNER
# =========================
def generate_reports(df: pd.DataFrame, output_dir: Path):
    ensure_dir(output_dir)

    report_metrics_summary(df, output_dir)

    for year in [2025, 2026]:
        chart_open_findings_by_sector_risk(df, year, output_dir)
        chart_sox_open_type_breakdown(df, year, output_dir)

    report_critical_findings(df, output_dir)
    report_repeat_findings(df, output_dir)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate upgraded ISRM 2025/2026 SOX reporting visuals.")
    parser.add_argument("dataset", nargs="?", default=None, help="Optional path to dataset")
    parser.add_argument("--output-dir", default="isrm_reporting_output_v3", help="Output directory")
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    raw_df, sheet_used = load_dataset(dataset_path)

    df = clean_headers(raw_df)
    df = apply_aliases(df)
    df = parse_dates(df)
    df = prepare_dataframe(df)
    df = filter_years(df, years=(2025, 2026))

    output_dir = Path(args.output_dir)
    generate_reports(df, output_dir)

    run_log = [
        f"Dataset: {dataset_path}",
        f"Sheet used: {sheet_used}",
        f"Rows loaded: {len(raw_df):,}",
        f"Rows after 2025/2026 filter: {len(df):,}",
        f"Columns after aliasing: {len(df.columns):,}",
        "",
        "Generated outputs:",
        "- metrics_current_open_findings.png",
        "- 2025_open_findings_risk_area_trends.png",
        "- 2025_sox_open_findings_type_breakdown.png",
        "- 2026_open_findings_risk_area_trends.png",
        "- 2026_sox_open_findings_type_breakdown.png",
        "- critical_it_sox_findings_page_X.png",
        "- current_repeat_it_sox_findings_page_X.png",
    ]
    (output_dir / "run_log.txt").write_text("\n".join(run_log), encoding="utf-8")

    print("Report generation complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Sheet used: {sheet_used}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Run log: {(output_dir / 'run_log.txt').resolve()}")


if __name__ == "__main__":
    main()
