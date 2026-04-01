
import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
TARGET_YEARS = [2025, 2026]

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
    "root_cause": [
        "Root_Cause_Description",
        "Finding Root Cause",
        "Root Cause",
        "Root_Cause_Insights",
    ],
    "control_owner": [
        "Business_Contact_Recommendations",
        "Control Owner",
        "Business_Contact_Issue",
    ],
    "sector_responsible": [
        "Sector Responsible",
        "Impacted_Sector",
        "Entity_Sector",
        "IT_Asset_Accountable_Sector",
    ],
    "system_area": [
        "System & Control / Area",
        "System & Control",
        "System_&_Control",
        "Area",
        "Control_Category",
        "Risk_Category",
        "Area / Risk Area",
    ],
    "release_date": ["Release_Date"],
    "issue_creation_date": ["Issue_Creation_Date", "Create_Date"],
    "issue_closed_date": ["Issue_Closed_Date"],
    "expected_due_date": ["Expected_Due_Date", "Due_Date_Sld"],
    "revised_due_date": ["Revised_Due_Date", "Revised_Remediation_Date"],
    "initial_remediation_date": ["Initial_Agreed_Remediation_Date"],
    "final_month_snapshot": ["Final_Month_Snapshot"],
    "audit_year": ["Audit_Year"],
    "repeat_year": ["Repeat_Year"],
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
        "Recommendation_State_Subaction",
    ],
    "sox_type": ["SOX_Type"],
    "rating": ["Rating", "Severity", "Issue Rating", "Alert_1_Rating_Verbal"],
    "repeat_finding": ["Repeat_Finding", "Repeat Finding", "Repeat_Finding_Flag"],
    "open_flag": ["Open_Issues_Sld_Flag", "Open_Critical_issues_Sld_Flag"],
    "overdue_flag": ["Overdue_Flag", "Open_And_Overdue_Flag", "Flag_3_Open_and_Overdue"],
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


# =========================
# CLEAN / STANDARDIZE
# =========================
def clean_text(value, missing_label=MISSING_LABEL) -> str:
    if pd.isna(value):
        return missing_label
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    if s == "" or s.lower() in {"nan", "none", "<na>", "nat"}:
        return missing_label
    return s


def clean_series(series: pd.Series, missing_label=MISSING_LABEL) -> pd.Series:
    return series.apply(lambda x: clean_text(x, missing_label))


def normalize_yes_no(series: pd.Series) -> pd.Series:
    s = clean_series(series, "").str.lower()
    mapping = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "x": "yes",
        "open": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
        "closed": "no",
    }
    return s.map(lambda x: mapping.get(x, x))


def is_open_status(value: str) -> bool:
    s = clean_text(value, "").lower()
    return any(k in s for k in ["open", "in progress", "not started", "overdue", "pending"])


def is_overdue_status(value: str) -> bool:
    s = clean_text(value, "").lower()
    return "overdue" in s


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


def coalesce_text_columns(df: pd.DataFrame, candidates: List[str], default=MISSING_LABEL) -> pd.Series:
    result = pd.Series([default] * len(df), index=df.index, dtype="object")
    for col in candidates:
        if col in df.columns:
            cur = clean_series(df[col], "")
            result = np.where(pd.Series(result).astype(str).eq(default) | pd.Series(result).astype(str).eq(""), cur.replace("", default), result)
            result = pd.Series(result, index=df.index, dtype="object")
    return clean_series(result, default)


def try_parse_year_from_any(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    dt = pd.to_datetime(series, errors="coerce")
    out = dt.dt.year
    if out.notna().any():
        return out.astype("float64")
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        # handle excel serial dates and plain years
        serial_mask = num.gt(30000) & num.lt(60000)
        if serial_mask.any():
            serial_dates = pd.to_datetime("1899-12-30") + pd.to_timedelta(num[serial_mask], unit="D")
            out2 = pd.Series(np.nan, index=series.index, dtype="float64")
            out2.loc[serial_mask] = serial_dates.dt.year.values
            out2.loc[~serial_mask] = num[~serial_mask]
            return out2
        return num.astype("float64")
    return pd.Series(np.nan, index=series.index, dtype="float64")


def derive_report_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    year_sources = []

    for col in ["repeat_year", "audit_year", "final_month_snapshot", "issue_creation_date", "release_date", "issue_closed_date", "expected_due_date", "revised_due_date", "initial_remediation_date"]:
        if col in df.columns:
            year_sources.append(try_parse_year_from_any(df[col]))

    if year_sources:
        report_year = year_sources[0].copy()
        for s in year_sources[1:]:
            report_year = report_year.fillna(s)
        df["report_year"] = pd.to_numeric(report_year, errors="coerce")
    else:
        df["report_year"] = np.nan

    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse date-like columns first
    for col in ["release_date", "issue_creation_date", "issue_closed_date", "expected_due_date", "revised_due_date", "initial_remediation_date", "final_month_snapshot", "audit_year", "repeat_year"]:
        if col in df.columns:
            # keep original values, but parse where helpful
            if col in ["release_date", "issue_creation_date", "issue_closed_date", "expected_due_date", "revised_due_date", "initial_remediation_date", "final_month_snapshot", "audit_year"]:
                parsed = pd.to_datetime(df[col], errors="coerce")
                # only replace when parsed gives something useful
                if parsed.notna().any():
                    df[col] = parsed

    df = derive_report_year(df)

    df["finding_summary"] = coalesce_text_columns(df, ["finding_summary", "Issue_Finding_Title", "Summary_Finding_Exec_Summary", "Finding"], "")
    df["root_cause"] = coalesce_text_columns(df, ["root_cause", "Root_Cause_Description", "Root_Cause_Insights"], "")
    df["control_owner"] = coalesce_text_columns(df, ["control_owner", "Business_Contact_Recommendations", "Business_Contact_Issue"], "")
    df["sector_responsible"] = coalesce_text_columns(df, ["sector_responsible", "Impacted_Sector", "Entity_Sector", "IT_Asset_Accountable_Sector"], MISSING_LABEL)
    df["system_area"] = coalesce_text_columns(df, ["system_area", "Area", "Control_Category", "Risk_Category"], MISSING_LABEL)
    df["status"] = coalesce_text_columns(df, ["status", "Issue_Status_Sld", "Issue_Status_SOX_Short", "Issue_Status_SOX_Cert", "Issue_Finding_State"], "")
    df["actual_status"] = coalesce_text_columns(df, ["actual_status", "Recommendation_State", "RecommendationsStateSubaction", "Issue_Finding_State_Subaction", "Remediation Status"], "")
    df["sox_type"] = coalesce_text_columns(df, ["sox_type", "SOX_Type"], MISSING_LABEL)
    df["rating"] = coalesce_text_columns(df, ["rating", "Rating", "Alert_1_Rating_Verbal"], MISSING_LABEL)
    df["repeat_finding"] = coalesce_text_columns(df, ["repeat_finding", "Repeat_Finding"], "")
    df["overdue_flag"] = coalesce_text_columns(df, ["overdue_flag", "Overdue_Flag", "Open_And_Overdue_Flag", "Flag_3_Open_and_Overdue"], "")
    df["open_flag"] = coalesce_text_columns(df, ["open_flag", "Open_Issues_Sld_Flag", "Open_Critical_issues_Sld_Flag"], "")
    df["finding_summary"] = df["finding_summary"].apply(clean_finding_text)

    # final formatted dates
    for src, fmt_col in [
        ("release_date", "release_date_fmt"),
        ("expected_due_date", "expected_due_date_fmt"),
        ("revised_due_date", "revised_due_date_fmt"),
        ("initial_remediation_date", "initial_remediation_date_fmt"),
    ]:
        if src in df.columns:
            df[fmt_col] = df[src].apply(format_date)
        else:
            df[fmt_col] = ""

    # best remediation date to show
    df["remediation_date_fmt"] = ""
    for col in ["revised_due_date_fmt", "expected_due_date_fmt", "initial_remediation_date_fmt"]:
        if col in df.columns:
            df["remediation_date_fmt"] = np.where(
                clean_series(df["remediation_date_fmt"], "").eq(""),
                df[col],
                df["remediation_date_fmt"],
            )

    return df


# =========================
# FILTERS
# =========================
def filter_target_years(df: pd.DataFrame) -> pd.DataFrame:
    if "report_year" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["report_year"].isin(TARGET_YEARS)].copy()


def filter_open_issues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "status" in df.columns and clean_series(df["status"], "").ne("").any():
        mask = df["status"].apply(is_open_status)
        return df[mask].copy()
    if "open_flag" in df.columns:
        mask = normalize_yes_no(df["open_flag"]).eq("yes")
        return df[mask].copy()
    return df.copy()


def filter_repeat_findings(df: pd.DataFrame) -> pd.DataFrame:
    if "repeat_finding" not in df.columns:
        return df.iloc[0:0].copy()
    s = normalize_yes_no(df["repeat_finding"])
    return df[s == "yes"].copy()


def filter_critical_findings(df: pd.DataFrame) -> pd.DataFrame:
    if "rating" not in df.columns:
        return df.iloc[0:0].copy()
    s = clean_series(df["rating"]).str.lower()
    crit_words = {"critical", "c"}
    return df[s.isin(crit_words)].copy()


def filter_sox_open(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_open_issues(df)
    if "sox_type" in df.columns:
        s = clean_series(df["sox_type"]).str.lower()
        df = df[~s.eq("type 3")].copy()
    return df


def filter_overdue(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns and clean_series(df["status"], "").ne("").any():
        mask = df["status"].apply(is_overdue_status)
        if mask.any():
            return df[mask].copy()
    if "overdue_flag" in df.columns:
        mask = normalize_yes_no(df["overdue_flag"]).eq("yes")
        return df[mask].copy()
    return df.iloc[0:0].copy()


# =========================
# OUTPUT UTILS
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    return [COLOR_ROW_A if i % 2 == 0 else COLOR_ROW_B for i in range(n_rows)]


def create_placeholder_image(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.axis("off")
    ax.text(0.01, 0.90, title, fontsize=18, fontweight="bold", color=COLOR_RED, transform=ax.transAxes)
    ax.text(0.01, 0.55, message, fontsize=12, color="#333333", transform=ax.transAxes)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


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
    title_wrap_width: int = 60,
    wrap_widths: Optional[Dict[str, int]] = None,
    figsize: Tuple[float, float] = (20, 10),
):
    if df.empty:
        create_placeholder_image(path, title, "No data available for this report.")
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
        create_placeholder_image(outdir / f"{filename_prefix}_page_1.png", title, "No data available for this report.")
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
# REPORTS
# =========================
def report_metrics_summary(df: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]]]:
    data = filter_target_years(df)

    open_data = filter_open_issues(data)
    repeat_data = filter_repeat_findings(data)
    crit_data = filter_critical_findings(data)
    overdue_data = filter_overdue(data)

    rows = []
    stats = {}
    for year in TARGET_YEARS:
        rpt = int(len(repeat_data[repeat_data["report_year"] == year]))
        crt = int(len(crit_data[crit_data["report_year"] == year]))
        opn = int(len(open_data[open_data["report_year"] == year]))
        ovd = int(len(overdue_data[overdue_data["report_year"] == year]))
        rows.append({
            "Year": year,
            "Current Repeat Findings": rpt,
            "Current Critical SOX Findings": crt,
            "Current Open SOX Findings": opn,
            "Current Overdue Findings": ovd,
        })
        stats[year] = {"repeat": rpt, "critical": crt, "open": opn, "overdue": ovd}

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
    return metrics_df, stats


def chart_open_findings_by_sector_risk(df: pd.DataFrame, year: int, outdir: Path):
    year_df = filter_sox_open(filter_target_years(df))
    year_df = year_df[year_df["report_year"] == year].copy()

    outfile = outdir / f"{year}_open_findings_risk_area_trends.png"

    if year_df.empty:
        create_placeholder_image(outfile, f"{year} YTD IT SOX Open Findings Risk Area & Trends", f"No qualifying open SOX findings found for {year}.")
        return

    year_df["system_area"] = clean_series(year_df["system_area"])
    year_df["rating"] = clean_series(year_df["rating"])

    pivot = year_df.pivot_table(
        index="system_area",
        columns="rating",
        values="record_id",
        aggfunc="count",
        fill_value=0,
    )

    ordered_cols = [c for c in ["Critical", "Major", "Minor"] if c in pivot.columns]
    if not ordered_cols:
        create_placeholder_image(outfile, f"{year} YTD IT SOX Open Findings Risk Area & Trends", f"No Critical/Major/Minor severity data found for {year}.")
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
    ax.set_title(textwrap.fill(f"{year} YTD IT SOX Open Findings Risk Area & Trends", width=55), loc="left", pad=26, color=COLOR_RED, fontsize=18, fontweight="bold")
    ax.legend(title="Severity", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=11, title_fontsize=12)

    plt.subplots_adjust(top=0.88, left=0.22, right=0.86, bottom=0.06)
    fig.savefig(outfile, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_sox_open_type_breakdown(df: pd.DataFrame, year: int, outdir: Path):
    year_df = filter_sox_open(filter_target_years(df))
    year_df = year_df[year_df["report_year"] == year].copy()

    if year_df.empty:
        create_placeholder_image(outdir / f"{year}_sox_open_findings_type_breakdown.png", f"{year} YTD IT SOX Open Findings by Sector - Type Breakdown", f"No qualifying open SOX findings found for {year}.")
        return

    year_df["sector_responsible"] = clean_series(year_df["sector_responsible"])
    year_df["sox_type"] = clean_series(year_df["sox_type"])
    year_df = year_df[~year_df["sox_type"].str.lower().eq("type 3")].copy()

    pivot = year_df.pivot_table(
        index="sector_responsible",
        columns="sox_type",
        values="record_id",
        aggfunc="count",
        fill_value=0,
    )

    ordered_cols = ["Missing / Not Provided", "Type 1", "Type 2"]
    existing_cols = [c for c in ordered_cols if c in pivot.columns]
    if not existing_cols:
        create_placeholder_image(outdir / f"{year}_sox_open_findings_type_breakdown.png", f"{year} YTD IT SOX Open Findings by Sector - Type Breakdown", f"No Type 1 / Type 2 SOX data found for {year}.")
        return

    pivot = pivot[existing_cols]
    pivot["Grand Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Grand Total", ascending=False)

    grand = pivot.sum(numeric_only=True).to_frame().T
    grand.index = ["Grand Total"]
    pivot = pd.concat([pivot, grand], axis=0)

    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"sector_responsible": "Sector Responsible"})

    col_widths = [0.18] + [0.18] * (len(pivot.columns) - 1)

    render_table_image(
        pivot,
        f"{year} YTD IT SOX Open Findings by Sector - Type Breakdown",
        outdir / f"{year}_sox_open_findings_type_breakdown.png",
        col_widths=col_widths,
        font_size=12,
        row_height_scale=1.9,
        wrap_widths={"Sector Responsible": 22},
        figsize=(16, 7),
    )


def report_critical_findings(df: pd.DataFrame, outdir: Path):
    crit = filter_critical_findings(filter_target_years(df))

    cols = [
        "report_year",
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
    if not report.empty:
        report["Year"] = report["Year"].fillna("").apply(lambda x: "" if x == "" else str(int(float(x))) if pd.notna(pd.to_numeric([x], errors="coerce")[0]) else str(x))
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
    rpt = filter_repeat_findings(filter_target_years(df))
    if "sox_type" in rpt.columns:
        rpt = rpt[~clean_series(rpt["sox_type"]).str.lower().eq("type 3")].copy()

    cols = [
        "report_year",
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
    if not report.empty:
        report["Year"] = report["Year"].fillna("").apply(lambda x: "" if x == "" else str(int(float(x))) if pd.notna(pd.to_numeric([x], errors="coerce")[0]) else str(x))
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
# MAIN REPORT RUNNER
# =========================
def generate_reports(df: pd.DataFrame, output_dir: Path) -> List[str]:
    ensure_dir(output_dir)
    created = []

    # Always create all expected outputs, even if placeholders
    report_metrics_summary(df, output_dir)
    created.append("metrics_current_open_findings.png")

    for year in TARGET_YEARS:
        chart_open_findings_by_sector_risk(df, year, output_dir)
        created.append(f"{year}_open_findings_risk_area_trends.png")

        chart_sox_open_type_breakdown(df, year, output_dir)
        created.append(f"{year}_sox_open_findings_type_breakdown.png")

    report_critical_findings(df, output_dir)
    created.append("critical_it_sox_findings_page_X.png")

    report_repeat_findings(df, output_dir)
    created.append("current_repeat_it_sox_findings_page_X.png")

    return created


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate upgraded ISRM 2025/2026 SOX reporting visuals.")
    parser.add_argument("dataset", nargs="?", default=None, help="Optional path to dataset")
    parser.add_argument("--output-dir", default="isrm_reporting_output_v5", help="Output directory")
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    raw_df, sheet_used = load_dataset(dataset_path)

    df = clean_headers(raw_df)
    df = apply_aliases(df)
    df = prepare_dataframe(df)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    sample_years = []
    if "report_year" in df.columns:
        sample_years = pd.to_numeric(df["report_year"], errors="coerce").dropna().astype(int).head(20).tolist()

    created_outputs = generate_reports(df, output_dir)

    run_log = [
        f"Dataset: {dataset_path}",
        f"Sheet used: {sheet_used}",
        f"Rows loaded: {len(raw_df):,}",
        f"Columns after aliasing/prep: {len(df.columns):,}",
        f"Sample report_year values: {sample_years}",
        f"2025 rows: {int((df['report_year'] == 2025).sum()) if 'report_year' in df.columns else 0}",
        f"2026 rows: {int((df['report_year'] == 2026).sum()) if 'report_year' in df.columns else 0}",
        "",
        "Expected outputs created:",
        *[f"- {name}" for name in created_outputs],
    ]
    (output_dir / "run_log.txt").write_text("\n".join(run_log), encoding="utf-8")

    print("Report generation complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Sheet used: {sheet_used}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Sample report_year values: {sample_years}")
    print(f"2025 rows: {int((df['report_year'] == 2025).sum()) if 'report_year' in df.columns else 0}")
    print(f"2026 rows: {int((df['report_year'] == 2026).sum()) if 'report_year' in df.columns else 0}")
    print(f"Run log: {(output_dir / 'run_log.txt').resolve()}")


if __name__ == "__main__":
    main()
