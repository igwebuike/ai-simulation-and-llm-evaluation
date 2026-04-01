import argparse
import math
import textwrap
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Professional default styling
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["font.family"] = "DejaVu Sans"

# J&J / business style-ish palette aligned to screenshots
RED = "#c00000"
CRITICAL_RED = "#d62728"
MAJOR_ORANGE = "#f4b400"
MINOR_YELLOW = "#fff200"
BLUE_2025 = "#4e79a7"
BLUE_2026 = "#1f4e79"
GREY = "#808080"
DARK = "#333333"
LIGHT_ROW = "#fff2f2"
WHITE = "#ffffff"

PREFERRED_SHEETS = ["RAW", "Raw", "raw", "Sheet1"]
TARGET_YEARS = [2025, 2026]
MISSING = "Missing / Not Provided"

COLUMN_ALIASES = {
    "issue_id": ["IssueId", "Issue_ID", "Issue ID", "ID", "Record_ID"],
    "audit_year": ["Audit_Year"],
    "release_date": ["Release_Date"],
    "create_date": ["Create_Date"],
    "issue_creation_date": ["Issue_Creation_Date"],
    "expected_due_date": ["Expected_Due_Date", "Due_Date_Sld"],
    "revised_due_date": ["Revised_Due_Date", "Revised_Remediation_Date", "Initial_Agreed_Remediation_Date"],
    "issue_closed_date": ["Issue_Closed_Date"],
    "days_old": ["Days_Old"],
    "days_until_due": ["Days_Until_Due"],
    "rating": ["Rating", "Severity", "Issue Rating", "Alert_1_Rating_Verbal"],
    "sox_type": ["SOX_Type"],
    "entity_sector": ["Entity_Sector", "Impacted_Sector", "IT_Asset_Accountable_Sector"],
    "area": ["Area", "Control_Category", "Risk_Category"],
    "repeat_finding": ["Repeat_Finding", "Repeat Finding"],
    "remediation_status": ["Remediation Status", "Recommendation_State", "Issue_Status_SOX_Short", "Issue_Status_Sld", "Issue_Status"],
    "issue_status_short": ["Issue_Status_SOX_Short", "Issue_Status_Sld", "Issue_Status"],
    "issue_status_cert": ["Issue_Status_SOX_Cert"],
    "overdue_flag": ["Overdue_Flag", "Open_And_Overdue_Flag", "Flag_3_Open_and_Overdue"],
    "common_insights": ["Common_Insights"],
    "root_cause": ["Root_Cause_Description", "Root_Cause_Insights"],
    # Business_Contact_Recommendations is intentionally first because user said it is the true control owner alert field
    "control_owner": [
        "Business_Contact_Recommendations",
        "Business_Contact_Issue",
        "Accountable_Contact_Issue",
        "Manager_Issue",
        "MRC_Company_Contact",
    ],
    "finding_title": ["Issue_Finding_Title", "Summary_Finding_Exec_Summary"],
    "finding": ["Finding", "Summary_Finding_Exec_Summary"],
    "final_month_snapshot": ["Final_Month_Snapshot"],
    "issue_finding_source": ["Issue_Finding_Source"],
}


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
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), "CSV"
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
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

    if "issue_id" not in df.columns:
        df["issue_id"] = range(1, len(df) + 1)

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "audit_year", "release_date", "create_date", "issue_creation_date",
        "expected_due_date", "revised_due_date", "issue_closed_date", "final_month_snapshot"
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_text(s: pd.Series, fill: str = MISSING) -> pd.Series:
    return (
        s.fillna(fill)
        .astype(str)
        .str.strip()
        .replace("", fill)
        .replace("nan", fill)
        .replace("NaN", fill)
        .replace("<NA>", fill)
    )


def normalize_yes_no(s: pd.Series) -> pd.Series:
    out = normalize_text(s).str.lower()
    return out.replace({"true": "yes", "false": "no", "1": "yes", "0": "no", "y": "yes", "n": "no", "x": "yes"})


def standardize_rating(s: pd.Series) -> pd.Series:
    s = normalize_text(s)
    mapping = {
        "critical": "Critical",
        "major": "Major",
        "minor": "Minor",
        "medium": "Major",
        "moderate": "Major",
        "low": "Minor",
    }
    lowered = s.str.strip().str.lower()
    return lowered.map(mapping).fillna(s)


def rating_colors(columns: List[str]) -> List[str]:
    cmap = {"Critical": CRITICAL_RED, "Major": MAJOR_ORANGE, "Minor": MINOR_YELLOW}
    return [cmap.get(c, GREY) for c in columns]


def infer_year(df: pd.DataFrame) -> pd.Series:
    for col in ["final_month_snapshot", "issue_creation_date", "release_date", "create_date", "audit_year"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                return dt.dt.year
    raise KeyError("Could not infer year. Expected one of: final_month_snapshot, issue_creation_date, release_date, create_date, audit_year")


def infer_month_date(df: pd.DataFrame) -> pd.Series:
    for col in ["final_month_snapshot", "issue_creation_date", "release_date", "create_date"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                return dt
    raise KeyError("Could not infer chart date field.")


def filter_target_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    out = df.copy()
    out["report_year"] = infer_year(out)
    out = out[out["report_year"].isin(years)].copy()
    out["report_month_date"] = infer_month_date(out)
    out["rating_std"] = standardize_rating(out["rating"]) if "rating" in out.columns else MISSING
    if "control_owner" in out.columns:
        out["control_owner"] = normalize_text(out["control_owner"])
    return out


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def wrap_label(text: str, width: int = 24) -> str:
    txt = str(text)
    return "\n".join(textwrap.wrap(txt, width=width, break_long_words=False, break_on_hyphens=False)) if len(txt) > width else txt


def shorten_cell(text: str, width: int = 44) -> str:
    txt = " ".join(str(text).split())
    if txt in {"", "nan", "NaN", "None"}:
        return ""
    return "\n".join(textwrap.wrap(txt, width=width, break_long_words=False, break_on_hyphens=False))


def style_axes(ax, title: str, subtitle: Optional[str] = None, xlabel: str = "", ylabel: str = ""):
    wrapped_title = wrap_label(title, 55)
    ax.set_title(wrapped_title, loc="left", color=RED, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x" if ylabel == "" else "y", alpha=0.22)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=DARK)


def finalize_figure(fig, top: float = 0.88):
    fig.subplots_adjust(top=top, left=0.12, right=0.88, bottom=0.12)


def save_kpi_table(kpis: List[Tuple[str, str]], path: Path, title: str):
    df = pd.DataFrame(kpis, columns=["Metric", "Value"])
    fig, ax = plt.subplots(figsize=(8.3, max(3.2, 0.58 * len(df) + 1.4)))
    ax.axis("off")
    ax.set_title(wrap_label(title, 48), fontsize=14, fontweight="bold", loc="left", color=RED, pad=12)

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="left", colLoc="left", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    col_widths = {0: 0.74, 1: 0.18}
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if col in col_widths:
            cell.set_width(col_widths[col])
        if row == 0:
            cell.set_facecolor(RED)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.08)
        else:
            cell.set_facecolor("#f9f9f9" if row % 2 else "white")
            cell.set_height(0.075)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_detail_table_pages(
    df: pd.DataFrame,
    outdir: Path,
    base_filename: str,
    title: str,
    rows_per_page: int = 14,
    font_size: int = 8,
    row_height: float = 0.072,
    wrap_widths: Optional[Dict[str, int]] = None,
):
    if df.empty:
        return []

    display = df.copy().fillna("")
    wrap_widths = wrap_widths or {}
    for col in display.columns:
        width = wrap_widths.get(col, 30)
        display[col] = display[col].map(lambda x: shorten_cell(x, width))

    # practical width allocation for business-facing detail tables
    col_width_map = {
        "Year": 0.05,
        "Finding Summary": 0.16,
        "System & Control": 0.10,
        "System & Control / Area": 0.11,
        "Sector Responsible": 0.11,
        "Control Owner": 0.11,
        "Finding Root Cause": 0.17,
        "Release Date": 0.08,
        "Action Plan Remediation Date": 0.10,
        "Remediation Date": 0.09,
        "Actual Status": 0.09,
        "Status": 0.08,
        "SOX Type": 0.07,
    }

    pages = []
    total_pages = math.ceil(len(display) / rows_per_page)
    for page_idx in range(total_pages):
        start = page_idx * rows_per_page
        end = start + rows_per_page
        part = display.iloc[start:end].copy()

        fig_h = max(6.3, len(part) * 0.55 + 1.8)
        fig, ax = plt.subplots(figsize=(19, fig_h))
        ax.axis("off")
        page_title = f"{title} (Page {page_idx + 1} of {total_pages})" if total_pages > 1 else title
        ax.set_title(wrap_label(page_title, 65), fontsize=14, fontweight="bold", loc="left", color=RED, pad=12)

        table = ax.table(
            cellText=part.values,
            colLabels=part.columns,
            cellLoc="left",
            colLoc="left",
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.18)

        for (row, col), cell in table.get_celld().items():
            header = row == 0
            cell.set_linewidth(0.75)
            col_name = part.columns[col]
            cell.set_width(col_width_map.get(col_name, 0.09))
            if header:
                cell.set_facecolor(RED)
                cell.set_text_props(color="white", weight="bold", va="center")
                cell.set_height(row_height + 0.01)
            else:
                cell.set_facecolor(LIGHT_ROW if row % 2 else WHITE)
                cell.set_text_props(color=DARK, va="top")
                cell.set_height(row_height)

        plt.tight_layout()
        file_path = outdir / f"{base_filename}_page_{page_idx + 1}.png"
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        pages.append(file_path)

    return pages


def chart_metrics_summary(df: pd.DataFrame, outdir: Path):
    rows = []
    for year in TARGET_YEARS:
        d = df[df["report_year"] == year].copy()
        total_repeat = int((normalize_yes_no(d["repeat_finding"]) == "yes").sum()) if "repeat_finding" in d.columns else 0
        critical_sox = int((d["rating_std"] == "Critical").sum())
        critical_non_sox = 0
        if "sox_type" in d.columns:
            sox_type = normalize_text(d["sox_type"]).str.lower()
            critical_non_sox = int(((d["rating_std"] == "Critical") & ~sox_type.str.contains("type", na=False)).sum())
        overdue = int((normalize_yes_no(d["overdue_flag"]) == "yes").sum()) if "overdue_flag" in d.columns else 0
        rows.extend([
            (f"{year} Current Repeat Findings", f"{total_repeat:,}"),
            (f"{year} Current Critical SOX Findings", f"{critical_sox:,}"),
            (f"{year} Current Critical Non-SOX Findings", f"{critical_non_sox:,}"),
            (f"{year} Current Overdue Findings", f"{overdue:,}"),
        ])
    save_kpi_table(rows, outdir / "01_metrics_summary_2025_2026.png", "Metrics – Current Open SOX / Audit Findings (2025 and 2026)")


def chart_open_findings_by_sector(df: pd.DataFrame, outdir: Path):
    if "entity_sector" not in df.columns:
        return
    temp = df.copy()
    temp["entity_sector"] = normalize_text(temp["entity_sector"])
    temp["entity_sector_label"] = temp["entity_sector"].map(lambda x: wrap_label(x, 24))
    temp["sox_type_std"] = normalize_text(temp["sox_type"]) if "sox_type" in temp.columns else "Unspecified"

    for year in TARGET_YEARS:
        d = temp[temp["report_year"] == year].copy()
        if d.empty:
            continue
        pivot = d.pivot_table(index="entity_sector_label", columns="rating_std", values="issue_id", aggfunc="count", fill_value=0)
        ordered_cols = [c for c in ["Critical", "Major", "Minor"] if c in pivot.columns] + [c for c in pivot.columns if c not in ["Critical", "Major", "Minor"]]
        pivot = pivot[ordered_cols]
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=True).drop(columns=["Total"])

        fig, ax = plt.subplots(figsize=(13.5, max(6.5, 0.62 * len(pivot) + 2.7)))
        left = np.zeros(len(pivot))
        y = np.arange(len(pivot))
        cols = list(pivot.columns)
        for col, color in zip(cols, rating_colors(cols)):
            vals = pivot[col].values
            bars = ax.barh(y, vals, left=left, label=col, color=color)
            for bar, v, l in zip(bars, vals, left):
                if v > 0:
                    ax.text(l + v / 2, bar.get_y() + bar.get_height() / 2, f"{int(v)}", ha="center", va="center", fontsize=9)
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index)
        style_axes(
            ax,
            f"{year} YTD IT SOX Open Findings by Sector",
            subtitle=f"Reporting year explicitly shown: {year} | Professional severity view",
            xlabel="Open Finding Count",
        )
        ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
        finalize_figure(fig, top=0.84)
        fig.savefig(outdir / f"02_open_findings_by_sector_{year}.png", bbox_inches="tight")
        plt.close(fig)

        if "sox_type" in d.columns:
            tbl = d.pivot_table(index="entity_sector", columns="sox_type_std", values="issue_id", aggfunc="count", fill_value=0)
            tbl["Grand Total"] = tbl.sum(axis=1)
            grand = pd.DataFrame(tbl.sum(axis=0)).T
            grand.index = ["Grand Total"]
            tbl = pd.concat([tbl, grand], axis=0)
            tbl = tbl.reset_index().rename(columns={"entity_sector": "Sector"})
            save_detail_table_pages(
                tbl,
                outdir,
                f"03_open_findings_sector_type_table_{year}",
                f"{year} YTD IT SOX Open Findings by Sector – Type Breakdown",
                rows_per_page=18,
                font_size=9,
                row_height=0.065,
                wrap_widths={"Sector": 22},
            )


def chart_risk_area_and_quarterly_trends(df: pd.DataFrame, outdir: Path):
    if "area" not in df.columns:
        return
    temp = df.copy()
    temp["area"] = normalize_text(temp["area"])
    temp["area_label"] = temp["area"].map(lambda x: wrap_label(x, 26))
    temp["quarter"] = temp["report_month_date"].dt.to_period("Q").astype(str)
    temp = temp[temp["quarter"].str.startswith(("2025", "2026"))]

    for year in TARGET_YEARS:
        d = temp[temp["report_year"] == year].copy()
        if d.empty:
            continue

        risk = d.pivot_table(index="area_label", columns="rating_std", values="issue_id", aggfunc="count", fill_value=0)
        risk = risk[[c for c in ["Critical", "Major", "Minor"] if c in risk.columns]]
        risk["Total"] = risk.sum(axis=1)
        risk = risk.sort_values("Total", ascending=True).tail(10).drop(columns=["Total"])

        fig, ax = plt.subplots(figsize=(14, max(6.5, 0.62 * len(risk) + 2.6)))
        left = np.zeros(len(risk))
        y = np.arange(len(risk))
        cols = list(risk.columns)
        for col, color in zip(cols, rating_colors(cols)):
            vals = risk[col].values
            bars = ax.barh(y, vals, left=left, label=col, color=color)
            for bar, v, l in zip(bars, vals, left):
                if v > 0:
                    ax.text(l + v / 2, bar.get_y() + bar.get_height() / 2, f"{int(v)}", ha="center", va="center", fontsize=9)
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels(risk.index)
        style_axes(
            ax,
            f"{year} YTD IT SOX Open Findings Risk Area & Trends",
            subtitle=f"Top risk areas for {year} | Year explicitly shown on chart",
            xlabel="Finding Count",
        )
        ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
        finalize_figure(fig, top=0.84)
        fig.savefig(outdir / f"04_risk_area_by_severity_{year}.png", bbox_inches="tight")
        plt.close(fig)

    quarter_pivot = temp.pivot_table(index="quarter", columns="rating_std", values="issue_id", aggfunc="count", fill_value=0)
    if not quarter_pivot.empty:
        ordered_index = sorted(quarter_pivot.index.tolist())
        quarter_pivot = quarter_pivot.loc[ordered_index]
        cols = [c for c in ["Critical", "Major", "Minor"] if c in quarter_pivot.columns]
        fig, ax = plt.subplots(figsize=(12.8, 6.0))
        for col, color in zip(cols, rating_colors(cols)):
            ax.plot(quarter_pivot.index, quarter_pivot[col], marker="o", linewidth=2.4, label=col, color=color)
            for x, y in zip(quarter_pivot.index, quarter_pivot[col]):
                ax.text(x, y + 0.2, f"{int(y)}", ha="center", va="bottom", fontsize=8)
        style_axes(
            ax,
            "Quarterly Trend of Open Findings (2025–2026)",
            subtitle="Quarter labels explicitly show both 2025 and 2026",
            xlabel="Quarter",
            ylabel="Open Finding Count",
        )
        ax.legend(title="Severity")
        ax.tick_params(axis="x", rotation=0)
        finalize_figure(fig, top=0.84)
        fig.savefig(outdir / "05_quarterly_trend_2025_2026.png", bbox_inches="tight")
        plt.close(fig)


def chart_issue_aging(df: pd.DataFrame, outdir: Path):
    if "days_old" not in df.columns:
        return
    temp = df.copy()
    temp["days_old_num"] = pd.to_numeric(temp["days_old"], errors="coerce")
    temp = temp[temp["days_old_num"].notna()].copy()
    if temp.empty:
        return

    bins = [-np.inf, 30, 60, 90, 180, 365, np.inf]
    labels = ["0-30", "31-60", "61-90", "91-180", "181-365", "366+"]
    temp["age_bucket"] = pd.cut(temp["days_old_num"], bins=bins, labels=labels)

    pivot = temp.pivot_table(index="age_bucket", columns="report_year", values="issue_id", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(labels)
    years_present = [y for y in TARGET_YEARS if y in pivot.columns]
    if not years_present:
        return

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(pivot.index))
    width = 0.34
    year_colors = {2025: BLUE_2025, 2026: RED}
    for i, year in enumerate(years_present):
        offset = (i - (len(years_present) - 1) / 2) * width
        bars = ax.bar(x + offset, pivot[year].values, width=width, label=str(year), color=year_colors.get(year, GREY))
        for bar, val in zip(bars, pivot[year].values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{int(val)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str))
    style_axes(
        ax,
        "Issue Aging Distribution (2025 vs 2026)",
        subtitle="Age buckets compare open findings across both years",
        xlabel="Age Bucket",
        ylabel="Finding Count",
    )
    ax.legend(title="Year")
    finalize_figure(fig, top=0.84)
    fig.savefig(outdir / "06_issue_aging_2025_2026.png", bbox_inches="tight")
    plt.close(fig)


def detail_critical_findings(df: pd.DataFrame, outdir: Path):
    d = df[df["rating_std"] == "Critical"].copy()
    if d.empty:
        return
    cols = []
    preferred = [
        ("report_year", "Year"),
        ("finding_title", "Finding Summary"),
        ("area", "System & Control / Area"),
        ("entity_sector", "Sector Responsible"),
        ("control_owner", "Control Owner"),
        ("root_cause", "Finding Root Cause"),
        ("release_date", "Release Date"),
        ("revised_due_date", "Action Plan Remediation Date"),
        ("remediation_status", "Status"),
    ]
    rename = {}
    for src, label in preferred:
        if src in d.columns:
            cols.append(src)
            rename[src] = label
    tbl = d[cols].copy().rename(columns=rename)
    if "Release Date" in tbl.columns:
        tbl["Release Date"] = pd.to_datetime(tbl["Release Date"], errors="coerce").dt.strftime("%m/%d/%Y")
    if "Action Plan Remediation Date" in tbl.columns:
        tbl["Action Plan Remediation Date"] = pd.to_datetime(tbl["Action Plan Remediation Date"], errors="coerce").dt.strftime("%m/%d/%Y")
    if "Control Owner" in tbl.columns:
        tbl["Control Owner"] = tbl["Control Owner"].replace(MISSING, "")
    sort_cols = [c for c in ["Year", "Release Date"] if c in tbl.columns]
    tbl = tbl.sort_values(sort_cols, ascending=True)
    save_detail_table_pages(
        tbl,
        outdir,
        "07_critical_findings_detail_2025_2026",
        "Critical IT SOX Findings – Includes Release Date and Control Owner",
        rows_per_page=12,
        font_size=8,
        row_height=0.09,
        wrap_widths={
            "Finding Summary": 34,
            "System & Control / Area": 20,
            "Sector Responsible": 18,
            "Control Owner": 18,
            "Finding Root Cause": 34,
            "Status": 16,
        },
    )


def detail_repeat_findings(df: pd.DataFrame, outdir: Path):
    if "repeat_finding" not in df.columns:
        return
    mask = normalize_yes_no(df["repeat_finding"]) == "yes"
    d = df[mask].copy()
    if d.empty:
        return
    cols = []
    preferred = [
        ("report_year", "Year"),
        ("finding_title", "Finding Summary"),
        ("area", "System & Control"),
        ("entity_sector", "Sector Responsible"),
        ("control_owner", "Control Owner"),
        ("root_cause", "Finding Root Cause"),
        ("remediation_status", "Actual Status"),
        ("revised_due_date", "Remediation Date"),
        ("sox_type", "SOX Type"),
    ]
    rename = {}
    for src, label in preferred:
        if src in d.columns:
            cols.append(src)
            rename[src] = label
    tbl = d[cols].copy().rename(columns=rename)
    if "Remediation Date" in tbl.columns:
        tbl["Remediation Date"] = pd.to_datetime(tbl["Remediation Date"], errors="coerce").dt.strftime("%m/%d/%Y")
    if "Actual Status" in tbl.columns:
        tbl["Actual Status"] = tbl["Actual Status"].replace(MISSING, "")
    if "Control Owner" in tbl.columns:
        tbl["Control Owner"] = tbl["Control Owner"].replace(MISSING, "")
    sort_cols = [c for c in ["Year", "Remediation Date"] if c in tbl.columns]
    tbl = tbl.sort_values(sort_cols, ascending=True)
    save_detail_table_pages(
        tbl,
        outdir,
        "08_repeat_findings_detail_2025_2026",
        "Current Repeat IT SOX Findings – Control Owner and Actual Status Included",
        rows_per_page=12,
        font_size=8,
        row_height=0.09,
        wrap_widths={
            "Finding Summary": 34,
            "System & Control": 20,
            "Sector Responsible": 18,
            "Control Owner": 18,
            "Finding Root Cause": 34,
            "Actual Status": 16,
            "SOX Type": 12,
        },
    )


def create_run_log(df: pd.DataFrame, outdir: Path, dataset_path: Path, sheet_name: str):
    years = df["report_year"].value_counts().sort_index().to_dict()
    lines = [
        f"Dataset: {dataset_path}",
        f"Sheet used: {sheet_name}",
        f"Rows after year filter: {len(df):,}",
        f"Columns after alias normalization: {len(df.columns):,}",
        f"Target years: {TARGET_YEARS}",
        f"Rows by year: {years}",
        "",
        "Key fixes in v2:",
        "- Fixed chart title/header overlap by increasing top margin and wrapping long titles.",
        "- Control Owner now prioritizes Business_Contact_Recommendations.",
        "- Repeat and critical detail reports are paginated for readability.",
        "- Table cells are wrapped and row heights expanded for professional output.",
    ]
    (outdir / "run_log.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate professional ISRM graphs aligned to the PowerPoint style for 2025 and 2026.")
    parser.add_argument("dataset", nargs="?", default=None, help="Optional path to CSV/XLSX dataset")
    parser.add_argument("--output-dir", default="isrm_ppt_graphs_2025_2026_v2", help="Output directory")
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    raw_df, sheet_name = load_dataset(dataset_path)
    df = clean_headers(raw_df)
    df = apply_aliases(df)
    df = parse_dates(df)

    filtered = filter_target_years(df, TARGET_YEARS)
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    chart_metrics_summary(filtered, outdir)
    chart_open_findings_by_sector(filtered, outdir)
    chart_risk_area_and_quarterly_trends(filtered, outdir)
    chart_issue_aging(filtered, outdir)
    detail_critical_findings(filtered, outdir)
    detail_repeat_findings(filtered, outdir)
    create_run_log(filtered, outdir, dataset_path, sheet_name)

    print("Professional ISRM chart generation complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Sheet used: {sheet_name}")
    print(f"Rows after filter: {len(filtered):,}")
    print(f"Output folder: {outdir.resolve()}")
    print(f"Run log: {(outdir / 'run_log.txt').resolve()}")


if __name__ == "__main__":
    main()
