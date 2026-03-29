
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


DATE_CANDIDATES = [
    "Release_Date",
    "Create_Date",
    "Issue_Creation_Date",
    "Expected_Closure_Date",
    "Issue_Closed_Date",
    "Final_Month_Snapshot",
]

COLUMN_ALIASES = {
    "Record_ID": ["Record_ID", "record_id", "ID", "Issue_ID"],
    "MRC_ID": ["MRC_ID", "MRC Id", "MRC", "MRC_Reference"],
    "Issue_Finding_Type": [
        "Issue_Finding_Type", "Issue Finding Type", "Issue_Finding_Typ",
        "Issue_Finding_Ty", "Issue_Finding"
    ],
    "Manager_Issue": ["Manager_Issue", "Manager Issue", "Control Owner", "Issue Owner"],
    "Global_Control_Reference": [
        "Global_Control_Reference", "Global Control Reference", "Global_Contr",
        "Global_Control", "Control Reference", "Global_Contro"
    ],
    "Repeat_Finding": ["Repeat_Finding", "Repeat Finding", "Repeat_Finding_Flag"],
    "Oversight_Error_Flag": [
        "Oversight_Error_Flag", "Oversight Error Flag", "Oversight_Error",
        "Key_Control_Oversight_Error"
    ],
    "SOX_Certificate_Issue_Flag": [
        "SOX_Certificate_Issue_Flag", "SOX Certificate Issue Flag",
        "SOX_Certificate_Issue", "SOX_Cert_Issue_Flag"
    ],
    "Rating": ["Rating", "Severity", "Issue Rating"],
    "Snapshot_Year": ["Snapshot_Year", "Year", "Snapshot Year", "Repeat_Year"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.strip().lower()
            if key in lower_map:
                rename_map[lower_map[key]] = canonical
                break

    df = df.rename(columns=rename_map)

    # Fill required helper columns where possible
    if "Record_ID" not in df.columns:
        df["Record_ID"] = range(1, len(df) + 1)

    if "Snapshot_Year" not in df.columns:
        if "Final_Month_Snapshot" in df.columns:
            df["Snapshot_Year"] = pd.to_datetime(df["Final_Month_Snapshot"], errors="coerce").dt.year
        elif "Issue_Creation_Date" in df.columns:
            df["Snapshot_Year"] = pd.to_datetime(df["Issue_Creation_Date"], errors="coerce").dt.year
        elif "Create_Date" in df.columns:
            df["Snapshot_Year"] = pd.to_datetime(df["Create_Date"], errors="coerce").dt.year

    return df


def auto_find_dataset(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    here = Path(__file__).resolve().parent
    candidates = [
        here / "isrm_sample_dataset.xlsx",
        here / "isrm_sample_dataset.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    xlsx_files = sorted(here.glob("*.xlsx"))
    csv_files = sorted(here.glob("*.csv"))
    for group in (xlsx_files, csv_files):
        if group:
            return group[0]

    raise FileNotFoundError(
        "No dataset file found in the script folder. Put an .xlsx or .csv file there, "
        "or pass the path explicitly."
    )


def load_excel_dataset(path: Path) -> pd.DataFrame:
    # Try RAW first, then first sheet
    xls = pd.ExcelFile(path)
    preferred_sheet = "RAW" if "RAW" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=preferred_sheet)
    return df


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        df = load_excel_dataset(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df = normalize_columns(df)

    for col in DATE_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Standardize Yes/No-like flags where present
    for col in ["Repeat_Finding", "Oversight_Error_Flag", "SOX_Certificate_Issue_Flag"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def require_columns(df: pd.DataFrame, needed: list[str], chart_name: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"{chart_name} cannot run because these columns are missing: {missing}. "
            f"Available columns are: {list(df.columns)}"
        )


def save_table_image(df, title, path, fontsize=10, scale=(1, 1.4), highlight_max=False):
    fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.8), max(3, df.shape[0] * 0.55)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=12)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(*scale)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E78")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#F3F6FA")
            cell.set_text_props(weight="bold")

    if highlight_max and df.shape[0] > 0 and df.shape[1] > 1:
        vals = df.iloc[:, 1:].copy()
        vals_numeric = vals.apply(pd.to_numeric, errors="coerce")
        max_val = np.nanmax(vals_numeric.values) if vals_numeric.size else np.nan
        if pd.notna(max_val):
            for i in range(vals_numeric.shape[0]):
                for j in range(vals_numeric.shape[1]):
                    if pd.notna(vals_numeric.iat[i, j]) and vals_numeric.iat[i, j] == max_val:
                        table[(i + 1, j + 1)].set_facecolor("#F9E79F")

    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def chart_issues_summary_by_type_grouped_by_mrc(df, outdir: Path):
    require_columns(df, ["MRC_ID", "Issue_Finding_Type", "Record_ID"], "Issues Summary by Type")
    pivot = df.pivot_table(
        index="MRC_ID",
        columns="Issue_Finding_Type",
        values="Record_ID",
        aggfunc="count",
        fill_value=0,
    )

    desired = [c for c in ["Operating Effectiveness", "Design Effectiveness", "Documentation"] if c in pivot.columns]
    if desired:
        pivot = pivot[desired]
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", stacked=True, ax=ax)
    ax.set_title("Issues Summary by Type, grouped by MRC", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel("Issue Count")
    ax.set_ylabel("MRC")
    ax.legend(title="Issue Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(outdir / "01_issues_summary_by_type_grouped_by_mrc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def chart_repeat_findings_by_owner_and_control(df, outdir: Path):
    require_columns(df, ["Repeat_Finding", "Manager_Issue", "Global_Control_Reference", "Record_ID"], "Repeat Findings")
    rpt = df[df["Repeat_Finding"].astype(str).str.upper() == "YES"]
    pivot = rpt.pivot_table(
        index="Manager_Issue",
        columns="Global_Control_Reference",
        values="Record_ID",
        aggfunc="count",
        fill_value=0,
    ).reset_index()
    save_table_image(
        pivot,
        "Repeat Findings by Control Owner, grouped by Control Reference",
        outdir / "02_repeat_findings_by_control_owner_grouped_by_control_reference.png",
        highlight_max=True,
    )


def chart_oversight_error_by_owner_and_control(df, outdir: Path):
    require_columns(df, ["Oversight_Error_Flag", "Manager_Issue", "Global_Control_Reference", "Record_ID"], "Oversight Error")
    ov = df[df["Oversight_Error_Flag"].astype(str).str.upper() == "YES"]
    pivot = ov.pivot_table(
        index="Manager_Issue",
        columns="Global_Control_Reference",
        values="Record_ID",
        aggfunc="count",
        fill_value=0,
    ).reset_index()
    save_table_image(
        pivot,
        "Oversight Error by Control Owner, grouped by Control Reference",
        outdir / "03_oversight_error_by_control_owner_grouped_by_control_reference.png",
        highlight_max=True,
    )


def chart_total_issues_by_control_reference(df, outdir: Path):
    require_columns(df, ["Global_Control_Reference"], "Total Issues by Control Reference")
    counts = df["Global_Control_Reference"].value_counts().rename_axis("Global_Control_Reference").reset_index(name="Issue Count")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(counts["Global_Control_Reference"], counts["Issue Count"])
    ax.invert_yaxis()
    ax.set_title("Total Number of Issues by Control Reference", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel("Issue Count")
    top3 = counts.head(3).index.tolist()
    for i, b in enumerate(bars):
        if i in top3:
            b.set_alpha(0.9)
            b.set_hatch("//")
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2, f"{int(b.get_width())}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(outdir / "04_total_number_of_issues_by_control_reference.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def chart_sox_yoy_summary(df, outdir: Path):
    require_columns(df, ["SOX_Certificate_Issue_Flag", "Rating", "Snapshot_Year", "Record_ID"], "SOX YoY Summary")
    sox = df[df["SOX_Certificate_Issue_Flag"].astype(str).str.upper() == "YES"]
    summary = sox.pivot_table(
        index="Rating",
        columns="Snapshot_Year",
        values="Record_ID",
        aggfunc="count",
        fill_value=0,
    )
    order = [idx for idx in ["Critical", "Major", "Minor"] if idx in summary.index]
    if order:
        summary = summary.reindex(order)
    summary.loc["Total"] = summary.sum(axis=0)
    years = list(summary.columns)
    if len(years) >= 2:
        latest = years[-1]
        prev = years[-2]
        yoy = ((summary[latest] - summary[prev]) / summary[prev].replace(0, np.nan)).round(3)
        summary["YoY vs Prior"] = yoy.map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
    table_df = summary.reset_index().rename(columns={"Rating": "Severity", "index": "Severity"})
    save_table_image(
        table_df,
        "SOX Certificate Issues, YoY",
        outdir / "05_sox_certificate_issues_yoy_summary.png",
        highlight_max=False,
    )


def chart_sox_yoy_all_sectors(df, outdir: Path):
    require_columns(df, ["SOX_Certificate_Issue_Flag", "Snapshot_Year", "Rating", "Record_ID"], "SOX YoY All Sectors")
    sox = df[df["SOX_Certificate_Issue_Flag"].astype(str).str.upper() == "YES"]
    summary = sox.pivot_table(
        index="Snapshot_Year",
        columns="Rating",
        values="Record_ID",
        aggfunc="count",
        fill_value=0,
    )
    summary = summary[[c for c in ["Critical", "Major", "Minor"] if c in summary.columns]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col in summary.columns:
        ax.plot(summary.index, summary[col], marker="o", linewidth=2, label=col)
    ax.set_title("SOX Certificate Issues, YoY – All Sectors", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Issue Count")
    ax.legend(title="Severity")
    ax.set_xticks(summary.index.tolist())
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(outdir / "06_sox_certificate_issues_yoy_all_sectors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate ISRM-style charts from a CSV or Excel dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional path to the CSV or XLSX dataset. If omitted, the script auto-searches the current folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_charts",
        help="Directory where chart PNGs will be written.",
    )
    args = parser.parse_args()

    dataset_path = auto_find_dataset(args.dataset)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    print(f"Loaded dataset: {dataset_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    chart_issues_summary_by_type_grouped_by_mrc(df, outdir)
    chart_repeat_findings_by_owner_and_control(df, outdir)
    chart_oversight_error_by_owner_and_control(df, outdir)
    chart_total_issues_by_control_reference(df, outdir)
    chart_sox_yoy_summary(df, outdir)
    chart_sox_yoy_all_sectors(df, outdir)

    print(f"Charts written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
