#edit this to work with generated csv WITHOUT editing csv or py

"""
generic_csv_analyzer.py

Reads *any* CSV file (specified on the command line) and prints a summary report of:
  - Total number of rows
  - For each numeric column: count, sum, mean, std, min, 25%, 50%, 75%, max
  - For each non-numeric (object) column: number of unique values, most frequent value (mode)
  - OPTIONAL: Save the same report to a text file (if --output is provided)

Usage:
    python generic_csv_analyzer.py /path/to/myfile.csv
    python generic_csv_analyzer.py /path/to/myfile.csv --output /path/to/report.txt
"""

import argparse
import os
import sys
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze any CSV and produce descriptive statistics."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the CSV file to analyze."
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional path to save the text report (e.g. report.txt).",
        default=None
    )
    return parser.parse_args()


def summarize_dataframe(df: pd.DataFrame) -> str:
    """
    Build a plain-text summary of:
      - Number of rows (len(df))
      - Numeric columns: pandas .describe()
      - Object (string/categorical) columns: unique counts + top (mode)
    Returns a multi-line string.
    """
    lines = []
    n_rows = len(df)
    n_cols = len(df.columns)
    lines.append("=" * 60)
    lines.append("GENERIC CSV ANALYSIS REPORT")
    lines.append(f"Rows: {n_rows:,}    Columns: {n_cols:,}")
    lines.append("=" * 60)
    lines.append("")

    # Separate numeric vs non-numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        lines.append("→ Numeric Column Summaries (describe):")
        # pandas .describe() gives count, mean, std, min, percentiles, max
        desc = df[numeric_cols].describe().T  # transpose for easier access
        for col in numeric_cols:
            stats = desc.loc[col]
            lines.append(f"  • {col!r}:")
            lines.append(f"       count = {int(stats['count'])}")
            lines.append(f"       mean  = {stats['mean']:.4f}")
            lines.append(f"       std   = {stats['std']:.4f}")
            lines.append(f"       min   = {stats['min']}")
            lines.append(f"       25%   = {stats['25%']}")
            lines.append(f"       50%   = {stats['50%']}")
            lines.append(f"       75%   = {stats['75%']}")
            lines.append(f"       max   = {stats['max']}")
            lines.append("")
    else:
        lines.append("→ (No numeric columns detected.)")
        lines.append("")

    if non_numeric_cols:
        lines.append("→ Non‐Numeric (Categorical/Text) Column Summaries:")
        for col in non_numeric_cols:
            col_series = df[col]
            n_unique = col_series.nunique(dropna=True)
            # safe mode: if there's a mode at all
            try:
                top_val = col_series.mode(dropna=True).iloc[0]
            except IndexError:
                top_val = None
            lines.append(f"  • {col!r}:")
            lines.append(f"       unique values = {n_unique}")
            if top_val is not None:
                freq = col_series.value_counts(dropna=True).iloc[0]
                lines.append(f"       top (mode) = {top_val!r} (freq={freq})")
            else:
                lines.append(f"       (no mode; maybe all NaN or empty)")
            lines.append("")
    else:
        lines.append("→ (No non-numeric columns detected.)")
        lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    csv_path = args.csv_path
    output_path = args.output

    # Check existence
    if not os.path.isfile(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    # Attempt to read into pandas DataFrame
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}")
        sys.exit(1)

    # Build the summary text
    report_text = summarize_dataframe(df)

    # Print to console
    print(report_text)

    # If the user requested an output file, save it
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)  # create directory if needed
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")
        except Exception as e:
            print(f"WARNING: Could not write report to file: {e}")


if __name__ == "__main__":
    main()
