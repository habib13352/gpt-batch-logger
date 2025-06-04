# edit this to work with generated csv WITHOUT editing csv or py

# this ones a mess with the naming of the cols. fix it later.

"""
report_builder_v2.py

A “v2” report generator that computes more advanced metrics from a GPT batch CSV
and produces either an HTML report (with embedded Base64 charts) or a Markdown report
(with separate PNG files).

Key features:
1) Cost‐per‐token calculations.
2) Prompt vs. completion token ratios.
3) Percentile tables for cost and total tokens.
4) Correlation matrix among numeric metrics (and optional heatmap).
5) Before‐vs‐After optimization comparison (if a grouping column exists).
6) Near‐duplicate prompt detection.
7) Multiple CLI filters (min cost, keyword search).
8) Choice of charts via `--charts`.
9) Outputs either HTML (with inline images) or Markdown (with separate PNGs).

Usage examples (all one‐line):
  # Full HTML with all features:
  python report_builder_v2.py \
    -i results/summary_log_v3.csv \
    -f html \
    -o batch_report_v2.html \
    --title "GPT-3.5 Batch Report v2" \
    --charts scatter_cost_tokens hist_response_length hist_cost_per_token scatter_prompt_cost heatmap_corr \
    --top-n 5 \
    --min-cost 0.0002 \
    --keyword "error" \
    --opt-flag-col optimized

  # Markdown-only with just two charts:
  python report_builder_v2.py \
    -i results/summary_log_v3.csv \
    -f md \
    -o batch_report_v2.md \
    --charts hist_cost_per_token scatter_cost_tokens \
    --title "My V2 Report (Markdown)"

-------------------------------------------------------------
"""

import argparse
import pandas as pd                # DataFrame manipulations
import matplotlib.pyplot as plt    # Chart generation
import os                          # Filepath handling
import io                          # In-memory bytes buffer for Base64
import base64                      # Base64 encoding for embedding images
from datetime import datetime      # Timestamping in report
from difflib import SequenceMatcher  # For near-duplicate prompt detection

# ───────────────────────────────────────────────────────────────
# 1) DATA LOADING & INITIAL CLEANUP
# ───────────────────────────────────────────────────────────────
def load_data(path):
    """
    1) Read the CSV from `path`.
    2) Ensure a 'cost' column exists; strip out '$' and ',' characters.
    3) Convert 'cost' to float; drop any rows where cost cannot be parsed.
    4) Return the cleaned DataFrame.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"❌ Error reading CSV '{path}': {e}")

    # Ensure 'cost' column is present
    if 'cost' not in df.columns:
        raise SystemExit("❌ CSV must have a column named 'cost'.")

    # 2) Remove any '$' or ',' so we can convert to numeric
    df['cost'] = (
        df['cost']
        .astype(str)                  # In case it's already string or mixed
        .str.replace(r'[\$,]', '', regex=True)
    )

    # 3) Convert to float; any parsing error → NaN → drop those rows
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df = df.dropna(subset=['cost'])

    return df

# ───────────────────────────────────────────────────────────────
# 2) DERIVED COLUMNS
# ───────────────────────────────────────────────────────────────
def compute_response_length(df):
    """
    If 'response' column exists, add a 'response_length' column (character count).
    Otherwise, leave DataFrame unchanged.
    """
    if 'response' in df.columns:
        df['response_length'] = df['response'].astype(str).str.len()
    return df

def compute_efficiency_metrics(df):
    """
    Compute “efficiency” metrics:
      - cost_per_token         = cost / total_tokens
      - cost_per_1k_tokens     = cost_per_token * 1000
      - prompt_completion_ratio = prompt_tokens / completion_tokens (NaN if completion_tokens==0)
      - prompt_fraction        = prompt_tokens / total_tokens
      - completion_fraction    = completion_tokens / total_tokens

    These columns will be appended to df if the requisite source columns exist.
    """
    # Only compute cost_per_token if 'total_tokens' is present and > 0
    if 'total_tokens' in df.columns:
        df['cost_per_token'] = df['cost'] / df['total_tokens']
        df['cost_per_1k_tokens'] = df['cost_per_token'] * 1000

    # Prompt vs completion ratio (avoid division by zero)
    if 'prompt_tokens' in df.columns and 'completion_tokens' in df.columns:
        df['prompt_completion_ratio'] = df.apply(
            lambda r: (r['prompt_tokens'] / r['completion_tokens'])
            if r['completion_tokens'] and r['completion_tokens'] > 0 else float('nan'),
            axis=1
        )
        # Fractions of total
        df['prompt_fraction'] = df['prompt_tokens'] / df['total_tokens']
        df['completion_fraction'] = df['completion_tokens'] / df['total_tokens']

    return df

# ───────────────────────────────────────────────────────────────
# 3) PERCENTILE & DUPLICATE HELPERS
# ───────────────────────────────────────────────────────────────
def percentile_stats(df, col):
    """
    Compute the 5th, 25th, 50th, 75th, and 95th percentiles for column `col`.
    Returns a Series indexed by ['5%', '25%', '50%', '75%', '95%'].
    """
    ps = [0.05, 0.25, 0.50, 0.75, 0.95]
    q = df[col].quantile(ps)
    q.index = [f"{int(p*100)}%" for p in ps]
    return q.round(5) #5 decimal places

def find_duplicates(df, threshold=0.85):
    """
    Identify near-duplicate prompt rows. For each row i, compare its 'prompt'
    against previously seen prompts. If SequenceMatcher.ratio() > threshold,
    record it as a duplicate pair (i, previous_index, similarity_score).

    Returns a list of tuples: (i, previous_index, similarity_score).
    If 'prompt' column doesn't exist, returns an empty list.
    """
    duplicates = []
    if 'prompt' not in df.columns:
        return duplicates

    seen = []  # List of (index, prompt_string)
    for idx, row in df.iterrows():
        prompt_text = str(row['prompt'])
        for prev_idx, prev_prompt in seen:
            ratio = SequenceMatcher(None, prompt_text, prev_prompt).ratio()
            if ratio > threshold:
                duplicates.append((idx, prev_idx, ratio))
                break
        seen.append((idx, prompt_text))

    return duplicates

# ───────────────────────────────────────────────────────────────
# 4) SUMMARY STATISTICS & CORRELATIONS
# ───────────────────────────────────────────────────────────────
def summary_stats(df, metrics):
    """
    For each column in `metrics` (a list of column-names), compute:
      - min
      - mean
      - median
      - max
    Returns a DataFrame whose index = ['min','mean','median','max'] and whose
    columns = the metric names. All values rounded to 2 decimal places.
    """
    stats = df[metrics].agg(['min', 'mean', 'median', 'max'])
    return stats.round(5) #lets try 5 decimal places

# ───────────────────────────────────────────────────────────────
# 5) GENERIC PLOT HELPERS
# ───────────────────────────────────────────────────────────────
def plot_scatter(x, y, xlab, ylab, title, embed, out_dir, filename):
    """
    Generic scatter plot:
      - x: Series or array for X-axis
      - y: Series or array for Y-axis
      - xlab, ylab, title: axis labels and title
      - embed: if True, return a Base64 data URI string; otherwise save to out_dir/filename
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    if embed:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data_uri = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{data_uri}"
    else:
        path = os.path.join(out_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return path

def plot_hist(series, xlabel, ylabel, title, embed, out_dir, filename):
    """
    Generic histogram:
      - series: Pandas Series or array to histogram
      - xlabel, ylabel, title: axis labels and title
      - embed: if True, return Base64 data URI; otherwise save to out_dir/filename
    """
    fig, ax = plt.subplots()
    ax.hist(series.dropna(), bins=30, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if embed:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data_uri = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{data_uri}"
    else:
        path = os.path.join(out_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return path

# ───────────────────────────────────────────────────────────────
# 6) BUILDING THE HTML REPORT CONTENT
# ───────────────────────────────────────────────────────────────
def build_html_report(
        title,
        stats_df,
        corr_df,
        pct_cost,
        pct_tokens,
        imgs,
        dup_list,
        compare_df_html
    ):
    """
    Assemble a full HTML document (as a single string) containing:
      1) Title + generation timestamp
      2) Summary statistics table
      3) Percentile tables (cost, total_tokens)
      4) Correlation matrix table
      5) Before/After optimization comparison (if provided)
      6) Any requested charts (embedded Base64 images)
      7) List of near-duplicate prompts (if any)

    Parameters:
      - title            : string for <title> and <h1>
      - stats_df         : DataFrame from summary_stats()
      - corr_df          : DataFrame of df.corr()
      - pct_cost         : Series of cost percentiles
      - pct_tokens       : Series of total_tokens percentiles
      - imgs             : dict mapping caption → Base64 URI (or None)
      - dup_list         : list of (i, j, similarity) tuples for duplicates
      - compare_df_html  : HTML string for before/after comparison table (or None)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<meta charset='utf-8'><title>{title}</title>",
        "<style>",
        "  body { font-family: Arial, sans-serif; margin: 20px; }",
        "  table.stats, table.corr { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "  table.stats th, table.stats td, table.corr th, table.corr td { border: 1px solid #ddd; padding: 8px; }",
        "  table.stats tr:nth-child(even), table.corr tr:nth-child(even) { background-color: #f9f9f9; }",
        "  table.stats th, table.corr th { background-color: #f2f2f2; }",
        "</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p><em>Generated on {now}</em></p>",

        "<h2>Summary Statistics</h2>",
        stats_df.to_html(classes='stats', border=0, index=True),

        "<h2>Percentile Breakdowns</h2>",
        "<h3>Cost Percentiles</h3>",
        pct_cost.to_frame(name='cost').to_html(border=0),
        "<h3>Total Tokens Percentiles</h3>",
        pct_tokens.to_frame(name='total_tokens').to_html(border=0),

        "<h2>Correlation Matrix</h2>",
        corr_df.to_html(classes='corr', border=0),
    ]

    # 5) Compare before vs. after optimization (if provided)
    if compare_df_html:
        html_parts += [
            "<h2>Before vs. After Optimization</h2>",
            compare_df_html
        ]

    # 6) Embed each requested chart
    for caption, uri in imgs.items():
        if uri:
            html_parts += [
                f"<h3>{caption}</h3>",
                f"<img src='{uri}' alt='{caption}' style='max-width:100%; height:auto;'>"
            ]

    # 7) List near-duplicate prompts (if any)
    if dup_list:
        html_parts += ["<h2>Near‐Duplicate Prompts Detected</h2>", "<ul>"]
        for i, j, score in dup_list:
            html_parts += [
                f"<li>Row {i} ≈ Row {j} (similarity={score:.2f})</li>"
            ]
        html_parts += ["</ul>"]

    html_parts += ["</body></html>"]
    return "\n".join(html_parts)

# ───────────────────────────────────────────────────────────────
# 7) BUILDING THE MARKDOWN REPORT CONTENT
# ───────────────────────────────────────────────────────────────
def build_markdown_report(title, stats_df, imgs, out_dir, md_path):
    """
    Assemble a Markdown report with:
      1) Title + generation timestamp
      2) Summary statistics table (as Markdown)
      3) Each requested chart inserted as image links (PNGs saved in out_dir)
      4) (Note: For simplicity, this example only shows summary + charts—no percentiles or duplicates.)

    You could extend this to include percentile tables or duplicates as well.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {title}",
        f"*Generated on {now}*",
        "",
        "## Summary Statistics",
        ""
    ]

    # Convert summary stats to Markdown table
    md_table = stats_df.reset_index().to_markdown(index=False).split("\n")
    lines += md_table + [""]

    # Insert chart images
    for caption, path in imgs.items():
        if path:
            rel = os.path.basename(path)
            lines += [
                f"### {caption}",
                f"![{caption}]({rel})",
                ""
            ]

    # Write out the Markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# ───────────────────────────────────────────────────────────────
# 8) MAIN ENTRYPOINT: PARSE ARGS, RUN EVERYTHING
# ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build an advanced HTML/Markdown report from a GPT batch CSV"
    )
    # Required inputs
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input CSV (must have columns: cost, total_tokens, prompt_tokens, completion_tokens, response, prompt, etc.)')
    parser.add_argument('-f', '--format', choices=['html','md'], required=True,
                        help='Output format: "html" (inline Base64 images) or "md" (separate PNGs)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file path (e.g. report.html or report.md)')

    # Optional flags
    parser.add_argument('--charts', nargs='+', choices=[
                            'scatter_cost_tokens',
                            'hist_response_length',
                            'hist_cost_per_token',
                            'scatter_prompt_cost',
                            'heatmap_corr'
                        ],
                        default=['scatter_cost_tokens','hist_response_length'],
                        help='Which charts to include in the report')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Top N prompts by cost for bar chart (if implemented)')
    parser.add_argument('--title', default='Batch Run Report (v2)',
                        help='Title for the report')
    parser.add_argument('--min-cost', type=float, default=None,
                        help='Filter: only include rows with cost ≥ this value')
    parser.add_argument('--keyword', type=str, default=None,
                        help='Filter: only include rows whose prompt or response contains this regex keyword')
    parser.add_argument('--opt-flag-col', type=str, default=None,
                        help='Name of the column marking optimized vs non-optimized (for comparison)')

    args = parser.parse_args()

    # ───────────────────────────────────────────────────────────
    # 1) LOAD & CLEAN DATA
    # ───────────────────────────────────────────────────────────
    df = load_data(args.input)

    # ───────────────────────────────────────────────────────────
    # 2) APPLY OPTIONAL FILTERS
    # ───────────────────────────────────────────────────────────
    if args.min_cost is not None:
        df = df[df['cost'] >= args.min_cost]

    if args.keyword:
        # Keep rows where prompt OR response contains the keyword (case-insensitive)
        pat = args.keyword
        mask = (
            df['prompt'].astype(str).str.contains(pat, case=False, na=False) |
            df['response'].astype(str).str.contains(pat, case=False, na=False)
        )
        df = df[mask]

    # ───────────────────────────────────────────────────────────
    # 3) COMPUTE DERIVED COLUMNS
    # ───────────────────────────────────────────────────────────
    df = compute_response_length(df)
    df = compute_efficiency_metrics(df)

    # ───────────────────────────────────────────────────────────
    # 4) BEFORE vs. AFTER OPTIMIZATION (if requested)
    # ───────────────────────────────────────────────────────────
    compare_df_html = None
    if args.opt_flag_col and args.opt_flag_col in df.columns:
        # Group DataFrame by the values in that column (e.g. "optimized" = yes/no)
        grouped = {
            grp_val: df[df[args.opt_flag_col] == grp_val]
            for grp_val in df[args.opt_flag_col].unique()
        }
        summary_groups = []
        for grp_val, subdf in grouped.items():
            s = summary_stats(subdf, [
                'prompt_tokens','completion_tokens',
                'total_tokens','cost','cost_per_token','response_length'
            ])
            # Prefix columns so we can tell them apart
            s.columns = pd.MultiIndex.from_product([[grp_val], s.columns])
            summary_groups.append(s)

        if summary_groups:
            compare_df = pd.concat(summary_groups, axis=1)
            compare_df_html = compare_df.to_html(border=0)

    # ───────────────────────────────────────────────────────────
    # 5) SUMMARY STATISTICS FOR ENTIRE DATASET
    # ───────────────────────────────────────────────────────────
    # Determine which numeric metrics actually exist in df
    metrics = [
        c for c in [
            'prompt_tokens','completion_tokens','total_tokens',
            'cost','cost_per_token','response_length'
        ] if c in df.columns
    ]
    if not metrics:
        raise SystemExit("❌ No numeric metrics found. CSV must include at least one of: 'prompt_tokens','completion_tokens','total_tokens','cost','cost_per_token','response_length'.")

    stats_df = summary_stats(df, metrics)

    # ───────────────────────────────────────────────────────────
    # 6) PERCENTILE CALCULATIONS
    # ───────────────────────────────────────────────────────────
    pct_cost = percentile_stats(df, 'cost') if 'cost' in df.columns else pd.Series()
    pct_tokens = percentile_stats(df, 'total_tokens') if 'total_tokens' in df.columns else pd.Series()

    # ───────────────────────────────────────────────────────────
    # 7) CORRELATION MATRIX
    # ───────────────────────────────────────────────────────────
    numeric_cols = [
        c for c in [
            'prompt_tokens','completion_tokens','total_tokens',
            'cost','cost_per_token','response_length'
        ] if c in df.columns
    ]
    corr_df = df[numeric_cols].corr().round(5) if numeric_cols else pd.DataFrame() #5

    # ───────────────────────────────────────────────────────────
    # 8) NEAR‐DUPLICATE PROMPT DETECTION
    # ───────────────────────────────────────────────────────────
    duplicates = find_duplicates(df, threshold=0.85)

    # ───────────────────────────────────────────────────────────
    # 9) GENERATE REQUESTED CHARTS
    # ───────────────────────────────────────────────────────────
    imgs = {}
    # If Markdown output, create directory for PNG files
    if args.format == 'md':
        base = os.path.splitext(os.path.basename(args.output))[0]
        imgs_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), f"{base}_files")
        os.makedirs(imgs_dir, exist_ok=True)
    else:
        imgs_dir = None  # Not used for inline-HTML mode

    def _maybe_plot(key, plot_func, *func_args):
        """
        Helper: if key is in args.charts, run plot_func(*func_args).
        Otherwise, return None.
        """
        if key in args.charts:
            return plot_func(*func_args)
        return None

    # a) Scatter: Total Tokens vs. Cost
    imgs['Cost vs. Total Tokens'] = _maybe_plot(
        'scatter_cost_tokens',
        plot_scatter,
        df['total_tokens'], df['cost'],
        'Total Tokens', 'Cost (USD)', 'Cost vs. Total Tokens',
        args.format == 'html',
        imgs_dir, 'scatter_cost_tokens.png'
    )

    # b) Histogram: Response Length
    imgs['Response Length Distribution'] = _maybe_plot(
        'hist_response_length',
        plot_hist,
        df['response_length'],               # the series
        'Response Length (chars)',           # xlabel
        'Count',                             # ylabel
        'Response Length Distribution',      # title
        args.format == 'html',
        imgs_dir, 'hist_response_length.png'
    )

    # c) Histogram: Cost per Token
    if 'cost_per_token' in df.columns:
        imgs['Cost per Token Distribution'] = _maybe_plot(
            'hist_cost_per_token',
            plot_hist,
            df['cost_per_token'],
            'Cost per Token (USD)',
            'Count',
            'Cost per Token Distribution',
            args.format == 'html',
            imgs_dir, 'hist_cost_per_token.png'
        )

    # d) Scatter: Prompt Tokens vs. Cost
    if 'prompt_tokens' in df.columns and 'cost' in df.columns:
        imgs['Prompt Tokens vs. Cost'] = _maybe_plot(
            'scatter_prompt_cost',
            plot_scatter,
            df['prompt_tokens'], df['cost'],
            'Prompt Tokens', 'Cost (USD)', 'Prompt Tokens vs. Cost',
            args.format == 'html',
            imgs_dir, 'scatter_prompt_cost.png'
        )

    # e) Correlation heatmap (if requested)
    if 'heatmap_corr' in args.charts and not corr_df.empty:
        def plot_corr_heatmap(corrd, embed, out_dir):
            """
            Plot a heatmap of the correlation matrix.
            If embed, return Base64 URI; otherwise save to out_dir.
            """
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.matshow(corrd, vmin=-1, vmax=1, cmap='RdBu')
            plt.colorbar(cax)
            ax.set_xticks(range(len(corrd.columns)))
            ax.set_xticklabels(corrd.columns, rotation=90)
            ax.set_yticks(range(len(corrd.index)))
            ax.set_yticklabels(corrd.index)
            plt.title('Correlation Matrix', pad=20)

            if embed:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                data_uri = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return f"data:image/png;base64,{data_uri}"
            else:
                path = os.path.join(out_dir, 'heatmap_corr.png')
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)
                return path

        imgs['Correlation Heatmap'] = plot_corr_heatmap(
            corr_df, args.format == 'html', imgs_dir
        )

    # ───────────────────────────────────────────────────────────
    # 10) BUILD FINAL REPORT
    # ───────────────────────────────────────────────────────────
    if args.format == 'md':
        # Build a Markdown report (summary + charts)
        build_markdown_report(
            args.title,
            stats_df,
            imgs,
            imgs_dir,
            args.output
        )
        print(f"✅ Markdown report written to: {args.output}")
    else:
        # Build an HTML report (all tables + embedded Base64 charts)
        html_content = build_html_report(
            args.title,
            stats_df,
            corr_df,
            pct_cost,
            pct_tokens,
            imgs,
            duplicates,
            compare_df_html
        )
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML report written to: {args.output}")

# ───────────────────────────────────────────────────────────────
# Run the main() function when script is invoked
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
