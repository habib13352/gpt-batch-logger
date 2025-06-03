#edit this to work with generated csv WITHOUT editing csv or py

"""
report_builder.py (simplified)

Assumes your CSV already has these exact columns:
  - prompt
  - response
  - prompt_tokens
  - completion_tokens
  - total_tokens
  - cost

The cost column may contain “$” or “,” (for example, "$0.00040"). We strip those
out and convert to float. All other columns must be named exactly as above.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64
from datetime import datetime

def load_data(path):
    """
    1) Read CSV from 'path'.
    2) Strip '$' and ',' from the 'cost' column so it can convert to float.
    3) Convert 'cost' to numeric and drop any rows where cost is invalid (NaN).
    4) Return the cleaned DataFrame.
    """
    try:
        df = pd.read_csv(path)

        # Ensure the 'cost' column exists
        if 'cost' not in df.columns:
            raise SystemExit("❌ Your CSV must have a column named exactly 'cost'.")

        # 2) Strip out $ and , from cost (so "$0.00040" → "0.00040")
        df['cost'] = (
            df['cost']
            .astype(str)
            .str.replace(r'[\$,]', '', regex=True)
        )

        # 3) Convert to numeric. Unparsable → NaN, then drop those rows.
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df = df.dropna(subset=['cost'])

        return df

    except Exception as e:
        raise SystemExit(f"❌ Error loading '{path}': {e}")

def compute_response_length(df):
    """
    If 'response' column exists, add 'response_length' (character count).
    """
    if 'response' in df.columns:
        df['response_length'] = df['response'].astype(str).str.len()
    return df

def summary_stats(df, metrics):
    """
    Compute min, mean, median, max for each numeric column in metrics.
    """
    stats = df[metrics].agg(['min', 'mean', 'median', 'max'])
    return stats.round(2)

def plot_scatter(df, metrics, embed, out_dir):
    """
    Scatter plot: x = total_tokens, y = cost.
    If embed=True, return a Base64 data URI; otherwise save PNG to out_dir.
    """
    if not {'cost', 'total_tokens'}.issubset(metrics):
        return None

    fig, ax = plt.subplots()
    ax.scatter(df['total_tokens'], df['cost'], alpha=0.7)
    ax.set_xlabel('Total Tokens')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Cost vs. Total Tokens')

    if embed:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{data}"
    else:
        path = os.path.join(out_dir, 'scatter_cost_tokens.png')
        fig.savefig(path)
        plt.close(fig)
        return path

def plot_hist(df, embed, out_dir):
    """
    Histogram of response_length. If embed=True, return Base64 data URI;
    otherwise save PNG to out_dir.
    """
    if 'response_length' not in df.columns:
        return None

    fig, ax = plt.subplots()
    ax.hist(df['response_length'], bins=30, edgecolor='black')
    ax.set_xlabel('Response Length (chars)')
    ax.set_ylabel('Count')
    ax.set_title('Response Length Distribution')

    if embed:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{data}"
    else:
        path = os.path.join(out_dir, 'hist_response_length.png')
        fig.savefig(path)
        plt.close(fig)
        return path

def plot_bar(df, top_n, embed, out_dir):
    """
    Bar chart of top N rows by cost. If embed=True, return Base64 data URI;
    otherwise save PNG to out_dir.
    """
    if 'cost' not in df.columns:
        return None

    top = df.nlargest(top_n, 'cost')
    fig, ax = plt.subplots()
    ax.bar(range(len(top)), top['cost'])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"#{i+1}" for i in range(len(top))], rotation=45)
    ax.set_ylabel('Cost (USD)')
    ax.set_title(f'Top {top_n} Prompts by Cost')

    if embed:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{data}"
    else:
        path = os.path.join(out_dir, 'bar_top_costs.png')
        fig.savefig(path)
        plt.close(fig)
        return path

def build_html_report(title, stats_df, imgs):
    """
    Build a fully self‐contained HTML page (with inline Base64 charts).
    """
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<meta charset='utf-8'><title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p><em>Generated on {date}</em></p>",
        "<h2>Summary Statistics</h2>",
        stats_df.to_html(classes='stats', border=0),
    ]
    for caption, uri in imgs.items():
        if uri:
            html += [
                f"<h3>{caption}</h3>",
                f"<img src='{uri}' alt='{caption}'>"
            ]
    html += ["</body></html>"]
    return "\n".join(html)

def build_markdown_report(title, stats_df, imgs, out_dir, md_path):
    """
    Build a Markdown report with image links (PNGs saved into out_dir).
    """
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {title}",
        f"*Generated on {date}*",
        "",
        "## Summary Statistics",
        ""
    ]
    lines += stats_df.reset_index().to_markdown(index=False).split("\n")
    lines += [""]
    for caption, path in imgs.items():
        if path:
            rel = os.path.basename(path)
            lines += [
                f"### {caption}",
                f"![{caption}]({rel})",
                ""
            ]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def main():
    """
    1) Parse CLI args:
         -i / --input    : path to input CSV
         -f / --format   : 'html' or 'md'
         -o / --output   : path to output (report.html or report.md)
         --charts        : which charts to include (scatter, hist, bar)
         --top-n         : how many top‐cost bars
         --title         : report title
    2) load_data(...)
    3) compute_response_length(...)
    4) summary_stats(...)
    5) plot chosen charts, either embedding or saving PNGs
    6) build the HTML or Markdown file
    """
    parser = argparse.ArgumentParser(
        description="Build HTML/Markdown report from batch CSV"
    )
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-f', '--format', choices=['html','md'], required=True,
                        help='Output format: html or md')
    parser.add_argument('-o', '--output', required=True,
                        help='Output path (e.g. report.html or report.md)')
    parser.add_argument('--charts', nargs='+', choices=['scatter','hist','bar'],
                        default=['scatter','hist','bar'], help='Which charts to include')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Top N for bar chart')
    parser.add_argument('--title', default='Batch Run Report',
                        help='Report title')
    args = parser.parse_args()

    # 1) Load and clean the data
    df = load_data(args.input)
    df = compute_response_length(df)

    # 2) Pick which metrics exist
    metrics = [
        c for c in ['prompt_tokens','completion_tokens','total_tokens','cost']
        if c in df.columns
    ]
    if 'response_length' in df.columns:
        metrics.append('response_length')

    if not metrics:
        raise SystemExit("❌ No numeric columns found (expecting prompt_tokens, completion_tokens, total_tokens, cost).")

    # 3) Compute summary statistics
    stats_df = summary_stats(df, metrics)

    # 4) Generate charts
    if args.format == 'md':
        # Markdown output → save PNGs next to the .md
        imgs_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            os.path.splitext(os.path.basename(args.output))[0] + "_files"
        )
        os.makedirs(imgs_dir, exist_ok=True)

        imgs = {
            'Cost vs. Total Tokens': plot_scatter(df, metrics, embed=False, out_dir=imgs_dir),
            'Response Length Distribution': plot_hist(df, embed=False, out_dir=imgs_dir),
            f'Top {args.top_n} Costs': plot_bar(df, args.top_n, embed=False, out_dir=imgs_dir)
        }
        build_markdown_report(args.title, stats_df, imgs, imgs_dir, args.output)
        print(f"✅ Markdown report written to: {args.output}")

    else:
        # HTML output → embed charts as Base64
        imgs = {
            'Cost vs. Total Tokens': plot_scatter(df, metrics, embed=True, out_dir=None),
            'Response Length Distribution': plot_hist(df, embed=True, out_dir=None),
            f'Top {args.top_n} Costs': plot_bar(df, args.top_n, embed=True, out_dir=None)
        }
        html_content = build_html_report(args.title, stats_df, imgs)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML report written to: {args.output}")

if __name__ == "__main__":
    main()
