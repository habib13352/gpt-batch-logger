import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64
from datetime import datetime
from difflib import SequenceMatcher

def load_data(path):
    df = pd.read_csv(path)
    if 'cost' not in df.columns:
        raise SystemExit("Your CSV needs a 'cost' column.")
    # Strip $, comma → float
    df['cost'] = df['cost'].astype(str).str.replace(r'[\$,]', '', regex=True)
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df = df.dropna(subset=['cost'])
    return df

def compute_response_length(df):
    if 'response' in df.columns:
        df['response_length'] = df['response'].astype(str).str.len()
    return df

def compute_efficiency_metrics(df):
    # Cost per token & per 1K tokens
    df['cost_per_token'] = df['cost'] / df['total_tokens']
    df['cost_per_1k_tokens'] = df['cost_per_token'] * 1000
    
    # Prompt/completion token ratio
    df['prompt_completion_ratio'] = df.apply(
        lambda r: (r['prompt_tokens']/r['completion_tokens'])
        if r['completion_tokens']>0 else float('nan'),
        axis=1
    )
    df['prompt_fraction']    = df['prompt_tokens'] / df['total_tokens']
    df['completion_fraction'] = df['completion_tokens'] / df['total_tokens']
    return df

def percentile_stats(df, col):
    ps = [0.05, 0.25, 0.5, 0.75, 0.95]
    q = df[col].quantile(ps)
    q.index = [f"{int(p*100)}%" for p in ps]
    return q.round(4)

def find_duplicates(df, threshold=0.85):
    seen = []
    duplicates = []
    if 'prompt' not in df.columns:
        return []
    for idx, r in df.iterrows():
        prom = str(r['prompt'])
        for prev_idx, prev_prom in seen:
            ratio = SequenceMatcher(None, prom, prev_prom).ratio()
            if ratio > threshold:
                duplicates.append((idx, prev_idx, ratio))
                break
        seen.append((idx, prom))
    return duplicates

def summary_stats(df, metrics):
    stats = df[metrics].agg(['min', 'mean', 'median', 'max'])
    return stats.round(2)

def plot_scatter(x, y, xlab, ylab, title, embed, out_dir, filename):
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    if embed:
        buf = io.BytesIO()
        plt.savefig(buf, format='png'); buf.seek(0)
        uri = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{uri}"
    else:
        path = os.path.join(out_dir, filename)
        fig.savefig(path); plt.close(fig)
        return path

def plot_hist(series, xlabel, ylabel, title, embed, out_dir, filename):
    fig, ax = plt.subplots()
    ax.hist(series, bins=30, edgecolor='black')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if embed:
        buf = io.BytesIO()
        plt.savefig(buf, format='png'); buf.seek(0)
        uri = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{uri}"
    else:
        path = os.path.join(out_dir, filename)
        fig.savefig(path); plt.close(fig)
        return path

def build_html_report(title, stats_df, corr_df, pct_cost, pct_tokens, imgs, dup_list, compare_df_html):
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

        "<h2>Percentiles</h2>",
        "<h3>Cost Percentiles</h3>",
        pct_cost.to_frame(name='cost').to_html(border=0),
        "<h3>Total Tokens Percentiles</h3>",
        pct_tokens.to_frame(name='total_tokens').to_html(border=0),

        "<h2>Correlation Matrix</h2>",
        corr_df.to_html(classes='corr', border=0),
    ]

    # If comparing optimized vs non‐optimized was computed, embed it:
    if compare_df_html:
        html += ["<h2>Before vs. After Optimization</h2>", compare_df_html]

    # Insert all plots (Base64 URIs)
    for caption, uri in imgs.items():
        if uri:
            html += [f"<h3>{caption}</h3>", f"<img src='{uri}' alt='{caption}'>"]

    # If duplicates were found:
    if dup_list:
        html += ["<h2>Near‐Duplicate Prompts</h2>", "<ul>"]
        for i, j, ratio in dup_list:
            html += [f"<li>Row {i} ≈ Row {j} (similarity={ratio:.2f})</li>"]
        html += ["</ul>"]

    html += ["</body></html>"]
    return "\n".join(html)

def main():
    parser = argparse.ArgumentParser(
        description="Build a more powerful, “v2” HTML/Markdown report from your batch CSV"
    )
    parser.add_argument('-i', '--input', required=True, help='Path to input CSV')
    parser.add_argument('-f', '--format', choices=['html','md'], required=True,
                        help='Output format: html or md')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file (e.g. report.html or report.md)')
    parser.add_argument('--charts', nargs='+', choices=[
                            'scatter_cost_tokens',
                            'hist_response_length',
                            'hist_cost_per_token',
                            'scatter_prompt_cost',
                            'heatmap_corr'
                        ],
                        default=['scatter_cost_tokens','hist_response_length'],
                        help='Which charts to include')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Top N prompts by cost for bar chart')
    parser.add_argument('--title', default='Batch Run Report (v2)',
                        help='Report title')
    parser.add_argument('--min-cost', type=float, default=None,
                        help='Filter: only include rows with cost ≥ this value')
    parser.add_argument('--keyword', type=str, default=None,
                        help='Filter: only include rows whose prompt or response contains this regex')
    parser.add_argument('--opt-flag-col', type=str, default=None,
                        help='If your CSV has a column marking optimization (e.g. "optimized"), group by it')
    args = parser.parse_args()

    # ───────────────────────────────────────────────────────────
    # 1) Load & clean
    # ───────────────────────────────────────────────────────────
    df = load_data(args.input)

    # 2) Optional filters
    if args.min_cost is not None:
        df = df[df['cost'] >= args.min_cost]
    if args.keyword:
        pat = args.keyword
        df = df[df['prompt'].astype(str).str.contains(pat, case=False, na=False) |
                df['response'].astype(str).str.contains(pat, case=False, na=False)]

    # 3) Compute derived columns
    df = compute_response_length(df)
    df = compute_efficiency_metrics(df)

    # 4) Basic “before vs after optimization” grouping
    compare_df_html = None
    if args.opt_flag_col and args.opt_flag_col in df.columns:
        # E.g. df['optimized'] has values like "yes"/"no" or True/False
        grouped = {grp: df[df[args.opt_flag_col] == grp] 
                   for grp in df[args.opt_flag_col].unique()}
        summary_by_group = []
        for grp, sub in grouped.items():
            s = summary_stats(sub, [
                'prompt_tokens','completion_tokens',
                'total_tokens','cost','cost_per_token','response_length'
            ])
            s.columns = pd.MultiIndex.from_product([[grp], s.columns])
            summary_by_group.append(s)
        if summary_by_group:
            compare_df = pd.concat(summary_by_group, axis=1)
            compare_df_html = compare_df.to_html(border=0)

    # 5) Build “metrics” list for summary stats
    metrics = [
        c for c in [
            'prompt_tokens','completion_tokens','total_tokens',
            'cost','cost_per_token','response_length'
        ] if c in df.columns
    ]
    stats_df = summary_stats(df, metrics)

    # 6) Percentiles
    pct_cost = percentile_stats(df, 'cost') if 'cost' in df.columns else pd.Series()
    pct_tokens = percentile_stats(df, 'total_tokens') if 'total_tokens' in df.columns else pd.Series()

    # 7) Correlation matrix
    numeric_cols = [c for c in [
        'prompt_tokens','completion_tokens','total_tokens','cost',
        'cost_per_token','response_length'
    ] if c in df.columns]
    corr_df = df[numeric_cols].corr().round(2) if numeric_cols else pd.DataFrame()

    # 8) Duplicate detection
    duplicates = find_duplicates(df, threshold=0.85)

    # 9) Generate requested charts
    imgs = {}
    # Ensure an output folder for non‐embedded charts (Markdown mode)
    if args.format == 'md':
        imgs_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            os.path.splitext(os.path.basename(args.output))[0] + "_files"
        )
        os.makedirs(imgs_dir, exist_ok=True)
    else:
        imgs_dir = None

    def _make_plot(key, func, *func_args):
        if key in args.charts:
            return func(*func_args)
        return None

    # a) Scatter: Total Tokens vs Cost
    imgs['Cost vs. Total Tokens'] = _make_plot(
        'scatter_cost_tokens',
        plot_scatter, df, metrics, args.format=='html', imgs_dir, 'scatter_cost_tokens.png'
    )
    # b) Histogram: Response Length
    imgs['Response Length Distribution'] = _make_plot(
        'hist_response_length',
        plot_hist, df, args.format=='html', imgs_dir, 'hist_response_length.png'
    )
    # c) Histogram: Cost per Token
    imgs['Cost per Token Distribution'] = _make_plot(
        'hist_cost_per_token',
        plot_hist, df['cost_per_token'], 'Cost per Token (USD)', 'Count',
        'Cost per Token Distribution', args.format=='html', imgs_dir, 'hist_cost_per_token.png'
    )
    # d) Scatter: Prompt Tokens vs Cost
    imgs['Prompt Tokens vs. Cost'] = _make_plot(
        'scatter_prompt_cost',
        plot_scatter, df['prompt_tokens'], df['cost'],
        'Prompt Tokens', 'Cost (USD)', 'Prompt Tokens vs. Cost',
        args.format=='html', imgs_dir, 'scatter_prompt_cost.png'
    )
    # e) Correlation Heatmap (if requested)
    if 'heatmap_corr' in args.charts and not corr_df.empty:
        def plot_corr_heatmap(corrd, embed, out_dir):
            fig, ax = plt.subplots(figsize=(6,6))
            cax = ax.matshow(corrd, vmin=-1, vmax=1, cmap='RdBu')
            plt.colorbar(cax)
            ax.set_xticks(range(len(corrd.columns)))
            ax.set_xticklabels(corrd.columns, rotation=90)
            ax.set_yticks(range(len(corrd.index)))
            ax.set_yticklabels(corrd.index)
            plt.title('Correlation Matrix', pad=20)
            if embed:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
                data = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return f"data:image/png;base64,{data}"
            else:
                path = os.path.join(out_dir, 'heatmap_corr.png')
                fig.savefig(path, bbox_inches='tight'); plt.close(fig)
                return path

        imgs['Correlation Matrix Heatmap'] = plot_corr_heatmap(
            corr_df, args.format=='html', imgs_dir
        )

    # 10) Build final report
    if args.format == 'md':
        # Write markdown + PNGs
        build_markdown_report(args.title, stats_df, imgs, imgs_dir, args.output)
        print(f"✅ Markdown report written to: {args.output}")
    else:
        # Inline HTML
        html = build_html_report(
            args.title, stats_df, corr_df, pct_cost, pct_tokens,
            imgs, duplicates, compare_df_html
        )
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ HTML report written to: {args.output}")

if __name__ == "__main__":
    main()
