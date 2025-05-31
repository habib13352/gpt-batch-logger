#!/usr/bin/env python3
"""
csv_analyzer.py

Reads 'results/summary_log_v3.csv' produced by your batch_from_file_v3.py script
and prints a summary report of:
  - total number of prompts processed
  - total & average prompt/completion/overall tokens
  - total & average estimated cost
  - cost-per-token, cost-per-1000-tokens, cost-per-100-prompts
  - number of truncated responses (and their Prompt #s)

Additionally, saves the exact same report text to a .txt file at:
    results/summary_report.txt

Usage:
    python csv_analyzer.py
"""

import csv
import sys
import os

# Input CSV location (relative to where you run this script)
CSV_FILENAME = "results/summary_log_v3.csv"

# Output TXT report location (relative to where you run this script)
OUTPUT_FILENAME = "results/summary_report.txt"


def parse_cost(cost_str):
    """
    Given a string like '$0.0025', strip the '$' and convert to float.
    If parsing fails, returns 0.0.
    """
    try:
        return float(cost_str.strip().lstrip("$"))
    except Exception:
        return 0.0


def main():
    # Ensure the 'results/' folder exists if we're about to write there
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    try:
        with open(CSV_FILENAME, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Make sure expected columns exist
            expected_cols = {
                "Prompt #",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
                "Truncated?"
            }
            missing = expected_cols - set(reader.fieldnames)
            if missing:
                print(f"ERROR: Missing columns in CSV: {missing}")
                sys.exit(1)

            total_prompts = 0
            sum_prompt_tokens = 0
            sum_completion_tokens = 0
            sum_total_tokens = 0
            sum_cost = 0.0
            truncated_count = 0
            truncated_list = []

            min_prompt_tokens = None
            max_prompt_tokens = None
            min_completion_tokens = None
            max_completion_tokens = None
            min_cost = None
            max_cost = None

            for row in reader:
                total_prompts += 1

                # Parse token counts
                try:
                    pt = int(row["prompt_tokens"])
                except ValueError:
                    pt = 0
                try:
                    ct = int(row["completion_tokens"])
                except ValueError:
                    ct = 0
                try:
                    tt = int(row["total_tokens"])
                except ValueError:
                    tt = pt + ct

                # Parse cost
                cost = parse_cost(row["cost"])

                # Parse truncated flag
                is_truncated = row["Truncated?"].strip().lower() in ("yes", "true", "1")

                # Accumulate sums
                sum_prompt_tokens += pt
                sum_completion_tokens += ct
                sum_total_tokens += tt
                sum_cost += cost

                # Track min/max prompt tokens
                if (min_prompt_tokens is None) or (pt < min_prompt_tokens):
                    min_prompt_tokens = pt
                if (max_prompt_tokens is None) or (pt > max_prompt_tokens):
                    max_prompt_tokens = pt

                # Track min/max completion tokens
                if (min_completion_tokens is None) or (ct < min_completion_tokens):
                    min_completion_tokens = ct
                if (max_completion_tokens is None) or (ct > max_completion_tokens):
                    max_completion_tokens = ct

                # Track min/max cost
                if (min_cost is None) or (cost < min_cost):
                    min_cost = cost
                if (max_cost is None) or (cost > max_cost):
                    max_cost = cost

                # Track truncated rows
                if is_truncated:
                    truncated_count += 1
                    try:
                        truncated_list.append(int(row["Prompt #"]))
                    except ValueError:
                        pass

            if total_prompts == 0:
                print("No data found in CSV.")
                return

            # Compute averages
            avg_prompt_tokens = sum_prompt_tokens / total_prompts
            avg_completion_tokens = sum_completion_tokens / total_prompts
            avg_total_tokens = sum_total_tokens / total_prompts
            avg_cost = sum_cost / total_prompts

            # New metrics:
            cost_per_token = sum_cost / sum_total_tokens if sum_total_tokens > 0 else 0.0
            cost_per_1000_tokens = cost_per_token * 1000
            cost_per_100_prompts = avg_cost * 100

            # Build the report lines in a list
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("CSV Analysis Report")
            report_lines.append(f"Source file: {CSV_FILENAME}")
            report_lines.append("=" * 60)
            report_lines.append(f"Total prompts processed: {total_prompts}")
            report_lines.append("")
            report_lines.append("→ Token usage (per-prompt stats):")
            report_lines.append(f"   • Total prompt tokens      : {sum_prompt_tokens}")
            report_lines.append(f"   • Total completion tokens  : {sum_completion_tokens}")
            report_lines.append(f"   • Total tokens             : {sum_total_tokens}")
            report_lines.append("")
            report_lines.append(f"   • Min prompt tokens        : {min_prompt_tokens}")
            report_lines.append(f"   • Max prompt tokens        : {max_prompt_tokens}")
            report_lines.append(f"   • Avg prompt tokens        : {avg_prompt_tokens:.2f}")
            report_lines.append("")
            report_lines.append(f"   • Min completion tokens    : {min_completion_tokens}")
            report_lines.append(f"   • Max completion tokens    : {max_completion_tokens}")
            report_lines.append(f"   • Avg completion tokens    : {avg_completion_tokens:.2f}")
            report_lines.append("")
            report_lines.append(f"   • Avg total tokens         : {avg_total_tokens:.2f}")
            report_lines.append("")
            report_lines.append("→ Cost (USD):")
            report_lines.append(f"   • Total estimated cost      : ${sum_cost:.4f}")
            report_lines.append(f"   • Min single-prompt cost    : ${min_cost:.4f}")
            report_lines.append(f"   • Max single-prompt cost    : ${max_cost:.4f}")
            report_lines.append(f"   • Avg cost per prompt       : ${avg_cost:.4f}")
            report_lines.append("")
            report_lines.append("→ Additional Cost Metrics:")
            report_lines.append(f"   • Cost per token            : ${cost_per_token:.6f}")
            report_lines.append(f"   • Cost per 1 000 tokens     : ${cost_per_1000_tokens:.4f}")
            report_lines.append(f"   • Cost per 100 prompts      : ${cost_per_100_prompts:.4f}")
            report_lines.append("")
            report_lines.append("→ Truncation:")
            report_lines.append(f"   • Number of truncated responses: {truncated_count}")
            if truncated_count > 0:
                truncated_list_str = ", ".join(map(str, truncated_list))
                report_lines.append(f"   • Truncated Prompt #s           : {truncated_list_str}")
            report_lines.append("=" * 60)

            # Join lines into one big string
            report_text = "\n".join(report_lines)

            # 1) Print to console
            print(report_text)

            # 2) Write to OUTPUT_FILENAME
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as out_f:
                out_f.write(report_text)

            print(f"\nReport also saved to: {OUTPUT_FILENAME}")

    except FileNotFoundError:
        print(f"ERROR: '{CSV_FILENAME}' not found in the current directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
