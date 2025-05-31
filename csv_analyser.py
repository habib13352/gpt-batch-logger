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

import csv       # For reading CSV files
import sys       # For exiting with an error code if something goes wrong
import os        # For file and directory path manipulations

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
# Path to the input CSV file (relative to where this script is run)
CSV_FILENAME = "results/summary_log_v3.csv"

# Path to the output TXT report (relative to where this script is run)
OUTPUT_FILENAME = "results/summary_report.txt"


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def parse_cost(cost_str):
    """
    Given a string like '$0.0025', strip the leading '$' and convert to float.
    If parsing fails (e.g., empty string or invalid format), return 0.0.
    """
    try:
        # Remove any leading/trailing whitespace, then strip out "$", then convert to float
        return float(cost_str.strip().lstrip("$"))
    except Exception:
        # In case of any error (ValueError, AttributeError, etc.), default to 0.0
        return 0.0


# ------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------------------
def main():
    # --------------------------------------------------------------------------
    # Step 1: Ensure the output directory exists
    # --------------------------------------------------------------------------
    # If the OUTPUT_FILENAME includes a directory, make sure it exists.
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if output_dir and not os.path.isdir(output_dir):
        # Create the directory (and any necessary parent directories)
        os.makedirs(output_dir)

    try:
        # ----------------------------------------------------------------------
        # Step 2: Open and read the CSV file
        # ----------------------------------------------------------------------
        with open(CSV_FILENAME, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # ------------------------------------------------------------------
            # Step 3: Validate that the expected columns exist in the CSV header
            # ------------------------------------------------------------------
            expected_cols = {
                "Prompt #",           # Column listing the prompt index/ID
                "prompt_tokens",      # Number of tokens used by the prompt
                "completion_tokens",  # Number of tokens used by the completion
                "total_tokens",       # Sum of prompt_tokens + completion_tokens
                "cost",               # Estimated cost, e.g., "$0.0025"
                "Truncated?"          # Flag indicating if the response was truncated
            }
            missing = expected_cols - set(reader.fieldnames)
            if missing:
                # If any expected column is missing, print an error and exit
                print(f"ERROR: Missing columns in CSV: {missing}")
                sys.exit(1)

            # ------------------------------------------------------------------
            # Step 4: Initialize accumulators and trackers
            # ------------------------------------------------------------------
            total_prompts = 0             # Count of rows (prompts) processed
            sum_prompt_tokens = 0         # Cumulative sum of prompt_tokens
            sum_completion_tokens = 0     # Cumulative sum of completion_tokens
            sum_total_tokens = 0          # Cumulative sum of total_tokens
            sum_cost = 0.0                # Cumulative sum of cost (in USD)
            truncated_count = 0           # Number of truncated responses
            truncated_list = []           # List of prompt IDs where truncation occurred

            # Track minimum/maximum values for tokens and cost
            min_prompt_tokens = None
            max_prompt_tokens = None
            min_completion_tokens = None
            max_completion_tokens = None
            min_cost = None
            max_cost = None

            # ------------------------------------------------------------------
            # Step 5: Iterate over each row in the CSV
            # ------------------------------------------------------------------
            for row in reader:
                total_prompts += 1

                # ---- Parse token counts (prompt_tokens, completion_tokens, total_tokens) ----
                try:
                    pt = int(row["prompt_tokens"])
                except ValueError:
                    pt = 0  # Default to 0 if parsing fails

                try:
                    ct = int(row["completion_tokens"])
                except ValueError:
                    ct = 0  # Default to 0 if parsing fails

                try:
                    tt = int(row["total_tokens"])
                except ValueError:
                    # If total_tokens is missing or invalid, fall back to pt + ct
                    tt = pt + ct

                # ---- Parse cost ----
                cost = parse_cost(row["cost"])

                # ---- Parse truncated flag ----
                # Some CSV exports might use "Yes"/"No" or "True"/"False"
                is_truncated = row["Truncated?"].strip().lower() in ("yes", "true", "1")

                # ---- Accumulate sums up to this row ----
                sum_prompt_tokens += pt
                sum_completion_tokens += ct
                sum_total_tokens += tt
                sum_cost += cost

                # ---- Track min/max prompt_tokens ----
                if (min_prompt_tokens is None) or (pt < min_prompt_tokens):
                    min_prompt_tokens = pt
                if (max_prompt_tokens is None) or (pt > max_prompt_tokens):
                    max_prompt_tokens = pt

                # ---- Track min/max completion_tokens ----
                if (min_completion_tokens is None) or (ct < min_completion_tokens):
                    min_completion_tokens = ct
                if (max_completion_tokens is None) or (ct > max_completion_tokens):
                    max_completion_tokens = ct

                # ---- Track min/max cost ----
                if (min_cost is None) or (cost < min_cost):
                    min_cost = cost
                if (max_cost is None) or (cost > max_cost):
                    max_cost = cost

                # ---- If truncated, note it down ----
                if is_truncated:
                    truncated_count += 1
                    try:
                        # Attempt to convert the "Prompt #" field to int
                        truncated_list.append(int(row["Prompt #"]))
                    except ValueError:
                        # If that fails (non-integer ID), ignore
                        pass

            # ----------------------------------------------------------------------
            # Step 6: Handle the case of an empty CSV (no prompts)
            # ----------------------------------------------------------------------
            if total_prompts == 0:
                print("No data found in CSV.")
                return

            # ----------------------------------------------------------------------
            # Step 7: Compute average values
            # ----------------------------------------------------------------------
            avg_prompt_tokens = sum_prompt_tokens / total_prompts
            avg_completion_tokens = sum_completion_tokens / total_prompts
            avg_total_tokens = sum_total_tokens / total_prompts
            avg_cost = sum_cost / total_prompts

            # ----------------------------------------------------------------------
            # Step 8: Compute additional cost metrics
            # ----------------------------------------------------------------------
            # Cost per single token (USD per token)
            if sum_total_tokens > 0:
                cost_per_token = sum_cost / sum_total_tokens
            else:
                cost_per_token = 0.0

            # Cost per 1,000 tokens
            cost_per_1000_tokens = cost_per_token * 1000

            # Cost per 100 prompts
            cost_per_100_prompts = avg_cost * 100

            # ----------------------------------------------------------------------
            # Step 9: Build the report text as a list of lines
            # ----------------------------------------------------------------------
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("CSV Analysis Report")
            report_lines.append(f"Source file: {CSV_FILENAME}")
            report_lines.append("=" * 60)
            report_lines.append(f"Total prompts processed: {total_prompts}")
            report_lines.append("")

            # Token usage section
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

            # Basic cost section
            report_lines.append("→ Cost (USD):")
            report_lines.append(f"   • Total estimated cost      : ${sum_cost:.4f}")
            report_lines.append(f"   • Min single-prompt cost    : ${min_cost:.4f}")
            report_lines.append(f"   • Max single-prompt cost    : ${max_cost:.4f}")
            report_lines.append(f"   • Avg cost per prompt       : ${avg_cost:.4f}")
            report_lines.append("")

            # Additional cost metrics section
            report_lines.append("→ Additional Cost Metrics:")
            report_lines.append(f"   • Cost per token            : ${cost_per_token:.6f}")
            report_lines.append(f"   • Cost per 1 000 tokens     : ${cost_per_1000_tokens:.4f}")
            report_lines.append(f"   • Cost per 100 prompts      : ${cost_per_100_prompts:.4f}")
            report_lines.append("")

            # Truncation summary
            report_lines.append("→ Truncation:")
            report_lines.append(f"   • Number of truncated responses: {truncated_count}")
            if truncated_count > 0:
                truncated_list_str = ", ".join(map(str, truncated_list))
                report_lines.append(f"   • Truncated Prompt #s           : {truncated_list_str}")
            report_lines.append("=" * 60)

            # Join all lines with newline characters into a single report string
            report_text = "\n".join(report_lines)

            # ------------------------------------------------------------------
            # Step 10: Print the report to the console
            # ------------------------------------------------------------------
            print(report_text)

            # ------------------------------------------------------------------
            # Step 11: Write the same report to the output TXT file
            # ------------------------------------------------------------------
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as out_f:
                out_f.write(report_text)

            # Confirmation message about file saving
            print(f"\nReport also saved to: {OUTPUT_FILENAME}")

    except FileNotFoundError:
        # If the CSV file is not found, notify the user and exit with an error code
        print(f"ERROR: '{CSV_FILENAME}' not found in the current directory.")
        sys.exit(1)


# Standard boilerplate: if this script is run directly, call main().
if __name__ == "__main__":
    main()
