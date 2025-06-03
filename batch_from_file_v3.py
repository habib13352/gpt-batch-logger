#edit this to make sure results are saved in results folder

from openai import OpenAI
import os #?
import time
import csv
from dotenv import load_dotenv

# === Load your API key from .env ===
load_dotenv()
client = OpenAI()

# === Model and cost settings ===
MODEL = "gpt-3.5-turbo"           # Cheaper and solid quality
MAX_TOKENS = 1000                 # Allow longer, full responses

# Cost per 1K tokens (GPT-3.5 Turbo)
COST_PER_1K_PROMPT = 0.0015
COST_PER_1K_COMPLETION = 0.002

# === Read prompts from file ===
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# === Prepare CSV log file ===
csv_filename = "summary_log_v3.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Prompt #",
        "Prompt",
        "Response (first 250 chars)",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Tokens",
        "Estimated Cost (USD)",
        "Truncated?"
    ])

# === Track total tokens ===
total_prompt_tokens = 0
total_completion_tokens = 0

# === Process each prompt ===
for idx, prompt in enumerate(prompts, start=1):
    print(f"\n⏳ Prompt {idx}: {prompt}\n")

    # === Send to OpenAI ===
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS
    )

    # === Extract response and token info ===
    answer = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    usage = response.usage

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    total_tokens = prompt_tokens + completion_tokens
    response_cost = (
        (prompt_tokens / 1000) * COST_PER_1K_PROMPT +
        (completion_tokens / 1000) * COST_PER_1K_COMPLETION
    )
    was_truncated = finish_reason != "stop"

    # === Print response summary to terminal ===
    print(f"✅ Response {idx}:\n{answer}")
    print(f"📊 Tokens → prompt: {prompt_tokens}, completion: {completion_tokens}")
    if was_truncated:
        print(f"⚠️ Warning: Response may be incomplete (finish_reason = '{finish_reason}')")
    print("=" * 60)

    # === Save full response to versioned .txt file ===
    with open(f"response_{idx}_v3.txt", "w", encoding="utf-8") as out_file:
        out_file.write(f"Prompt {idx}:\n{prompt}\n\n")
        out_file.write(f"Response:\n{answer}\n\n")
        out_file.write("Usage Info:\n")
        out_file.write(f"- Prompt tokens: {prompt_tokens}\n")
        out_file.write(f"- Completion tokens: {completion_tokens}\n")
        out_file.write(f"- Total tokens: {total_tokens}\n")
        out_file.write(f"- Estimated cost: ${response_cost:.4f} USD\n")
        if was_truncated:
            out_file.write(f"- ⚠️ Response may have been truncated (finish_reason = '{finish_reason}')\n")

    # === Append summary to versioned CSV ===
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            idx,
            prompt,
            answer[:250],
            prompt_tokens,
            completion_tokens,
            total_tokens,
            f"${response_cost:.4f}",
            "Yes" if was_truncated else "No"
        ])

    # === Rate limit protection ===
    time.sleep(1.5)

# === Final cost summary ===
total_cost = (
    (total_prompt_tokens / 1000) * COST_PER_1K_PROMPT +
    (total_completion_tokens / 1000) * COST_PER_1K_COMPLETION
)

print(f"\n✅ All prompts complete!")
print(f"📈 Total tokens used: prompt={total_prompt_tokens}, completion={total_completion_tokens}")
print(f"💵 Estimated total cost: ${total_cost:.4f} USD")
