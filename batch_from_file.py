#edit this to make sure results are saved in results folder


# === Import required libraries ===
from openai import OpenAI        # New v1.x OpenAI client
import os                        # To access environment variables
from dotenv import load_dotenv   # To load your .env file securely

# === Step 1: Load your API key from .env ===
# Make sure you have a file named ".env" in the same folder with:
# OPENAI_API_KEY=sk-...
load_dotenv()
client = OpenAI()  # Client picks up OPENAI_API_KEY from environment

# === Step 2: Configuration ===
MODEL = "gpt-3.5-turbo"        # Cheapest model for low‐cost testing
MAX_TOKENS = 150               # Limit each response to ~150 tokens
COST_PER_1K_PROMPT = 0.0015    # $ per 1,000 input tokens (GPT-3.5)
COST_PER_1K_COMPLETION = 0.002 # $ per 1,000 output tokens

# === Step 3: Load your prompts from prompts.txt ===
# prompts.txt should contain one prompt per line, no blank trailing lines
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# === Step 4: Initialize cost/token counters ===
total_prompt_tokens = 0
total_completion_tokens = 0

# === Step 5: Loop over each prompt ===
for idx, prompt in enumerate(prompts, start=1):
    print(f"\n⏳ Prompt {idx}: {prompt}\n")

    # 5a) Send the prompt and get a completion
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS
    )

    # 5b) Extract text and usage info
    answer = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    # 5c) Accumulate totals
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    # 5d) Display to console
    print(f"✅ Response {idx}:\n{answer}\n")
    print(f"📊 Tokens used → prompt: {prompt_tokens}, completion: {completion_tokens}")
    print("=" * 60)

    # 5e) Save each response to its own file
    with open(f"response_{idx}.txt", "w", encoding="utf-8") as out_f:
        out_f.write(f"Prompt:\n{prompt}\n\nResponse:\n{answer}")

# === Step 6: Compute and display estimated cost ===
total_cost = (
    (total_prompt_tokens / 1000) * COST_PER_1K_PROMPT +
    (total_completion_tokens / 1000) * COST_PER_1K_COMPLETION
)
print(f"\n✅ All done! Total tokens → prompt: {total_prompt_tokens}, completion: {total_completion_tokens}")
print(f"💵 Estimated total cost: ${total_cost:.4f} USD")
