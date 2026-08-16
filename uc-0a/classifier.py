"""
UC-0A — Complaint Classifier
classifier.py — Full implementation using R.I.C.E + CRAFT framework.

Run:
  python classifier.py --input ../data/city-test-files/test_kolkata.csv --output results_kolkata.csv
"""

import argparse
import csv
import json
import os
import re
import sys


# ── Load .env file ─────────────────────────────────────────────────────────────
def load_dotenv():
    """Load .env file variables into os.environ if not already set."""
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(4):
        env_path = os.path.join(cur, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
            break
        cur = os.path.dirname(cur)


load_dotenv()

# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    """Call Gemini using the new google.genai SDK with automatic retry on rate limits."""
    import time
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            # Rate limited — extract retry-after delay or use exponential backoff
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s...
                # Try to parse retryDelay from message
                import re
                m = re.search(r'retry.*?(\d+)s', err_str, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 2
                print(f"  [Rate limit] Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
                continue
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 10 * (attempt + 1)
                print(f"  [Unavailable] Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
                continue
            else:
                raise RuntimeError(f"LLM call failed: {e}")
    raise RuntimeError(f"LLM call failed after {max_retries} retries (rate limit)")


# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_CATEGORIES = [
    "Pothole", "Flooding", "Streetlight", "Waste", "Noise",
    "Road Damage", "Heritage Damage", "Heat Hazard", "Drain Blockage", "Other"
]

SEVERITY_KEYWORDS = [
    "injury", "child", "school", "hospital", "ambulance",
    "fire", "hazard", "fell", "collapse"
]


# ── SKILL: classify_complaint ─────────────────────────────────────────────────
def classify_complaint(row: dict) -> dict:
    """
    Classify a single complaint row.
    Returns dict with: complaint_id, category, priority, reason, flag
    """
    description = str(row.get("description", "")).strip()
    location = str(row.get("location", "")).strip()
    complaint_id = str(row.get("complaint_id", ""))

    # Vague/short description fallback
    if len(description) < 10:
        return {
            "complaint_id": complaint_id,
            "category": "Other",
            "priority": "Low",
            "reason": "Description is too short to classify.",
            "flag": "NEEDS_REVIEW"
        }

    prompt = f"""You are a Complaint Classifier agent for a city operations team.

Classify the following complaint. Output ONLY a raw JSON object with no markdown code fences.

Complaint:
  Description: {description}
  Location: {location}

Allowed categories (use EXACTLY one): Pothole, Flooding, Streetlight, Waste, Noise, Road Damage, Heritage Damage, Heat Hazard, Drain Blockage, Other

Priority rules:
  - Urgent: if description contains any of: injury, child, school, hospital, ambulance, fire, hazard, fell, collapse
  - Standard: most complaints
  - Low: minor or cosmetic issues

Enforcement:
  - category must be EXACTLY one from the allowed list. No variations.
  - reason must quote specific words from the description.
  - If category cannot be determined confidently, use "Other" and set flag to "NEEDS_REVIEW".
  - flag is "NEEDS_REVIEW" only when ambiguous, otherwise set to "" (empty string).

Return only this JSON structure (no code blocks, no extra text):
{{"category": "...", "priority": "...", "reason": "...", "flag": ""}}"""

    response_text = call_llm(prompt).strip()

    # Strip markdown fences if present
    # Match ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fence_match:
        response_text = fence_match.group(1).strip()
    else:
        # Try to extract the first JSON object
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0).strip()

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        data = {
            "category": "Other",
            "priority": "Standard",
            "reason": f"LLM returned unparseable response: {response_text[:120]}",
            "flag": "NEEDS_REVIEW"
        }

    # Enforce: category must be in allowed list
    category = data.get("category", "Other")
    if category not in ALLOWED_CATEGORIES:
        category = "Other"
        data["flag"] = "NEEDS_REVIEW"

    # Programmatic severity keyword override
    priority = data.get("priority", "Standard")
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in SEVERITY_KEYWORDS):
        priority = "Urgent"

    if priority not in ("Urgent", "Standard", "Low"):
        priority = "Standard"

    return {
        "complaint_id": complaint_id,
        "category": category,
        "priority": priority,
        "reason": data.get("reason", "No reason provided."),
        "flag": data.get("flag", "")
    }


# ── SKILL: batch_classify ──────────────────────────────────────────────────────
def batch_classify(input_path: str, output_path: str):
    """Read input CSV, classify each row, write results CSV."""
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            try:
                if "description" not in row:
                    print(f"  Row {i}: skipping — missing 'description' column.")
                    continue
                result = classify_complaint(row)
                results.append(result)
                flag_info = f"  [{result['flag']}]" if result["flag"] else ""
                print(f"  Row {i} ({result['complaint_id']}): {result['category']} | {result['priority']}{flag_info}")
            except Exception as e:
                print(f"  Row {i}: ERROR — {e}. Skipping.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = ["complaint_id", "category", "priority", "reason", "flag"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nClassified {len(results)} rows -> {output_path}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UC-0A Complaint Classifier")
    parser.add_argument("--input",  required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output results CSV file")
    args = parser.parse_args()
    print(f"Classifying complaints from: {args.input}")
    batch_classify(args.input, args.output)
    print(f"Done. Results written to {args.output}")
