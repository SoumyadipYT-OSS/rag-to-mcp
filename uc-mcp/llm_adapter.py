"""
llm_adapter.py — Swappable LLM Call
Default: Google Gemini (free tier)
Alternatives: Claude, OpenAI — uncomment the relevant section

SETUP (Gemini default):
  1. Get a free API key at https://aistudio.google.com/app/apikey
  2. Set environment variable:
       export GEMINI_API_KEY="your-key-here"   (Mac/Linux)
       set GEMINI_API_KEY=your-key-here        (Windows CMD)
  3. Install:
       pip3 install google-generativeai

ALTERNATIVE — Claude:
  export ANTHROPIC_API_KEY="your-key-here"
  pip3 install anthropic
  Uncomment the Claude section below and comment out the Gemini section.

ALTERNATIVE — OpenAI:
  export OPENAI_API_KEY="your-key-here"
  pip3 install openai
  Uncomment the OpenAI section below and comment out the Gemini section.
"""

import os

# ══════════════════════════════════════════════════════════════════════
# GEMINI (DEFAULT)
# ══════════════════════════════════════════════════════════════════════
def call_llm(prompt: str) -> str:
    """
    Call Gemini Flash with the given prompt.
    Returns the text response as a string.
    Uses the new google.genai SDK (replaces deprecated google.generativeai).
    """
    # Load .env if GEMINI_API_KEY is not already set
    if not os.environ.get("GEMINI_API_KEY"):
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

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "[LLM NOT CONFIGURED] Set GEMINI_API_KEY environment variable.\n"
            "Get a free key at https://aistudio.google.com/app/apikey\n\n"
            "Prompt that would have been sent:\n" + prompt[:500] + "..."
        )
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        return response.text
    except ImportError:
        # Fallback to deprecated google.generativeai if google.genai not available
        try:
            import google.generativeai as old_genai  # type: ignore
            old_genai.configure(api_key=api_key)
            model = old_genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"[LLM ERROR] {str(e2)}"
    except Exception as e:
        return f"[LLM ERROR] {str(e)}"


# ══════════════════════════════════════════════════════════════════════
# CLAUDE (ALTERNATIVE — uncomment to use)
# ══════════════════════════════════════════════════════════════════════
# def call_llm(prompt: str) -> str:
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     if not api_key:
#         return "[LLM NOT CONFIGURED] Set ANTHROPIC_API_KEY environment variable."
#     try:
#         import anthropic
#         client = anthropic.Anthropic(api_key=api_key)
#         message = client.messages.create(
#             model="claude-3-haiku-20240307",
#             max_tokens=1024,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return message.content[0].text
#     except ImportError:
#         return "[ERROR] anthropic not installed. Run: pip3 install anthropic"
#     except Exception as e:
#         return f"[LLM ERROR] {str(e)}"


# ══════════════════════════════════════════════════════════════════════
# OPENAI (ALTERNATIVE — uncomment to use)
# ══════════════════════════════════════════════════════════════════════
# def call_llm(prompt: str) -> str:
#     api_key = os.environ.get("OPENAI_API_KEY")
#     if not api_key:
#         return "[LLM NOT CONFIGURED] Set OPENAI_API_KEY environment variable."
#     try:
#         from openai import OpenAI
#         client = OpenAI(api_key=api_key)
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content
#     except ImportError:
#         return "[ERROR] openai not installed. Run: pip3 install openai"
#     except Exception as e:
#         return f"[LLM ERROR] {str(e)}"


# ══════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Testing LLM adapter...")
    result = call_llm("Say 'LLM adapter working' and nothing else.")
    print(f"Response: {result}")
