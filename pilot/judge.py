#!/usr/bin/env python3
"""Yes/no visual judge for the pilot.

Adapted from coig/evaluate/evaluate_sbs_images.py's evaluate_image_with_gemini(),
duplicated here (rather than imported) so pilot/ has no runtime dependency on
the vendored coig/ submodule's internal structure.

One deliberate change from the source template: it drops the "Prompt: {prompt}"
context line. The original script always passes the chain's real, true prompt
regardless of which step or question is being asked. For the Substituted
condition here, the question intentionally names an attribute that does not
belong to this chain -- passing the real prompt as context risks leaking
ground truth into the judge's decision. Asking a pure vision question keeps
all three conditions (Real, Shuffled, Substituted) judged under identical
conditions.
"""

import os
import time
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from PIL import Image as PILImage

YES_NO_TEMPLATE = """You are an AI quality auditor for text-to-image generation.

Your task is to analyze the given image and answer a yes/no question based solely on its visual content.

Your judgment must rely entirely on the image itself. The image must contain clear, unmistakable visual evidence to justify a "yes" answer -- the question cannot be answered "yes" based on assumptions or guesses.

Respond with:
- "yes" only if the answer is clearly and unambiguously yes based solely on the visual content.
- "no" in all other cases -- including if the relevant visual detail is missing, unclear, ambiguous, partially shown, obscured, or only suggested.

Do not provide any explanation, justification, or extra text. Only return a single word: either "yes" or "no".

Now, evaluate this image:

Question: {question}"""


# -------------------- Optional OpenRouter judge backend --------------------
# Free Gemini tier caps at a small daily-request quota per model (429,
# GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit 20/day). When
# USE_OPENROUTER=1, route judge calls through OpenRouter's vision-capable
# chat completions instead.

def _use_openrouter() -> bool:
    return os.getenv("USE_OPENROUTER") == "1"


def _openrouter_ask_yes_no(image_path: str, question: str, max_retries: int = 3) -> str:
    import json as _json
    import base64 as _base64
    import urllib.request as _urlreq

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return "error"
    model = os.getenv("OPENROUTER_JUDGE_MODEL", "google/gemini-2.5-flash")
    url = "https://openrouter.ai/api/v1/chat/completions"

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except Exception as exc:
        print(f"Error reading image {image_path}: {exc}")
        return "error"
    data_uri = "data:image/png;base64," + _base64.b64encode(img_bytes).decode("ascii")
    prompt = YES_NO_TEMPLATE.format(question=question)

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        "temperature": 0.0,
    }
    data = _json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            req = _urlreq.Request(url, data=data, headers=headers)
            with _urlreq.urlopen(req, timeout=float(os.getenv("OPENROUTER_TIMEOUT", "120"))) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            if isinstance(body, dict) and body.get("error"):
                raise RuntimeError(str(body["error"]))
            text = body["choices"][0]["message"]["content"].strip().lower()
            if "yes" in text:
                return "yes"
            if "no" in text:
                return "no"
            print(f"Unexpected response: {text}")
            return "error"
        except Exception as exc:
            print(f"Error on attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return "error"
    return "error"


def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        if os.getenv("USE_OPENROUTER") == "1":
            return "dummy_client"
        raise ValueError("GOOGLE_AI_API_KEY is required")
    return genai.Client(api_key=api_key)


def ask_yes_no(
    client: genai.Client,
    image_path: str,
    question: str,
    model: str = "gemini-flash-latest",
    temperature: float = 0.0,
    max_retries: int = 6,
) -> str:
    """Returns "yes", "no", or "error"."""
    if _use_openrouter():
        return _openrouter_ask_yes_no(image_path, question, max_retries=max_retries)

    config: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1000,
    }
    prompt = YES_NO_TEMPLATE.format(question=question)

    try:
        pil_image = PILImage.open(image_path)
    except Exception as exc:
        print(f"Error reading image {image_path}: {exc}")
        return "error"

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, pil_image],
                config=GenerateContentConfig(**config),
            )
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text.strip().lower()
                if "yes" in text:
                    return "yes"
                if "no" in text:
                    return "no"
                print(f"Unexpected response: {text}")
                return "error"
            print("Empty response from Gemini")
            return "error"
        except Exception as exc:
            print(f"Error on attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return "error"
    return "error"
