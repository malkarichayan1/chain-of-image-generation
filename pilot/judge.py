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


def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_AI_API_KEY is required")
    return genai.Client(api_key=api_key)


def ask_yes_no(
    client: genai.Client,
    image_path: str,
    question: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Returns "yes", "no", or "error"."""
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
