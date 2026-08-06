#!/usr/bin/env python3
"""
build_hard_prompts.py
Generates manifest_hard.json — the harder FLUX.1-dev prompt set designed to
force attribute-binding failures at ~50% rate.

Design goals (per Chayan + briefing doc §9.1):
  - Higher subject counts (n = 4, 5, 6)
  - Confusable same-type attributes (two red/blue aprons in one image)
  - Attribute–subject prior fights (a chef wearing a cycling helmet)
  - Near-duplicate subjects (chef + baker, nurse + doctor)

100 prompts total: 25 × n=4, 41 × n=5, 34 × n=6.
Prompt IDs start at 300 to avoid collision with the existing manifest (IDs 0–227).

Color-clash rule: within any single prompt, no two attributes share the same
color word (e.g., "red apron" and "red helmet" in the same prompt would break
T5 token-span lookup in locate_attribute_phrase). This is enforced by
check_no_color_clash() at build time.
"""
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Positional phrase templates
# ---------------------------------------------------------------------------
POS = {
    4: ["on the far left", "on the center-left", "on the center-right", "on the far right"],
    5: ["on the far left", "on the center-left", "in the middle", "on the center-right", "on the far right"],
    6: ["first from the left", "second from the left", "third from the left",
        "fourth from the left", "fifth from the left", "sixth from the left"],
}
N_NAMES = {4: "four", 5: "five", 6: "six"}

# ---------------------------------------------------------------------------
# Colors tracked for clash detection
# ---------------------------------------------------------------------------
COLOR_WORDS = {"red", "blue", "yellow", "white", "black", "dark", "round", "green", "orange"}


def color_of(attribute: str) -> str | None:
    for c in COLOR_WORDS:
        if c in attribute:
            return c
    return None


def check_no_color_clash(pairs: list[tuple[str, str]], prompt_id: int) -> None:
    seen = {}
    for subj, attr in pairs:
        c = color_of(attr)
        if c is not None:
            if c in seen:
                raise ValueError(
                    f"prompt_id {prompt_id}: color '{c}' appears in both "
                    f"'{seen[c]}' and '{attr}' — T5 span lookup will break!"
                )
            seen[c] = attr


def check_unique_subjects(pairs: list[tuple[str, str]], prompt_id: int) -> None:
    subjects = [s for s, _ in pairs]
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"prompt_id {prompt_id}: duplicate subjects {subjects}")


# ---------------------------------------------------------------------------
# Natural-language description helpers
# ---------------------------------------------------------------------------
def desc(subject: str, attribute: str) -> str:
    """Return 'a <subject> <preposition> <attribute>' phrasing."""
    if any(w in attribute for w in ("apron", "helmet", "hat")):
        return f"a {subject} in a {attribute}"
    if "gloves" in attribute:
        return f"a {subject} wearing {attribute}"
    if "sunglasses" in attribute or "glasses" in attribute:
        return f"a {subject} wearing {attribute}"
    if attribute in ("shovel", "pan", "book"):
        return f"a {subject} holding a {attribute}"
    # fallback
    return f"a {subject} with {attribute}"


def make_prompt(pairs: list[tuple[str, str]]) -> str:
    n = len(pairs)
    positions = POS[n]
    parts = [f"{pos} {desc(s, a)}" for pos, (s, a) in zip(positions, pairs)]
    return f"a photo of {N_NAMES[n]} people standing in a row, " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Entry builder
# ---------------------------------------------------------------------------
def entry(prompt_id: int, seed: int, pairs: list[tuple[str, str]]) -> dict:
    check_no_color_clash(pairs, prompt_id)
    check_unique_subjects(pairs, prompt_id)
    n = len(pairs)
    if n not in POS:
        raise ValueError(f"prompt_id {prompt_id}: unsupported n={n}")
    prompt = make_prompt(pairs)
    subjects = [s for s, _ in pairs]
    attributes = [{"attribute": a, "intended_subject": s} for s, a in pairs]
    return {
        "prompt_id": prompt_id,
        "n": n,
        "prompt": prompt,
        "subjects": subjects,
        "seed": seed,
        "detected": True,
        "num_people_detected": n,
        "image_path": "",
        "attributes": attributes,
    }


# ===========================================================================
# PROMPT DEFINITIONS
# Legend: (subject, attribute)
# Prior fights marked [PF], confusable pair marked [CF], near-duplicate [ND]
# ===========================================================================
images = []
pid = 300

# ---------------------------------------------------------------------------
# n=4 — 25 prompts (IDs 300–324)
# ---------------------------------------------------------------------------

# --- Confusable: two helmets (different colors, same noun "helmet") ---
images.append(entry(pid, 42, [("farmer", "red helmet"),    # [PF][CF]
                               ("cyclist", "yellow helmet"),
                               ("barista", "blue apron"),
                               ("teacher", "book")])); pid += 1  # 300

images.append(entry(pid, 42, [("nurse", "blue helmet"),    # [PF][CF]
                               ("cyclist", "yellow helmet"),
                               ("chef", "red apron"),
                               ("teacher", "dark sunglasses")])); pid += 1  # 301

images.append(entry(pid, 7,  [("pilot", "yellow helmet"),  # [PF][CF]
                               ("teacher", "blue helmet"),  # [PF]
                               ("barista", "red apron"),
                               ("farmer", "book")])); pid += 1  # 302

images.append(entry(pid, 42, [("chef", "red helmet"),      # [PF][CF]
                               ("farmer", "yellow helmet"), # [PF]
                               ("nurse", "white hat"),      # [PF]
                               ("barista", "dark sunglasses")])); pid += 1  # 303

# --- Confusable: two aprons (different colors, same noun "apron") ---
images.append(entry(pid, 42, [("chef", "red apron"),       # [PF][CF]
                               ("barista", "blue apron"),   # [PF]
                               ("farmer", "shovel"),
                               ("pilot", "dark sunglasses")])); pid += 1  # 304

images.append(entry(pid, 42, [("teacher", "red apron"),    # [PF][CF]
                               ("pilot", "blue apron"),     # [PF]
                               ("nurse", "yellow gloves"),  # [PF]
                               ("chef", "black hat")])); pid += 1  # 305  # [PF]

images.append(entry(pid, 7,  [("nurse", "red apron"),      # [PF][CF]
                               ("farmer", "blue apron"),    # [PF]
                               ("cyclist", "yellow helmet"),
                               ("chef", "white hat")])); pid += 1  # 306

images.append(entry(pid, 42, [("barista", "white apron"),  # [CF] barista w/non-red apron
                               ("waiter", "blue apron"),    # [ND] waiter ≈ barista
                               ("chef", "black hat"),
                               ("pilot", "dark sunglasses")])); pid += 1  # 307

# --- Confusable: two hats ---
images.append(entry(pid, 42, [("chef", "white hat"),
                               ("baker", "black hat"),      # [ND][CF] baker ≈ chef
                               ("nurse", "blue gloves"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 308

images.append(entry(pid, 7,  [("teacher", "white hat"),    # [PF][CF]
                               ("chef", "black hat"),       # [PF]
                               ("barista", "red apron"),
                               ("farmer", "shovel")])); pid += 1  # 309

images.append(entry(pid, 42, [("pilot", "white hat"),      # [PF][CF]
                               ("farmer", "black hat"),     # [PF]
                               ("cyclist", "yellow helmet"),
                               ("nurse", "blue gloves")])); pid += 1  # 310

# --- Confusable: two gloves ---
images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("farmer", "red gloves"),    # [PF][CF]
                               ("chef", "white hat"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 311

images.append(entry(pid, 7,  [("pilot", "blue gloves"),    # [PF][CF]
                               ("barista", "red gloves"),   # [PF]
                               ("teacher", "black hat"),    # [PF]
                               ("chef", "pan")])); pid += 1  # 312

# --- Pure prior-fight pairings (all attributes in wrong roles) ---
images.append(entry(pid, 42, [("nurse", "yellow helmet"),  # [PF]
                               ("chef", "shovel"),          # [PF]
                               ("barista", "dark sunglasses"),  # [PF]
                               ("farmer", "pan")])); pid += 1  # 313

images.append(entry(pid, 42, [("teacher", "red apron"),    # [PF]
                               ("pilot", "blue gloves"),    # [PF]
                               ("cyclist", "book"),         # [PF]
                               ("farmer", "white hat")])); pid += 1  # 314

images.append(entry(pid, 7,  [("barista", "white hat"),    # [PF]
                               ("nurse", "shovel"),         # [PF]
                               ("chef", "yellow helmet"),   # [PF]
                               ("teacher", "dark sunglasses")])); pid += 1  # 315

images.append(entry(pid, 42, [("pilot", "red apron"),      # [PF]
                               ("farmer", "dark sunglasses"),  # [PF]
                               ("barista", "book"),         # [PF]
                               ("cyclist", "blue gloves")])); pid += 1  # 316

images.append(entry(pid, 7,  [("farmer", "white hat"),     # [PF]
                               ("cyclist", "pan"),          # [PF]
                               ("teacher", "yellow helmet"),# [PF]
                               ("barista", "blue gloves")])); pid += 1  # 317

images.append(entry(pid, 42, [("chef", "dark sunglasses"), # [PF]
                               ("nurse", "red apron"),      # [PF]
                               ("pilot", "shovel"),         # [PF]
                               ("barista", "black hat")])); pid += 1  # 318

images.append(entry(pid, 42, [("barista", "yellow helmet"),# [PF]
                               ("teacher", "blue gloves"),  # [PF]
                               ("pilot", "white hat"),      # [PF]
                               ("farmer", "pan")])); pid += 1  # 319

# --- Near-duplicate subjects ---
images.append(entry(pid, 42, [("chef", "white hat"),
                               ("baker", "black hat"),      # [ND][CF]
                               ("barista", "red apron"),
                               ("nurse", "blue gloves")])); pid += 1  # 320

images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),    # [ND][CF]
                               ("barista", "white apron"),  # [CF] w/ blue apron
                               ("cyclist", "yellow helmet")])); pid += 1  # 321

images.append(entry(pid, 42, [("barista", "red apron"),
                               ("waiter", "blue apron"),    # [ND][CF]
                               ("chef", "white hat"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 322

images.append(entry(pid, 7,  [("barista", "black hat"),    # [PF]
                               ("waiter", "white hat"),     # [ND][PF][CF]
                               ("farmer", "shovel"),
                               ("nurse", "blue gloves")])); pid += 1  # 323

images.append(entry(pid, 42, [("teacher", "book"),
                               ("professor", "dark sunglasses"),  # [ND]
                               ("barista", "red apron"),
                               ("nurse", "blue gloves")])); pid += 1  # 324

assert pid == 325, f"Expected pid=325 after n=4 block, got {pid}"

# ---------------------------------------------------------------------------
# n=5 — 41 prompts (IDs 325–365)
# ---------------------------------------------------------------------------

# --- Confusable: two helmets ---
images.append(entry(pid, 42, [("farmer", "red helmet"),    # [PF][CF]
                               ("cyclist", "yellow helmet"),
                               ("barista", "blue apron"),
                               ("chef", "white hat"),
                               ("nurse", "shovel")])); pid += 1  # 325

images.append(entry(pid, 42, [("nurse", "blue helmet"),    # [PF][CF]
                               ("cyclist", "yellow helmet"),
                               ("chef", "red apron"),
                               ("teacher", "dark sunglasses"),
                               ("farmer", "book")])); pid += 1  # 326

images.append(entry(pid, 7,  [("pilot", "yellow helmet"),  # [PF][CF]
                               ("teacher", "blue helmet"),  # [PF]
                               ("barista", "red apron"),
                               ("nurse", "black hat"),      # [PF]
                               ("farmer", "pan")])); pid += 1  # 327

# --- Confusable: two aprons ---
images.append(entry(pid, 42, [("chef", "red apron"),       # [PF][CF]
                               ("barista", "blue apron"),   # [PF]
                               ("cyclist", "yellow helmet"),
                               ("nurse", "dark sunglasses"),# [PF]
                               ("teacher", "book")])); pid += 1  # 328

images.append(entry(pid, 42, [("teacher", "red apron"),    # [PF][CF]
                               ("pilot", "blue apron"),     # [PF]
                               ("nurse", "yellow gloves"),  # [PF]
                               ("chef", "black hat"),       # [PF]
                               ("farmer", "shovel")])); pid += 1  # 329

images.append(entry(pid, 7,  [("nurse", "red apron"),      # [PF][CF]
                               ("farmer", "blue apron"),    # [PF]
                               ("cyclist", "yellow helmet"),
                               ("chef", "white hat"),
                               ("teacher", "dark sunglasses")])); pid += 1  # 330

# --- Confusable: two hats ---
images.append(entry(pid, 42, [("chef", "white hat"),
                               ("baker", "black hat"),      # [ND][CF]
                               ("barista", "red apron"),
                               ("nurse", "blue gloves"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 331

images.append(entry(pid, 7,  [("teacher", "white hat"),    # [PF][CF]
                               ("chef", "black hat"),       # [PF]
                               ("barista", "red apron"),
                               ("nurse", "blue gloves"),
                               ("farmer", "shovel")])); pid += 1  # 332

images.append(entry(pid, 42, [("pilot", "white hat"),      # [PF][CF]
                               ("farmer", "black hat"),     # [PF]
                               ("cyclist", "yellow helmet"),
                               ("nurse", "blue gloves"),
                               ("barista", "red apron")])); pid += 1  # 333

# --- Confusable: two gloves ---
images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("farmer", "red gloves"),    # [PF][CF]
                               ("chef", "white hat"),
                               ("cyclist", "yellow helmet"),
                               ("teacher", "dark sunglasses")])); pid += 1  # 334

images.append(entry(pid, 7,  [("pilot", "blue gloves"),    # [PF][CF]
                               ("barista", "red gloves"),   # [PF]
                               ("teacher", "black hat"),    # [PF]
                               ("farmer", "yellow helmet"), # [PF]
                               ("chef", "shovel")])); pid += 1  # 335

# --- Pure prior-fight pairings (n=5) ---
images.append(entry(pid, 42, [("nurse", "yellow helmet"),  # [PF]
                               ("chef", "shovel"),          # [PF]
                               ("barista", "dark sunglasses"),  # [PF]
                               ("farmer", "pan"),
                               ("teacher", "blue gloves")])); pid += 1  # 336

images.append(entry(pid, 42, [("teacher", "red apron"),    # [PF]
                               ("pilot", "blue gloves"),    # [PF]
                               ("cyclist", "book"),         # [PF]
                               ("farmer", "white hat"),     # [PF]
                               ("nurse", "dark sunglasses")])); pid += 1  # 337

images.append(entry(pid, 7,  [("barista", "white hat"),    # [PF]
                               ("nurse", "shovel"),         # [PF]
                               ("chef", "yellow helmet"),   # [PF]
                               ("teacher", "dark sunglasses"),
                               ("farmer", "blue gloves")])); pid += 1  # 338

images.append(entry(pid, 42, [("pilot", "red apron"),      # [PF]
                               ("farmer", "dark sunglasses"),  # [PF]
                               ("barista", "book"),
                               ("cyclist", "blue gloves"),  # [PF]
                               ("chef", "yellow helmet")])); pid += 1  # 339

images.append(entry(pid, 7,  [("farmer", "white hat"),     # [PF]
                               ("cyclist", "pan"),          # [PF]
                               ("teacher", "yellow helmet"),# [PF]
                               ("barista", "blue gloves"),  # [PF]
                               ("nurse", "red apron")])); pid += 1  # 340

images.append(entry(pid, 42, [("chef", "dark sunglasses"), # [PF]
                               ("nurse", "red apron"),      # [PF]
                               ("pilot", "shovel"),         # [PF]
                               ("barista", "black hat"),    # [PF]
                               ("farmer", "yellow gloves")])); pid += 1  # 341

images.append(entry(pid, 42, [("barista", "yellow helmet"),# [PF]
                               ("teacher", "blue gloves"),  # [PF]
                               ("pilot", "white hat"),      # [PF]
                               ("farmer", "pan"),
                               ("nurse", "red apron")])); pid += 1  # 342

images.append(entry(pid, 7,  [("cyclist", "white hat"),    # [PF]
                               ("nurse", "red apron"),      # [PF]
                               ("chef", "dark sunglasses"), # [PF]
                               ("teacher", "shovel"),       # [PF]
                               ("barista", "blue gloves")])); pid += 1  # 343

images.append(entry(pid, 42, [("teacher", "yellow helmet"),# [PF]
                               ("barista", "shovel"),       # [PF]
                               ("pilot", "blue gloves"),    # [PF]
                               ("farmer", "white hat"),     # [PF]
                               ("nurse", "black hat")])); pid += 1  # 344

images.append(entry(pid, 7,  [("farmer", "dark sunglasses"),  # [PF]
                               ("cyclist", "white hat"),    # [PF]
                               ("barista", "blue gloves"),  # [PF]
                               ("chef", "red apron"),       # [PF]
                               ("nurse", "yellow helmet")])); pid += 1  # 345

images.append(entry(pid, 42, [("pilot", "shovel"),          # [PF]
                               ("teacher", "red apron"),    # [PF]
                               ("barista", "black hat"),    # [PF]
                               ("farmer", "blue gloves"),   # [PF]
                               ("chef", "yellow helmet")])); pid += 1  # 346

images.append(entry(pid, 7,  [("nurse", "white hat"),      # [PF]
                               ("barista", "shovel"),       # [PF]
                               ("chef", "dark sunglasses"), # [PF]
                               ("pilot", "yellow helmet"),  # [PF]
                               ("teacher", "red apron")])); pid += 1  # 347

images.append(entry(pid, 42, [("cyclist", "blue gloves"),  # [PF]
                               ("barista", "yellow helmet"),# [PF]
                               ("nurse", "pan"),            # [PF]
                               ("teacher", "white hat"),    # [PF]
                               ("pilot", "red apron")])); pid += 1  # 348

images.append(entry(pid, 7,  [("barista", "dark sunglasses"),  # [PF]
                               ("chef", "blue gloves"),     # [PF]
                               ("nurse", "yellow helmet"),  # [PF]
                               ("farmer", "book"),
                               ("pilot", "red apron")])); pid += 1  # 349

images.append(entry(pid, 42, [("teacher", "blue gloves"),  # [PF]
                               ("farmer", "yellow helmet"), # [PF]
                               ("barista", "white hat"),    # [PF]
                               ("nurse", "red apron"),      # [PF]
                               ("cyclist", "shovel")])); pid += 1  # 350

# --- Near-duplicate subjects (n=5) ---
images.append(entry(pid, 42, [("chef", "white hat"),
                               ("baker", "black hat"),      # [ND][CF]
                               ("nurse", "blue gloves"),
                               ("barista", "red apron"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 351

images.append(entry(pid, 7,  [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),    # [ND][CF]
                               ("barista", "white apron"),
                               ("cyclist", "yellow helmet"),
                               ("chef", "black hat")])); pid += 1  # 352

images.append(entry(pid, 42, [("barista", "red apron"),
                               ("waiter", "blue apron"),    # [ND][CF]
                               ("chef", "white hat"),
                               ("cyclist", "yellow helmet"),
                               ("nurse", "dark sunglasses")])); pid += 1  # 353

images.append(entry(pid, 42, [("teacher", "book"),
                               ("professor", "dark sunglasses"),  # [ND]
                               ("barista", "red apron"),
                               ("nurse", "blue gloves"),
                               ("farmer", "yellow helmet")])); pid += 1  # 354

images.append(entry(pid, 7,  [("chef", "pan"),
                               ("baker", "white hat"),      # [ND]
                               ("farmer", "shovel"),
                               ("nurse", "blue gloves"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 355

images.append(entry(pid, 42, [("barista", "red apron"),
                               ("waiter", "blue apron"),    # [ND][CF]
                               ("chef", "white hat"),
                               ("nurse", "black hat"),      # [PF][CF] two hats!
                               ("farmer", "shovel")])); pid += 1  # 356

images.append(entry(pid, 7,  [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),    # [ND][CF]
                               ("farmer", "shovel"),
                               ("chef", "white hat"),
                               ("teacher", "dark sunglasses")])); pid += 1  # 357

images.append(entry(pid, 42, [("chef", "white hat"),
                               ("baker", "blue apron"),     # [ND][PF][CF] baker w/apron
                               ("barista", "red apron"),    # [CF] two aprons!
                               ("farmer", "shovel"),
                               ("nurse", "dark sunglasses")])); pid += 1  # 358

images.append(entry(pid, 7,  [("teacher", "white hat"),    # [PF]
                               ("professor", "black hat"),  # [ND][CF]
                               ("barista", "red apron"),
                               ("nurse", "blue gloves"),
                               ("farmer", "shovel")])); pid += 1  # 359

images.append(entry(pid, 42, [("chef", "pan"),
                               ("cook", "white hat"),       # [ND]
                               ("barista", "red apron"),
                               ("nurse", "blue gloves"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 360

images.append(entry(pid, 7,  [("farmer", "blue gloves"),   # [PF][CF]
                               ("gardener", "red gloves"),  # [ND][PF]
                               ("barista", "white apron"),
                               ("chef", "black hat"),
                               ("cyclist", "yellow helmet")])); pid += 1  # 361

images.append(entry(pid, 42, [("barista", "red apron"),
                               ("waiter", "blue apron"),    # [ND][CF]
                               ("chef", "white hat"),
                               ("baker", "black hat"),      # [ND][CF] two hats!
                               ("nurse", "dark sunglasses")])); pid += 1  # 362

images.append(entry(pid, 7,  [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),    # [ND][CF]
                               ("chef", "white hat"),
                               ("baker", "black hat"),      # [ND][CF] two hats!
                               ("farmer", "shovel")])); pid += 1  # 363

images.append(entry(pid, 42, [("cyclist", "yellow helmet"),
                               ("biker", "red helmet"),     # [ND][CF] biker ≈ cyclist
                               ("barista", "blue apron"),   # [PF]
                               ("chef", "white hat"),
                               ("farmer", "dark sunglasses")])); pid += 1  # 364

images.append(entry(pid, 7,  [("barista", "white apron"),  # [CF]
                               ("waiter", "blue apron"),    # [ND][CF]
                               ("chef", "black hat"),
                               ("farmer", "red gloves"),    # [PF]
                               ("nurse", "dark sunglasses")])); pid += 1  # 365

assert pid == 366, f"Expected pid=366 after n=5 block, got {pid}"

# ---------------------------------------------------------------------------
# n=6 — 34 prompts (IDs 366–399)
# ---------------------------------------------------------------------------

# All 6 distinct colors to maximize confusion within each prompt
# Pattern: R, B, Y, W, K, D = red, blue, yellow, white, black, dark

images.append(entry(pid, 42, [("chef", "red apron"),        # [PF]
                               ("barista", "blue apron"),    # [PF][CF] two aprons
                               ("farmer", "yellow helmet"),  # [PF]
                               ("cyclist", "white hat"),     # [PF]
                               ("nurse", "black hat"),       # [PF][CF] two hats!
                               ("pilot", "dark sunglasses")])); pid += 1  # 366

images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("farmer", "red gloves"),     # [PF][CF] two gloves
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF] two hats
                               ("barista", "yellow apron"),  # [PF]
                               ("cyclist", "dark sunglasses")])); pid += 1  # 367

images.append(entry(pid, 42, [("barista", "red apron"),
                               ("waiter", "blue apron"),     # [ND][CF] two aprons
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF] two hats
                               ("nurse", "yellow gloves"),   # [PF]
                               ("cyclist", "dark sunglasses")])); pid += 1  # 368

images.append(entry(pid, 7,  [("cyclist", "yellow helmet"),
                               ("farmer", "red helmet"),     # [PF][CF] two helmets
                               ("nurse", "blue gloves"),
                               ("pilot", "white hat"),       # [PF]
                               ("barista", "dark sunglasses"),  # [PF]
                               ("teacher", "shovel")])); pid += 1  # 369

images.append(entry(pid, 42, [("chef", "white hat"),
                               ("teacher", "black hat"),     # [PF][CF] two hats
                               ("nurse", "blue gloves"),
                               ("barista", "red apron"),
                               ("cyclist", "yellow helmet"),
                               ("farmer", "dark sunglasses")])); pid += 1  # 370

images.append(entry(pid, 7,  [("barista", "red apron"),
                               ("chef", "blue apron"),       # [PF][CF] two aprons
                               ("farmer", "yellow helmet"),  # [PF]
                               ("nurse", "white hat"),       # [PF]
                               ("pilot", "black hat"),       # [PF][CF] two hats!
                               ("teacher", "dark sunglasses")])); pid += 1  # 371

images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),     # [ND][CF] two gloves
                               ("barista", "white apron"),
                               ("waiter", "yellow apron"),   # [ND][CF] two aprons!
                               ("chef", "black hat"),
                               ("farmer", "dark sunglasses")])); pid += 1  # 372

images.append(entry(pid, 7,  [("cyclist", "yellow helmet"),
                               ("biker", "red helmet"),      # [ND][CF] two helmets
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF] two hats!
                               ("nurse", "blue gloves"),
                               ("barista", "dark sunglasses")])); pid += 1  # 373

images.append(entry(pid, 42, [("chef", "red apron"),        # [PF]
                               ("barista", "blue apron"),    # [PF][CF]
                               ("nurse", "yellow gloves"),   # [PF]
                               ("farmer", "white hat"),      # [PF]
                               ("pilot", "black hat"),       # [PF][CF] two hats!
                               ("teacher", "dark sunglasses")])); pid += 1  # 374

images.append(entry(pid, 7,  [("barista", "red apron"),
                               ("waiter", "blue apron"),     # [ND][CF]
                               ("chef", "white hat"),
                               ("cook", "black hat"),        # [ND][CF] two hats!
                               ("nurse", "yellow gloves"),   # [PF]
                               ("farmer", "dark sunglasses")])); pid += 1  # 375

images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),     # [ND][CF]
                               ("barista", "white apron"),
                               ("chef", "black hat"),
                               ("cyclist", "yellow helmet"),
                               ("teacher", "dark sunglasses")])); pid += 1  # 376

images.append(entry(pid, 7,  [("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF]
                               ("barista", "red apron"),
                               ("waiter", "blue apron"),     # [ND][CF] two aprons!
                               ("cyclist", "yellow helmet"),
                               ("farmer", "shovel")])); pid += 1  # 377

images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("farmer", "red gloves"),     # [PF][CF]
                               ("cyclist", "yellow helmet"),
                               ("teacher", "white hat"),     # [PF]
                               ("barista", "dark sunglasses"),  # [PF]
                               ("chef", "shovel")])); pid += 1  # 378

images.append(entry(pid, 7,  [("teacher", "red apron"),     # [PF]
                               ("pilot", "blue apron"),      # [PF][CF]
                               ("nurse", "yellow helmet"),   # [PF]
                               ("farmer", "white hat"),      # [PF]
                               ("barista", "black hat"),     # [PF]
                               ("chef", "dark sunglasses")])); pid += 1  # 379

images.append(entry(pid, 42, [("barista", "yellow helmet"), # [PF]
                               ("chef", "red apron"),        # [PF]
                               ("nurse", "blue gloves"),
                               ("farmer", "white hat"),      # [PF]
                               ("pilot", "book"),
                               ("teacher", "dark sunglasses")])); pid += 1  # 380

images.append(entry(pid, 7,  [("cyclist", "white hat"),     # [PF]
                               ("barista", "blue gloves"),   # [PF]
                               ("teacher", "yellow helmet"), # [PF]
                               ("nurse", "red apron"),       # [PF]
                               ("farmer", "dark sunglasses"),  # [PF]
                               ("chef", "shovel")])); pid += 1  # 381

images.append(entry(pid, 42, [("chef", "red apron"),        # [PF]
                               ("barista", "blue apron"),    # [PF][CF]
                               ("nurse", "yellow gloves"),   # [PF]
                               ("pilot", "white hat"),       # [PF]
                               ("farmer", "black hat"),      # [PF][CF] two hats!
                               ("cyclist", "dark sunglasses")])); pid += 1  # 382

images.append(entry(pid, 7,  [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),     # [ND][CF]
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF]
                               ("barista", "yellow apron"),  # [PF]
                               ("farmer", "dark sunglasses")])); pid += 1  # 383

images.append(entry(pid, 42, [("barista", "red apron"),
                               ("chef", "blue apron"),       # [PF][CF]
                               ("nurse", "yellow gloves"),   # [PF]
                               ("pilot", "white hat"),       # [PF]
                               ("teacher", "black hat"),     # [PF][CF] two hats!
                               ("farmer", "shovel")])); pid += 1  # 384

images.append(entry(pid, 7,  [("cyclist", "yellow helmet"),
                               ("farmer", "red helmet"),     # [PF][CF]
                               ("chef", "white hat"),
                               ("nurse", "blue gloves"),
                               ("barista", "dark sunglasses"),  # [PF]
                               ("pilot", "book")])); pid += 1  # 385

images.append(entry(pid, 42, [("teacher", "red apron"),     # [PF]
                               ("barista", "white apron"),   # [CF] two aprons
                               ("chef", "blue gloves"),      # [PF]
                               ("nurse", "yellow gloves"),   # [PF][CF] two gloves!
                               ("cyclist", "black hat"),     # [PF]
                               ("farmer", "dark sunglasses")])); pid += 1  # 386

images.append(entry(pid, 7,  [("barista", "red apron"),
                               ("waiter", "blue apron"),     # [ND][CF]
                               ("nurse", "yellow gloves"),   # [PF]
                               ("doctor", "white hat"),      # [ND][PF]
                               ("chef", "black hat"),        # [CF] two hats!
                               ("farmer", "dark sunglasses")])); pid += 1  # 387

images.append(entry(pid, 42, [("chef", "white hat"),
                               ("cook", "red hat"),          # [ND][CF] three-way hat confusion!
                               ("baker", "black hat"),       # [ND][CF]
                               ("barista", "blue apron"),    # [PF]
                               ("nurse", "yellow gloves"),   # [PF]
                               ("farmer", "dark sunglasses")])); pid += 1  # 388

images.append(entry(pid, 7,  [("nurse", "blue gloves"),
                               ("farmer", "red gloves"),     # [PF][CF]
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF]
                               ("barista", "yellow apron"),  # [PF]
                               ("cyclist", "dark sunglasses")])); pid += 1  # 389

images.append(entry(pid, 42, [("barista", "red apron"),
                               ("waiter", "blue apron"),     # [ND][CF]
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF]
                               ("cyclist", "yellow helmet"),
                               ("farmer", "dark sunglasses")])); pid += 1  # 390

images.append(entry(pid, 7,  [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),     # [ND][CF]
                               ("barista", "white apron"),
                               ("waiter", "yellow apron"),   # [ND][CF]
                               ("chef", "black hat"),
                               ("pilot", "dark sunglasses")])); pid += 1  # 391

images.append(entry(pid, 42, [("cyclist", "yellow helmet"),
                               ("biker", "blue helmet"),     # [ND][CF]
                               ("barista", "red apron"),
                               ("chef", "white hat"),
                               ("nurse", "dark sunglasses"), # [PF]
                               ("farmer", "shovel")])); pid += 1  # 392

images.append(entry(pid, 7,  [("teacher", "red apron"),     # [PF]
                               ("pilot", "blue apron"),      # [PF][CF]
                               ("barista", "yellow apron"),  # [PF][CF] three-way apron!
                               ("chef", "white hat"),
                               ("nurse", "black hat"),       # [PF][CF] two hats!
                               ("farmer", "dark sunglasses")])); pid += 1  # 393

images.append(entry(pid, 42, [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),     # [ND][CF]
                               ("farmer", "yellow gloves"),  # [PF][CF] three-way gloves!
                               ("chef", "white hat"),
                               ("barista", "black hat"),     # [PF]
                               ("cyclist", "dark sunglasses")])); pid += 1  # 394

images.append(entry(pid, 7,  [("cyclist", "yellow helmet"),
                               ("farmer", "red helmet"),     # [PF][CF]
                               ("teacher", "blue helmet"),   # [PF][CF] three-way helmets!
                               ("barista", "white apron"),
                               ("chef", "black hat"),
                               ("nurse", "dark sunglasses")])); pid += 1  # 395

images.append(entry(pid, 42, [("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF]
                               ("cook", "red hat"),          # [ND][CF] three-way hat!
                               ("barista", "blue apron"),    # [PF]
                               ("nurse", "yellow gloves"),   # [PF]
                               ("farmer", "dark sunglasses")])); pid += 1  # 396

images.append(entry(pid, 7,  [("barista", "red apron"),
                               ("waiter", "blue apron"),     # [ND][CF]
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF]
                               ("nurse", "yellow gloves"),   # [PF]
                               ("doctor", "dark sunglasses")])); pid += 1  # 397

images.append(entry(pid, 42, [("cyclist", "yellow helmet"),
                               ("biker", "red helmet"),      # [ND][CF]
                               ("chef", "white hat"),
                               ("baker", "black hat"),       # [ND][CF]
                               ("barista", "blue apron"),    # [PF]
                               ("waiter", "dark sunglasses")])); pid += 1  # 398

images.append(entry(pid, 7,  [("nurse", "blue gloves"),
                               ("doctor", "red gloves"),     # [ND][CF]
                               ("barista", "white apron"),
                               ("waiter", "yellow apron"),   # [ND][CF]
                               ("chef", "black hat"),
                               ("baker", "dark sunglasses")])); pid += 1  # 399

assert pid == 400, f"Expected pid=400 after n=6 block, got {pid}"

# ===========================================================================
# Assemble and write manifest
# ===========================================================================
n4 = sum(1 for e in images if e["n"] == 4)
n5 = sum(1 for e in images if e["n"] == 5)
n6 = sum(1 for e in images if e["n"] == 6)
print(f"Built {len(images)} prompts: {n4}×n=4, {n5}×n=5, {n6}×n=6")
print(f"Total attribute rows: {sum(e['n'] for e in images)}")

manifest = {
    "model": "black-forest-labs/FLUX.1-dev",
    "candidate_seeds": [42, 7, 1234, 2024],
    "img_size": 1024,
    "num_inference_steps": 25,
    "early_window_fraction": 0.5,
    "_comment": (
        "Hard-prompt evaluation set. Prompt IDs 300–399. "
        "Designed to force FLUX.1-dev attribute-binding failures ~50% of the time. "
        "Strategies: confusable same-type attributes (two aprons/helmets/hats/gloves of "
        "different colors), attribute-subject prior fights, and near-duplicate subjects "
        "(chef+baker, nurse+doctor, barista+waiter, cyclist+biker). "
        "n=4/5/6 only; higher subject counts give lower chance baseline and more binding "
        "complexity. Prompt IDs start at 300 to avoid collision with original set (0-227)."
    ),
    "images": images,
}

out_path = Path(__file__).parent / "manifest_hard.json"
out_path.write_text(json.dumps(manifest, indent=2))
print(f"Written to {out_path}")
