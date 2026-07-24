# Puter image service

A tiny Node microservice that drives **Puter.js** image generation/editing from
a headless browser and exposes it over local HTTP, so the Python CoIG pipeline
can generate images without a paid Gemini API key.

> **Status: experimental, pilot-only.** This routes through an unofficial free
> proxy. It is **not** version-faithful to the paper's
> `gemini-2.5-flash-image-preview`, has no SLA, and depends on Puter's uptime.
> Use it to get the pilot moving when no API key is available — not for numbers
> you intend to publish as a faithful CoIG reproduction. See
> [`../docs/pilot-design.md`](../docs/pilot-design.md).

## Why a microservice

Puter.js only runs in a browser (it needs `window`/DOM). The Python pipeline
can't call it directly, so this wraps it in a headless Chromium page (Playwright)
and fronts it with an HTTP endpoint. `puter.ai.txt2img` accepts an optional
`input_image`, so one endpoint covers both CoIG operations:

- **initial step** (text → image): no `input_image`
- **refine steps 2–6** (image + text → image): pass the previous step's image as
  `input_image`. This is the compositional-lock behavior the pilot audits.

## Install

```
cd image_service
npm install
npx playwright install chromium
```

## One-time sign-in (required)

Puter bills AI calls to a signed-in Puter account. A headless browser starts
signed-out, so do this once:

```
npm run login          # launches a VISIBLE browser at http://localhost:8787/
```

In that window: click **Sign in to Puter** and complete sign-in. Then persist the
session and stop the server:

```
curl -X POST http://localhost:8787/save-auth
# then Ctrl-C
```

This writes `storageState.json` (gitignored — it holds a session token). Headless
runs reuse it automatically.

## Run (headless)

```
npm start              # http://localhost:8787
```

Endpoints:

| Method | Path              | Purpose |
|--------|-------------------|---------|
| GET    | `/health`         | up? signed in? default model |
| GET    | `/models`         | list models Puter exposes (find the Gemini **image** id) |
| POST   | `/generate-image` | `{ prompt, input_image?, model?, timeout_ms?, save_path? }` → `{ ok, image_data_url, model }` |
| POST   | `/save-auth`      | persist current signed-in state |

## Smoke test

```
npm run test:generate  # writes test_output/test_txt2img.png and test_output/test_edit.png
```

It checks, in order: health → model list → text→image → image+text→image edit.
**Open both PNGs.** The edit must preserve the scene and only recolor the bowl —
if it regenerates from scratch instead, Puter's `input_image` isn't doing true
editing for this model and the CoIG chain won't be valid (tell me and we'll
rethink the backend).

## The undocumented model id

Puter's docs confirm Gemini image generation ("Nano Banana") exists but don't
give an identifier. Default is `gemini-2.5-flash-image` (a guess). If generation
fails or ignores `input_image`, run `/models`, find the real image-capable id,
and set it:

```
PUTER_IMAGE_MODEL="<real-id>" npm start
```

## Wiring into the pipeline

The ARM generator [`../coig/create_images/generate_multi_step_image_genai_simple.py`](../coig/create_images/generate_multi_step_image_genai_simple.py)
now has a `_puter_generate()` helper, and both `generate_initial_image()` and
`edit_image()` route through it **only when `USE_PUTER_SERVICE=1`**. Default
(unset) keeps the official Gemini SDK path untouched.

Run the pilot's generation step against this service (from `coig/create_images/`,
with the service already running):

```
USE_PUTER_SERVICE=1 PUTER_IMAGE_MODEL="<id-from-/models>" \
  python generate_multi_step_image_genai_simple.py \
  --csv ../create_prompt/sbs_prompts_results.csv --outdir multi_step_out
```

Env vars honored by the Python side: `USE_PUTER_SERVICE`, `PUTER_SERVICE_URL`
(default `http://localhost:8787/generate-image`), `PUTER_IMAGE_MODEL`,
`PUTER_TIMEOUT` (seconds). The judge pipeline is untouched — only image
generation is rerouted.

> Note: that Python file lives in the `coig/` **submodule** (your fork), so its
> change commits inside `coig/`, separately from the parent repo.
