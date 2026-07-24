// Smoke test for the Puter image service.
//
// Verifies the three things that are actually uncertain about this integration,
// in order, and stops at the first failure so you know exactly what to fix:
//   1) the service is up and signed in            -> GET /health
//   2) which model id Puter really exposes         -> GET /models
//   3) text->image works                            -> POST /generate-image
//   4) image+text->image (editing) works            -> POST /generate-image w/ input_image
//
// Start the service first (see README): `npm start` (headless, after login).
// Then: `npm run test:generate`.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.PUTER_SERVICE_URL || "http://localhost:8787";
const OUT_DIR = path.join(__dirname, "test_output");

function saveDataUrl(dataUrl, filename) {
  const base64 = dataUrl.split(",", 2)[1] || "";
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const dest = path.join(OUT_DIR, filename);
  fs.writeFileSync(dest, Buffer.from(base64, "base64"));
  return dest;
}

async function main() {
  // 1) health
  const health = await (await fetch(`${BASE}/health`)).json();
  console.log("health:", health);
  if (!health.ok) throw new Error("service not healthy");
  if (!health.signed_in) {
    console.warn("WARNING: not signed in to Puter -- generation will likely fail. Run `npm run login` first.");
  }

  // 2) model discovery (informational -- the Gemini image id is undocumented)
  try {
    const models = await (await fetch(`${BASE}/models`)).json();
    console.log("models (look for a gemini *image* id):", JSON.stringify(models).slice(0, 2000));
  } catch (e) {
    console.warn("model listing failed (non-fatal):", String(e));
  }

  // 3) text -> image
  const genResp = await fetch(`${BASE}/generate-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: "A single red ceramic bowl on a plain white table, studio lighting, photorealistic",
    }),
  });
  const gen = await genResp.json();
  if (!gen.ok) throw new Error("txt2img failed: " + gen.error);
  const genPath = saveDataUrl(gen.image_data_url, "test_txt2img.png");
  console.log(`txt2img OK (model=${gen.model}) -> ${genPath}`);

  // 4) image + text -> image (the CoIG edit step -- the capability the pilot depends on)
  const editResp = await fetch(`${BASE}/generate-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: "Keep everything the same, but change the bowl's color to blue. Do not alter anything else.",
      input_image: gen.image_data_url,
    }),
  });
  const edit = await editResp.json();
  if (!edit.ok) throw new Error("image-edit failed: " + edit.error);
  const editPath = saveDataUrl(edit.image_data_url, "test_edit.png");
  console.log(`edit OK -> ${editPath}`);
  console.log("\nInspect both PNGs in test_output/: the edit should preserve the bowl/table and only recolor the bowl.");
}

main().catch((e) => {
  console.error("TEST FAILED:", e.message || e);
  process.exit(1);
});
