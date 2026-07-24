// Puter.js image microservice.
//
// Runs a single headless Chromium page with Puter.js loaded and exposes a
// local HTTP endpoint that drives puter.ai.txt2img in the browser context.
// The same call handles both CoIG operations:
//   - initial generation: text -> image        (no input_image)
//   - refinement / edit:   image + text -> image (with input_image)
//
// Auth: Puter bills AI calls to a signed-in Puter account. A fresh headless
// browser is NOT signed in, so run once with HEADFUL=1 (npm run login), click
// "Sign in to Puter" on the served page, then POST /save-auth (or Ctrl-C) to
// persist storageState.json. Subsequent headless runs reuse it.
//
// NOTE: model identifier for Gemini image via Puter is not documented; default
// below is a best guess. Hit GET /models to discover the real id, then set
// PUTER_IMAGE_MODEL. Puter has no SLA -- this depends on its uptime, not ours.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.PORT || 8787);
// Support both `HEADFUL=1 node server.js` (Unix shells) and
// `node server.js --headful` (portable, works on Windows PowerShell/cmd too).
const HEADFUL = process.env.HEADFUL === "1" || process.argv.includes("--headful");
const DEFAULT_MODEL = process.env.PUTER_IMAGE_MODEL || "gemini-2.5-flash-image";
const DEFAULT_TIMEOUT_MS = Number(process.env.PUTER_TIMEOUT_MS || 180000);
const STATE_PATH = path.join(__dirname, "storageState.json");

const PAGE_HTML = `<!doctype html>
<html>
  <head><meta charset="utf-8"><title>puter-image-service</title></head>
  <body>
    <h3>puter-image-service</h3>
    <p id="status">loading puter.js...</p>
    <button id="signin">Sign in to Puter</button>
    <script src="https://js.puter.com/v2/"></script>
    <script>
      const status = document.getElementById("status");
      function refresh() {
        try {
          status.textContent = puter.auth.isSignedIn()
            ? "signed in - you can POST /save-auth now, then stop and run headless"
            : "not signed in - click the button";
        } catch (e) { status.textContent = "puter not ready: " + e; }
      }
      document.getElementById("signin").onclick = async () => {
        try { await puter.auth.signIn(); refresh(); }
        catch (e) { status.textContent = "sign-in failed: " + e; }
      };
      const ready = setInterval(() => {
        if (window.puter && puter.ai) { clearInterval(ready); refresh(); }
      }, 200);
    </script>
  </body>
</html>`;

let browser;
let context;
let page;

// Serialize page work: one page, so requests must not run evaluate concurrently.
let chain = Promise.resolve();
function enqueue(task) {
  const result = chain.then(task);
  chain = result.then(() => {}, () => {});
  return result;
}

function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise.finally(() => clearTimeout(timer)), timeout]);
}

// Runs inside the browser page. Returns a data: URL string for the image.
async function browserGenerate({ prompt, input_image, model, testMode }) {
  const options = {};
  if (model) options.model = model;
  if (input_image) options.input_image = input_image;

  const result = await window.puter.ai.txt2img(prompt, !!testMode, options);

  // Puter may return an <img> element, a URL string, or a data URL.
  let src = result && result.src ? result.src : result;
  if (typeof src !== "string") {
    throw new Error("Unexpected txt2img result type: " + Object.prototype.toString.call(result));
  }
  if (src.startsWith("data:")) return src;

  // blob: or http(s): -> fetch and convert to a data URL so it survives the bridge.
  const resp = await fetch(src);
  const blob = await resp.blob();
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("failed to read image blob"));
    reader.readAsDataURL(blob);
  });
}

async function startBrowser() {
  browser = await chromium.launch({ headless: !HEADFUL });
  const hasState = fs.existsSync(STATE_PATH);
  context = await browser.newContext(hasState ? { storageState: STATE_PATH } : {});
  page = await context.newPage();
  page.on("console", (msg) => {
    if (msg.type() === "error") console.error("[page]", msg.text());
  });
  // Serve our own origin so Puter's auth state has a stable place to live.
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.puter && window.puter.ai, null, { timeout: 30000 });
  console.log(hasState ? "Loaded saved Puter auth state." : "No saved auth state -- run `npm run login` first.");
}

const app = express();
app.use(express.json({ limit: "32mb" }));

// Serve the Puter host page.
app.get("/", (_req, res) => res.type("html").send(PAGE_HTML));

app.get("/health", async (_req, res) => {
  try {
    const signedIn = await page.evaluate(() => {
      try { return window.puter.auth.isSignedIn(); } catch { return false; }
    });
    res.json({ ok: true, signed_in: signedIn, default_model: DEFAULT_MODEL });
  } catch (e) {
    res.status(503).json({ ok: false, error: String(e) });
  }
});

// Discover which models Puter actually exposes (the Gemini image id is undocumented).
app.get("/models", async (_req, res) => {
  try {
    const models = await enqueue(() =>
      page.evaluate(async () => {
        try { return await window.puter.ai.listModels(); }
        catch (e) { return { error: String(e) }; }
      })
    );
    res.json({ ok: true, models });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

// Persist the current (signed-in) auth state. Run after signing in headful.
app.post("/save-auth", async (_req, res) => {
  try {
    await context.storageState({ path: STATE_PATH });
    res.json({ ok: true, saved_to: STATE_PATH });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

app.post("/generate-image", async (req, res) => {
  const { prompt, input_image, model, testMode, timeout_ms, save_path } = req.body || {};
  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ ok: false, error: "`prompt` (string) is required" });
  }
  const ms = Number(timeout_ms || DEFAULT_TIMEOUT_MS);
  const useModel = model || DEFAULT_MODEL;

  try {
    const dataUrl = await enqueue(() =>
      withTimeout(
        page.evaluate(browserGenerate, {
          prompt,
          input_image: input_image || null,
          model: useModel,
          testMode: !!testMode,
        }),
        ms,
        "generate-image"
      )
    );

    if (save_path) {
      const base64 = dataUrl.split(",", 2)[1] || "";
      fs.mkdirSync(path.dirname(save_path), { recursive: true });
      fs.writeFileSync(save_path, Buffer.from(base64, "base64"));
      return res.json({ ok: true, image_data_url: dataUrl, saved_to: save_path, model: useModel });
    }
    return res.json({ ok: true, image_data_url: dataUrl, model: useModel });
  } catch (e) {
    return res.status(500).json({ ok: false, error: String(e && e.message ? e.message : e) });
  }
});

async function shutdown() {
  try {
    if (context && fs.existsSync(STATE_PATH)) await context.storageState({ path: STATE_PATH });
  } catch {}
  try { await browser?.close(); } catch {}
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

app.listen(PORT, async () => {
  console.log(`puter-image-service listening on http://localhost:${PORT} (headful=${HEADFUL})`);
  await startBrowser();
  console.log("Ready.");
});
