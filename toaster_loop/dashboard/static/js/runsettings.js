// runsettings.js — shared "model run settings" editor (reference style).
// Used by the Settings page (defaults for the default server) and the Serve
// page (per-instance settings captured when an instance is created/edited).
//
// Vision follows the studio pattern: NO mmproj path input. We ask the backend
// (GET /api/mmproj?gguf=...) whether a projector GGUF sits next to the model;
// if not but the hub repo publishes one, we offer a download into that same
// folder. The user only ever flips a "load vision projector" toggle.
//
// Speculative decoding mirrors studio's mode dropdown (auto / ngram / mtp /
// chained / off) plus an optional draft length (-1 = platform default).

import { api } from "./api.js";
import { el, toast } from "./ui.js";

const KV_TYPES = ["f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl"];

// Sampling + thinking defaults (reference style): request-time
// parameters merged into every /chat/completions call — per server instance
// (Serve page) and globally (Settings page). Distinct from the llama-server
// CLI run settings above; these work for attached servers too.
export const REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high"];
export const SPEC_MODES = [
  ["", "— off (llama.cpp default) —"],
  ["auto", "auto (recommended: n-gram self-speculation)"],
  ["ngram", "ngram (zero-VRAM self-speculation)"],
  ["mtp", "mtp (needs a model with an MTP head / drafter)"],
  ["ngram+mtp", "ngram + mtp (chained)"],
  ["off", "off (explicitly disabled)"],
];

function field(labelText, input, hint) {
  // Hover tooltip: the inline hint doubles as the control's title so every
  // knob explains itself on hover (widgets with their own title are left alone).
  if (hint && input && input.setAttribute && !input.getAttribute("title"))
    input.setAttribute("title", String(hint));
  return el("div", { class: "field" }, el("label", { text: labelText }),
    input, hint ? el("span", { class: "hint", text: hint }) : null);
}

function numInput(value, ph = "") {
  return el("input", { type: "number", step: "any", value: value ?? "", placeholder: ph });
}

function triSelect(value) {
  const s = el("select", {},
    el("option", { value: "", text: "—" }),
    el("option", { value: "true", text: "true" }),
    el("option", { value: "false", text: "false" }));
  if (value === true || value === false) s.value = String(value);
  return s;
}

/**
 * Build the editor. Returns { node, getPatch() }.
 * opts:
 *   gguf(): () => string||null — current model GGUF path for mmproj status
 *   showVision (default true), showExtra (default true)
 */
export function runSettingsEditor(initial = {}, opts = {}) {
  const rs = initial || {};
  const host = el("div");

  const kvSel = el("select", {}, el("option", { value: "", text: "—" }),
    ...KV_TYPES.map((t) => el("option",
      { value: t, text: t === "f16" ? "f16 (default)" : t,
        selected: rs.kv_cache_type === t })));
  const faSel = triSelect(rs.flash_attn);
  const parIn = numInput(rs.parallel, "auto");
  const bsIn = numInput(rs.batch_size, "auto");
  const ubIn = numInput(rs.ubatch_size, "auto");
  const tpSel = triSelect(rs.tensor_parallel);
  const vramPctIn = numInput(rs.gpu_memory_pct, "100 = full offload");
  const mmapSel = triSelect(rs.no_mmap);
  const mlockSel = triSelect(rs.mlock);

  // Speculative decoding (studio-style mode dropdown + draft length).
  const specSel = el("select", {},
    ...SPEC_MODES.map(([v, t]) => el("option", { value: v, text: t,
      selected: (rs.speculative_type || "") === v })));
  const specNMaxIn = el("input", { type: "number", min: "1", max: "16",
    value: rs.spec_draft_n_max ?? "", placeholder: "draft n-max (default 4)" });

  // Vision: toggle + detected/downloaded mmproj status (no path input).
  const visionSel = triSelect(rs.vision === undefined ? true : rs.vision);
  const mmStatus = el("span", { class: "hint" });
  const dlBtn = el("button", { class: "small", text: "download projector",
    style: "display:none" });

  async function refreshMmproj() {
    const gguf = (opts.gguf && opts.gguf()) || "";
    dlBtn.style.display = "none";
    if (!gguf) { mmStatus.textContent = ""; return; }
    mmStatus.textContent = "checking vision projector…";
    try {
      const d = await api("/api/mmproj?gguf=" + encodeURIComponent(gguf));
      if (d.local) {
        mmStatus.textContent = `mmproj detected: ${d.local.split("/").pop()}`;
      } else if (d.downloadable) {
        mmStatus.textContent = d.repo_id
          ? `no local projector — ${d.remote[0].rfilename} available from ${d.repo_id}`
          : "no local projector found";
        dlBtn.style.display = "";
      } else {
        mmStatus.textContent = "no vision projector for this model";
      }
    } catch (e) { mmStatus.textContent = e.message; }
  }
  dlBtn.addEventListener("click", async () => {
    const gguf = (opts.gguf && opts.gguf()) || "";
    if (!gguf) return;
    dlBtn.disabled = true;
    mmStatus.textContent = "downloading projector…";
    try {
      await api("/api/mmproj/download", { method: "POST", body: { gguf } });
      toast("mmproj download started — watch the console", "ok");
      // Poll briefly; projectors are small (hundreds of MB) but not instant.
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        const d = await api("/api/mmproj?gguf=" + encodeURIComponent(gguf));
        if (d.local) { mmStatus.textContent = `downloaded: ${d.local.split("/").pop()}`;
          dlBtn.style.display = "none"; return; }
      }
      mmStatus.textContent = "download still running — check the console";
    } catch (e) { mmStatus.textContent = e.message; }
    finally { dlBtn.disabled = false; }
  });

  const extraTa = el("textarea", { rows: 2, spellcheck: "false",
    placeholder: "--rope-scaling yarn --yarn-orig-ctx 32768" });
  extraTa.value = rs.extra_args || "";

  const kids = [
    field("KV cache dtype", kvSel,
      "-ctk/-ctv: quantize the attention caches. q8_0 roughly halves their VRAM " +
      "with a tiny quality cost; blank = llama.cpp default (f16)."),
    field("Flash attention", faSel,
      "--flash-attn: faster long-context attention, less KV-cache memory. " +
      "Recommended on when the build supports it."),
    field("Parallel slots / batch / micro-batch",
      el("span", { title: "-np concurrent requests · -b logical batch · -ub physical " +
        "micro-batch. Bigger = more throughput, more VRAM. Blank = auto." },
        parIn, bsIn, ubIn),
      "-np · -b · -ub"),
    field("Tensor parallelism", tpSel,
      "--tensor-parallel: split layers across several GPUs (multi-GPU machines only)."),
    field("VRAM budget %", vramPctIn,
      "scales GPU layer offload when gpu_layers is auto (VRAM budget percentage)"),
    field("mmap / mlock", el("span", { title:
      "no-mmap reads weights directly instead of memory-mapping (helps when the OS " +
      "evicts pages); mlock pins them in RAM so they are never paged out." },
      mmapSel, mlockSel), "no-mmap · mlock"),
    field("Speculative decoding", el("span", {}, specSel, specNMaxIn),
      "--spec-type (n-gram self-speculation is free VRAM-wise; mtp needs an MTP model)"),
  ];
  if (opts.showVision !== false) {
    kids.push(field("Vision (mmproj)", el("div", {}, visionSel,
      el("div", {}, mmStatus, " ", dlBtn)),
      "load the projector GGUF found next to the model (auto-downloaded from its hub repo on request)"));
  }
  if (opts.showExtra !== false) {
    kids.push(field("Extra arguments", extraTa,
      "appended verbatim to the llama-server command"));
  }
  host.replaceChildren(...kids);

  function getPatch() {
    const p = {};
    const set = (k, v, kind) => {
      if (v === "" || v == null) return;
      p[k] = kind === "num" ? Number(v) : kind === "bool" ? v === "true" : v;
    };
    set("kv_cache_type", kvSel.value);
    set("flash_attn", faSel.value, "bool");
    set("parallel", parIn.value, "num");
    set("batch_size", bsIn.value, "num");
    set("ubatch_size", ubIn.value, "num");
    set("tensor_parallel", tpSel.value, "bool");
    set("gpu_memory_pct", vramPctIn.value, "num");
    set("no_mmap", mmapSel.value, "bool");
    set("mlock", mlockSel.value, "bool");
    set("speculative_type", specSel.value);
    set("spec_draft_n_max", specNMaxIn.value, "num");
    if (opts.showVision !== false) set("vision", visionSel.value, "bool");
    const extra = extraTa.value.trim();
    if (extra !== (rs.extra_args || "")) p.extra_args = extra;
    return p;
  }

  return { node: host, getPatch, refreshMmproj };
}

/**
 * Image-generation settings editor (diffusion engines, e.g. stable-diffusion.
 * cpp's sd-server). Same shape as runSettingsEditor: returns { node,
 * getPatch() } producing an `image_settings` map ({steps, cfg_scale, width,
 * height, sampler, scheduler, seed, batch_count, extra_args}). These are the
 * generation DEFAULTS baked into the sd-server command — chat sampling
 * (top_k/temperature) does not apply to an image model. Flag names mirror
 * `sd-server --help`.
 */
export const SD_SAMPLERS = [
  "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
  "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd",
];
export const SD_SCHEDULERS = [
  "discrete", "karras", "exponential", "ays", "sgm_uniform", "simple",
  "smoothstep", "lcm",
];

export function imageSettingsEditor(initial = {}) {
  const im = initial || {};
  const host = el("div");

  const stepsIn = numInput(im.steps, "20 (server default)");
  const cfgIn = numInput(im.cfg_scale, "7.0 (server default)");
  const wIn = numInput(im.width, "512");
  const hIn = numInput(im.height, "512");
  const samplerSel = el("select", {},
    el("option", { value: "", text: "— server default —" }),
    ...SD_SAMPLERS.map((s) => el("option",
      { value: s, text: s, selected: im.sampler === s })));
  const schedSel = el("select", {},
    el("option", { value: "", text: "— model-specific —" }),
    ...SD_SCHEDULERS.map((s) => el("option",
      { value: s, text: s, selected: im.scheduler === s })));
  const seedIn = numInput(im.seed, "-1 = random");
  const batchIn = numInput(im.batch_count, "1");
  const extraTa = el("textarea", { rows: 2, spellcheck: "false",
    placeholder: "--vae-tiling --offload-to-cpu" });
  extraTa.value = im.extra_args || "";

  host.replaceChildren(
    field("Steps", stepsIn, "--steps: denoising iterations — more can improve " +
      "detail at linearly more time"),
    field("CFG scale", cfgIn, "--cfg-scale (prompt adherence; 4.5-8 typical)"),
    field("Width × height", el("span", { title:
      "-W / -H output size in pixels; keep multiples of 8 within the model's max." },
      wIn, hIn),
      "-W / -H in pixels (multiples of 8; model-dependent max)"),
    field("Sampler", samplerSel,
      "--sampling-method: the solver taking each denoising step (euler_a is a " +
      "safe default; dpm++ families are fast and good)."),
    field("Scheduler", schedSel, "--scheduler (sigma schedule)"),
    field("Seed", seedIn, "-s (-1 = random per request; fixed seed reproduces images)"),
    field("Batch count", batchIn, "-b images generated per request"),
    field("Extra arguments", extraTa,
      "appended verbatim to the sd-server command"),
  );

  function getPatch() {
    const p = {};
    const set = (k, v, kind) => {
      if (v === "" || v == null) return;
      p[k] = kind === "num" ? Number(v) : v;
    };
    set("steps", stepsIn.value, "num");
    set("cfg_scale", cfgIn.value, "num");
    set("width", wIn.value, "num");
    set("height", hIn.value, "num");
    set("sampler", samplerSel.value);
    set("scheduler", schedSel.value);
    set("seed", seedIn.value, "num");
    set("batch_count", batchIn.value, "num");
    const extra = extraTa.value.trim();
    if (extra !== (im.extra_args || "")) p.extra_args = extra;
    return p;
  }
  return { node: host, getPatch };
}

/**
 * Sampling + thinking editor (reference style). Same shape as
 * runSettingsEditor: returns { node, getPatch() } producing a `sampling:`
 * patch ({temperature, top_p, top_k, min_p, repeat_penalty,
 * presence_penalty, max_tokens, enable_thinking, reasoning_effort,
 * preserve_thinking}). Used by the Serve page (per-instance), the Settings
 * page (global defaults) and the Profiles panel (per-agent overrides).
 */
export function samplingEditor(initial = {}) {
  const sp = initial || {};
  const host = el("div");

  const tempIn = numInput(sp.temperature, "server default");
  const topPIn = numInput(sp.top_p, "server default");
  const topKIn = numInput(sp.top_k, "server default");
  const minPIn = numInput(sp.min_p, "server default");
  const repIn = numInput(sp.repeat_penalty, "server default");
  const presIn = numInput(sp.presence_penalty, "server default");
  const maxTokIn = numInput(sp.max_tokens, "server default");

  // Thinking / reasoning controls: gate + effort + history preservation.
  const thinkSel = triSelect(sp.enable_thinking);
  const effortSel = el("select", {}, el("option", { value: "", text: "—" }),
    ...REASONING_EFFORTS.map((e) => el("option",
      { value: e, text: e, selected: sp.reasoning_effort === e })));
  const preserveSel = triSelect(sp.preserve_thinking);

  host.replaceChildren(
    field("temperature", tempIn,
      "randomness of token choice: 0 = deterministic, ~0.7 balanced, >1 creative."),
    field("top_p / top_k / min_p", el("span", { title:
      "Nucleus sampling cuts off low-probability tokens: top_p keeps the smallest " +
      "set covering p of the probability mass; top_k caps to k candidates; " +
      "min_p drops tokens below a fraction of the best one." },
      topPIn, topKIn, minPIn),
      "nucleus sampling cutoffs"),
    field("repetition / presence penalty", el("span", { title:
      "Discourage reusing tokens: repetition_penalty multiplies down already-used " +
      "tokens (>1 = stronger); presence_penalty adds a flat cost for any token " +
      "seen in the prompt." }, repIn, presIn),
      "anti-repetition nudges"),
    field("max_tokens", maxTokIn,
      "hard cap on completion length per request (blank = server default)."),
    field("Thinking (enable_thinking)", thinkSel,
      "on/off for reasoning-capable models (Qwen3, DeepSeek-R1, gpt-oss…)"),
    field("Reasoning effort", effortSel,
      "low / medium / high — for effort-style models; none disables thinking"),
    field("Preserve thinking history", preserveSel,
      "keep past  blocks in the prompt (Qwen-style templates)"),
  );

  function getPatch() {
    const p = {};
    const set = (k, v, kind) => {
      if (v === "" || v == null) return;
      p[k] = kind === "num" ? Number(v) : kind === "bool" ? v === "true" : v;
    };
    set("temperature", tempIn.value, "num");
    set("top_p", topPIn.value, "num");
    set("top_k", topKIn.value, "num");
    set("min_p", minPIn.value, "num");
    set("repeat_penalty", repIn.value, "num");
    set("presence_penalty", presIn.value, "num");
    set("max_tokens", maxTokIn.value, "num");
    set("enable_thinking", thinkSel.value, "bool");
    set("preserve_thinking", preserveSel.value, "bool");
    set("reasoning_effort", effortSel.value);
    return p;
  }
  return { node: host, getPatch };
}
