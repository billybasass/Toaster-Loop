// panels/models.js — model registry: table, detail drawer (validation report,
// artifact paths, lineage chain), promote/rollback with confirm + state gates.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, toast, badge, table, empty, fmtTime, confirmDialog, copyButton,
  fmtNum } from "../ui.js";

// Registry state machine: which actions are legal for a given status.
const CAN_PROMOTE = new Set(["validated"]);
const ROLLBACKABLE = (m, activeId) => m.model_id === activeId;

// Pass 4 #29: relative age of the last activation ("3d ago" / "never"),
// next to the existing created date. last_used_at flows through the models
// API row (registry to_dict); None means never served/batched.
function fmtLastUsed(iso) {
  if (!iso) return "never";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "never";
  const days = (Date.now() - t) / 86400000;
  if (days < 1) return "today";
  if (days < 30) return `${Math.floor(days)}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

export default {
  title: "Models",
  render(mount, params) {
    const tableHost = el("div");
    const drawer = el("aside", { class: "drawer hidden", role: "complementary",
      "aria-label": "model detail" });
    let data = { models: [], edges: [], active_model: null };
    const unsubs = [];

    async function load() {
      data = await api("/api/models");
      renderTable();
      if (!drawer.classList.contains("hidden") && drawer.dataset.modelId)
        openDrawer(drawer.dataset.modelId);
    }

    function renderTable() {
      // No dedup by output_dir needed (catalog pattern): registry
      // model_ids are unique rows, so duplicates cannot reach this list.
      const models = [...(data.models || [])].reverse(); // newest first
      tableHost.replaceChildren(table([
        { label: "id", render: (m) => {
            // Backlog #15: catalog-style detail line under the name,
            // composed server-side ("dataset · N steps · date · loss").
            const cell = el("div");
            cell.appendChild(el("span", { class: "mono" },
              m.model_id, m.model_id === data.active_model ? " ★" : ""));
            if (m.detail)
              cell.appendChild(el("div", { class: "hint", style:
                "font-size:0.8em; opacity:0.75;", text: m.detail }));
            return cell;
          } },
        { label: "base", key: "base_model" },
        { label: "status", render: (m) => {
            const cell = el("div");
            cell.appendChild(badge(m.status));
            // Pass 4 #41: derived capability tier badge (registry/tiers.py —
            // computed server-side at read time, never stored). Context only:
            // manual promote/rollback stay permissive. Tooltip is the honest
            // missing-evidence hint for the NEXT tier.
            if (typeof m.tier === "number") {
              const tb = badge(`T${m.tier} ${m.tier_name || ""}`.trim(),
                m.tier >= 2 ? "done" : "");
              tb.title = m.tier_hint || `tier T${m.tier} (derived)`;
              cell.appendChild(el("div", {}, tb));
            }
            return cell;
          } },
        { label: "dataset", render: (m) => m.dataset_id
            ? el("a", { href: "#/datasets/" + encodeURIComponent(m.dataset_id),
                class: "mono", text: m.dataset_id,
                onclick: (ev) => ev.stopPropagation() })
            : "–" },
        { label: "created", render: (m) => {
            const cell = el("div");
            cell.appendChild(el("span", {}, fmtTime(m.created_at, false)));
            // Pass 4 #29: last activation stamp ("3d ago" / "never").
            cell.appendChild(el("div", { class: "hint", style:
              "font-size:0.8em; opacity:0.75;",
              text: "used " + fmtLastUsed(m.last_used_at) }));
            return cell;
          } },
        { label: "validation", render: (m) => {
            const cell = el("div");
            const v = m.validation;
            if (v)
              cell.appendChild(badge(v.ok ? "pass" : "fail",
                v.ok ? "done" : "failed"));
            else cell.appendChild(el("span", { text: "–" }));
            // Pass 4 #32: shadow-baseline verdict badge next to the losses,
            // following the #29 last-used pattern (row field flows through
            // registry to_dict; absent until a shadow_eval job records one).
            const s = m.shadow;
            if (s && !s.skipped) {
              const imp = typeof s.improvement === "number"
                ? (s.improvement >= 0 ? "+" : "") + s.improvement.toFixed(2)
                : "?";
              cell.appendChild(el("div", {}, badge(
                `shadow ${imp}${s.is_regression ? " ⚠" : ""}`,
                s.is_regression ? "failed" : "done")));
            }
            // Pass 4 #37: cross-profile transfer probes (#32 machinery on a
            // foreign profile's replay set) — compact per-profile badges from
            // transfers_json (newest entry per profile wins). Not a matrix
            // view; deliberately the smallest thing that answers "has this
            // adapter been probed against profile B, and how did it go".
            const seen = new Set();
            for (const t of (m.transfers || [])) {
              const p = String(t.profile || "?");
              if (seen.has(p)) continue; // list is newest-first
              seen.add(p);
              const imp = typeof t.improvement === "number"
                ? (t.improvement >= 0 ? "+" : "") + t.improvement.toFixed(1)
                : (typeof t.candidate_mean === "number"
                   ? `~${t.candidate_mean.toFixed(1)}` : "?");
              const b = badge(
                `${p} ${imp}${t.is_regression ? " ⚠" : ""}`,
                t.is_regression ? "failed" : "done");
              b.title = `transfer probe ${t.ts || "?"} · n=${t.n ?? "?"}` +
                ` · baseline ${t.baseline_id || "none (candidate-only)"}`;
              cell.appendChild(el("div", {}, b));
            }
            return cell;
          } },
      ], models, {
        empty: "No models yet — they appear after a training run exports one.",
        rowClass: (m) => m.model_id === data.active_model ? "active-row" : "",
        onRowClick: (m) => openDrawer(m.model_id),
      }));
    }

    // ------------------------------------------------------------- drawer
    async function openDrawer(modelId) {
      drawer.dataset.modelId = modelId;
      drawer.classList.remove("hidden");
      drawer.replaceChildren(el("p", { class: "hint", text: "loading…" }));
      let d;
      try { d = await api("/api/models/" + encodeURIComponent(modelId)); }
      catch (e) {
        drawer.replaceChildren(el("p", { class: "err", text: e.message }));
        return;
      }
      const m = d.model;
      const closeBtn = el("button", { class: "small", text: "✕",
        "aria-label": "close", onclick: () => drawer.classList.add("hidden") });

      drawer.replaceChildren(
        el("div", { class: "drawer-head" },
          el("h3", { text: m.model_id }), badge(m.status), closeBtn),
        lineageBlock(m),
        kvBlock([
          ["base model", m.base_model || "–"],
          ["dataset", m.dataset_id
            ? el("a", { href: "#/datasets/" + encodeURIComponent(m.dataset_id),
                class: "mono", text: m.dataset_id })
            : "–"],
          ["parent adapter", m.parent_adapter || "–"],
          ["quant", m.quant || "–"],
          ["created", fmtTime(m.created_at)],
        ]),
        el("h4", { text: "Artifacts" }),
        pathRow("adapter", d.adapter_path),
        pathRow("gguf", d.gguf_path),
        validationBlock(d.validation),
        gateTrendBlock(modelId),
        actionsBlock(m));
    }

    // WP9b: validation-gate trend for THIS model from the analytics API —
    // pass-rate history plus merged promote/rollback events, newest first.
    // Purely additive: an empty or failed history renders as a hint only.
    function gateTrendBlock(modelId) {
      const host = el("div");
      host.appendChild(el("h4", { text: "Gate trend" }));
      const body = el("p", { class: "hint", text: "loading…" });
      host.appendChild(body);
      api("/api/analytics/validation?model=" + encodeURIComponent(modelId))
        .then((d) => {
          const tl = [...(d.timeline || [])].reverse(); // newest first
          if (!tl.length) {
            body.textContent = "No validation or promotion events recorded yet.";
            return;
          }
          const rows = tl.slice(0, 20).map((e) => {
            const when = fmtTime(e.ts, false);
            if (e.kind === "promotion") {
              return el("tr", {}, el("td", { text: when }),
                el("td", {}, badge(e.op, e.op === "promote" ? "done" : "failed")),
                el("td", { class: "hint", text: e.intent || e.actor || "" }));
            }
            const rate = typeof e.pass_rate === "number"
              ? (e.pass_rate * 100).toFixed(0) + "%" : "?";
            return el("tr", {}, el("td", { text: when }),
              el("td", {}, badge("gate " + (e.gate_verdict || "?"),
                e.gate_verdict === "pass" ? "done" : "failed")),
              el("td", { class: "hint" },
                `${rate} (${e.passed ?? 0}/${(e.passed ?? 0) + (e.failed ?? 0)} ran` +
                `${e.skipped ? `, ${e.skipped} skipped` : ""})`));
          });
          body.replaceChildren(el("table", { class: "kv" }, el("tbody", {}, ...rows)));
        })
        .catch((e) => { body.textContent = "trend unavailable: " + e.message; });
      return host;
    }

    function kvBlock(rows) {
      return el("table", { class: "kv" }, el("tbody", {},
        ...rows.map(([k, v]) => el("tr", {},
          el("td", { class: "k", text: k }),
          el("td", { text: String(v ?? "–") })))));
    }

    function pathRow(label, p) {
      if (!p) return el("p", { class: "hint" }, label + ": none");
      return el("div", { class: "pathrow" },
        el("span", { class: "k mono", text: label }),
        el("code", { class: "mono path", text: p }), copyButton(p));
    }

    function validationBlock(v) {
      const host = el("div");
      if (!v) {
        host.appendChild(el("p", { class: "hint", text: "No validation report yet." }));
        return host;
      }
      host.appendChild(el("h4", {}, "Validation ", badge(v.ok ? "pass" : "fail",
        v.ok ? "done" : "failed")));
      const entries = Object.entries(v);
      if (entries.length <= 1) return host;
      host.appendChild(el("pre", { class: "log json" },
        JSON.stringify(v, null, 2)));
      return host;
    }

    // Lineage chain: walk parent_adapter links upward from this model.
    function lineageBlock(m) {
      const byId = new Map((data.models || []).map((x) => [x.model_id, x]));
      const chain = [];
      let cur = m;
      const seen = new Set();
      while (cur && !seen.has(cur.model_id)) {
        seen.add(cur.model_id);
        chain.push(cur);
        cur = cur.parent_adapter ? byId.get(cur.parent_adapter) : null;
      }
      const host = el("div", { class: "lineage" });
      host.appendChild(el("h4", { text: "Lineage" }));
      const row = el("div", { class: "chain" });
      chain.reverse().forEach((x, i) => {
        if (i) row.appendChild(el("span", { class: "arrow", text: "→" }));
        const chip = el("button", {
          class: "chip" + (x.model_id === m.model_id ? " cur" : "") +
            (x.model_id === data.active_model ? " active" : ""),
          text: x.model_id, title: `${x.status} · ${x.base_model || "?"}`,
        });
        chip.addEventListener("click", () => openDrawer(x.model_id));
        row.appendChild(chip);
      });
      host.appendChild(row);
      if (!chain.length || (chain.length === 1 && !m.parent_adapter))
        host.appendChild(el("p", { class: "hint",
          text: "Root model (no trained parent in this registry)." }));
      return host;
    }

    function actionsBlock(m) {
      const host = el("div", { class: "btnrow" });
      const isActive = m.model_id === data.active_model;
      const promoteBtn = el("button", { class: "primary", text: "Promote" });
      promoteBtn.disabled = !CAN_PROMOTE.has(m.status) || isActive;
      if (promoteBtn.disabled)
        promoteBtn.title = isActive ? "already active"
          : `status “${m.status}” cannot be promoted (needs validation pass)`;
      promoteBtn.addEventListener("click", async () => {
        const ok = await confirmDialog({
          title: `Promote ${m.model_id}?`,
          body: "This model becomes the active model for serving and future " +
            "training lineage. The previous active model is superseded.",
          confirmLabel: "Promote",
        });
        if (!ok) return;
        // Pass 4 #36: optional intent one-liner — ledgered with the op so
        // `history --intents` can show expectation vs observed delta later.
        const intent = prompt(
          `Why promote ${m.model_id}? Expected effect (optional):`)?.trim();
        try {
          await api("/api/models/promote", { method: "POST",
            body: { model_id: m.model_id, ...(intent ? { intent } : {}) } });
          toast(`promoted ${m.model_id}`, "ok"); // bus event refreshes too
        } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
      });

      const rbBtn = el("button", { class: "danger", text: "Roll back" });
      rbBtn.disabled = !ROLLBACKABLE(m, data.active_model);
      if (rbBtn.disabled) rbBtn.title = "only the active model can be rolled back";
      rbBtn.addEventListener("click", async () => {
        const ok = await confirmDialog({
          title: "Roll back active model?",
          body: "The registry reverts to the previous promoted model (or none).",
          confirmLabel: "Roll back", danger: true,
        });
        if (!ok) return;
        // Pass 4 #36: optional "why" behind the rollback (empty is fine).
        const intent = prompt(
          "Why roll back? Expected effect (optional):")?.trim();
        try {
          await api("/api/models/rollback", { method: "POST",
            body: intent ? { intent } : {} });
          toast("rolled back", "ok");
        } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
      });

      const serveBtn = el("button", { text: "Serve this model" });
      serveBtn.disabled = !m.gguf_path;
      if (!m.gguf_path) serveBtn.title = "model has no GGUF artifact";
      serveBtn.addEventListener("click", async () => {
        try {
          const r = await api("/api/serve/deploy", { method: "POST",
            body: { model_id: m.model_id } });
          toast(`deploying ${m.model_id} → ${r.base_url}`, "ok");
        } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
      });

      host.append(promoteBtn, rbBtn, serveBtn);
      return host;
    }

    // ------------------------------------------------------------- wiring
    const toolbar = el("div", { class: "toolbar" },
      el("button", { class: "small", text: "refresh", onclick: () =>
        load().catch((e) => toast(e.message, "error")) }));
    mount.replaceChildren(el("h2", { text: "Models" }), toolbar,
      el("div", { class: "with-drawer" }, tableHost, drawer));

    load().catch((e) => tableHost.replaceChildren(empty("failed to load: " + e.message)));

    unsubs.push(bus.on("model.promoted", () => load()));
    unsubs.push(bus.on("model.rolled_back", () => load()));
    unsubs.push(bus.on("job.state", (p) => {
      if (["done", "failed"].includes(p.state)) load(); // new model may exist
    }));

    return { destroy() { for (const u of unsubs) u(); } };
  },
};
