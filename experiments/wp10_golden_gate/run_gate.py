"""WP10 golden-task gate — CPU end-to-end proof through the normal promotion path.

Program-complete criterion under test: "streamed backend trains a checkpoint
that passes the golden task validation gate through the normal registry
promotion path." WP10 proved GGUF-LoRA LOADS under llama-server (ROADMAP
WP10 steps 2–3 entry); this closes the remaining gate half ON CPU:

  1. train      — real streamed worker (`sable.training.worker.main`,
                  training.backend: streamed) on a tiny synthetic llama
                  checkpoint built from locally written safetensors;
  2. export     — real `worker.run_export` relocation branch (method
                  "gguf-lora", no merge/quantize);
  3. serve      — REAL serving machinery: ServingSupervisor -> spawn
                  llama-server for a locally-built tiny f32 llama base GGUF,
                  adapter passed via cfg.extra_llama_args ["--lora", lora.gguf]
                  (the supervisor's documented last-wins extra-args channel;
                  no serving code was modified);
  4. golden gate— real `training.validate.validate_model` + run_golden_suite
                  against the served base+adapter, plus a NEGATIVE CONTROL:
                  the same golden replayed against the base WITHOUT the
                  adapter must FAIL (proves the golden is tied to the adapter,
                  not vacuous);
  5. promote    — real ModelRegistry flow: register_model -> set_artifacts ->
                  set_validation(passed=True) -> promote -> active_model().

Honesty notes (cardinal rule 2):
  * The tiny random-init model cannot pass semantic goldens. The golden is a
    MINIMAL DETERMINISTIC tripwire — an exact-match substring of the greedy
    continuation that the BASE+ADAPTER produces and the base WITHOUT the
    adapter does NOT contain (the adapter effect WP10 measured). It exercises
    the real gate code path end-to-end; it says nothing about quality.
  * Determinism relies on temperature-0 greedy decoding on this machine +
    this llama.cpp build. No downloads, no network: llama-server discovery is
    pinned to an existing binary with auto-setup disabled.

Reproduce (repo root):
    .venv/bin/python -m experiments.wp10_golden_gate.run_gate
Exit 0 = gate closed; prints an evidence JSON blob at the end.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VOCAB = (
    ["[UNK]", "[PAD]", "[SYS]", "[/SYS]", "[USR]", "[/USR]",
     "[TOOL]", "[/TOOL]", "[ASST]", "[/ASST]"]
    + [f"w{i}" for i in range(54)]
)  # exactly 64: GGUF vocab == checkpoint vocab


# ------------------------------------------------------------ prerequisites
def find_binary() -> Optional[str]:
    """Existing llama-server binary or None (NEVER triggers a build/clone)."""
    env = os.environ.get("SABLE_LLAMA_SERVER")
    if env and Path(env).is_file():
        return env
    found = shutil.which("llama-server")
    if found:
        return found
    for cand in (Path.home() / ".sable" / "llama.cpp" / "build" / "bin",):
        p = cand / "llama-server"
        if p.is_file():
            return str(p)
    return None


def need_heavy_deps():
    import importlib

    for mod in ("torch", "transformers", "tokenizers", "safetensors"):
        importlib.import_module(mod)
    import numpy  # noqa: F401


# ------------------------------------------------- tiny checkpoint + tokenizer
def build_tiny_ckpt(root: Path):
    """Synthetic 2-layer llama checkpoint (vocab 64) + word tokenizer, local."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    specials = VOCAB[:10]
    vocab = {w: i for i, w in enumerate(VOCAB)}
    tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_obj.add_special_tokens(specials)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok_obj, unk_token="[UNK]",
                                   pad_token="[PAD]")
    fast.chat_template = (
        "{% for m in messages %}"
        "{% if m.role == 'system' %}[SYS]{{ m.content }}[/SYS]"
        "{% elif m.role == 'user' %}[USR]{{ m.content }}[/USR]"
        "{% else %}[ASST]{{ m.content }}[/ASST]{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}[ASST]{% endif %}")

    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=len(VOCAB), max_position_embeddings=128,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    ckpt = root / "base"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    fast.save_pretrained(str(ckpt))
    return ckpt


def train_streamed(root: Path, ckpt: Path) -> Dict[str, Any]:
    """Run the REAL streamed worker; returns its streamed_sft_done event."""
    from sable.training.worker import main as worker_main

    rows = [{"conversations": [
        {"from": "human", "value": "w5 w12"},
        {"from": "gpt", "value": "w40 w7 w33"}]}] * 4
    sft = root / "sft.jsonl"
    sft.write_text("\n".join(json.dumps(r) for r in rows))

    out = root / "out"
    train_yaml = root / "train.yaml"
    train_yaml.write_text(json.dumps({
        "model": str(ckpt),
        "data": {"local_dataset": [str(sft)], "format_type": "sharegpt"},
        "training": {"output_dir": str(out), "num_epochs": 6, "batch_size": 2,
                     "learning_rate": 1e-2, "backend": "streamed",
                     "max_seq_length": 64},
        "lora": {"lora_r": 4, "lora_alpha": 8},
    }))
    log = root / "job.log"
    rc = worker_main(["--job", "wp10-gate", "--kind", "sft",
                      "--config", str(train_yaml), "--log-path", str(log)])
    if rc != 0:
        raise RuntimeError(f"streamed worker failed rc={rc}; see {log}")
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "streamed_sft_done":
            return rec
    raise RuntimeError(f"no streamed_sft_done event in {log}")


def export_lora_gguf(root: Path, adapter_dir: str, gguf_out: Path) -> str:
    """Real run_export relocation branch (method gguf-lora)."""
    from sable.training import worker

    res = worker.run_export(
        {"_adapter_dir": adapter_dir, "_gguf_out": str(gguf_out),
         "training": {}},
        str(root / "export.log"), False)
    if res.get("method") != "gguf-lora":
        raise RuntimeError(f"unexpected export method: {res}")
    return res["gguf_path"]


# ------------------------------------------------------ tiny base GGUF builder
# Re-implementation of the WP10 scratch builder (docs/ROADMAP.md WP10 entry),
# inline per D-12; reads the trained checkpoint's safetensors.
def build_base_gguf(ckpt: Path, out_path: Path) -> None:
    import numpy as np
    from safetensors import safe_open

    STR, U32, I32, F32, BOOL = 8, 4, 5, 6, 7
    H, NH, NKV, FF, NL, V = 32, 4, 2, 64, 2, len(VOCAB)

    tensors: Dict[str, Any] = {}
    with safe_open(str(ckpt / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).float().numpy()

    names = {"token_embd.weight": "model.embed_tokens.weight",
             "output.weight": "lm_head.weight",
             "output_norm.weight": "model.norm.weight"}
    for i in range(NL):
        p, g = f"model.layers.{i}.", f"blk.{i}."
        names.update({
            g + "attn_q.weight": p + "self_attn.q_proj.weight",
            g + "attn_k.weight": p + "self_attn.k_proj.weight",
            g + "attn_v.weight": p + "self_attn.v_proj.weight",
            g + "attn_output.weight": p + "self_attn.o_proj.weight",
            g + "ffn_gate.weight": p + "mlp.gate_proj.weight",
            g + "ffn_up.weight": p + "mlp.up_proj.weight",
            g + "ffn_down.weight": p + "mlp.down_proj.weight",
            g + "attn_norm.weight": p + "input_layernorm.weight",
            g + "ffn_norm.weight": p + "post_attention_layernorm.weight",
        })

    kvs: List[Tuple[str, int, Any]] = []
    arrs: List[Tuple[str, int, list]] = []

    def kv(k, t, v):
        kvs.append((k, t, v))

    def kv_arr(k, t, vals):
        arrs.append((k, t, vals))

    kv("general.architecture", STR, "llama")
    kv("general.name", STR, "sable-tiny-gate")
    kv("general.file_type", U32, 1)  # f32
    kv("tokenizer.ggml.model", STR, "gpt2")
    kv_arr("tokenizer.ggml.tokens", STR, list(VOCAB))
    kv_arr("tokenizer.ggml.token_type", I32, [3] * 10 + [1] * (V - 10))
    kv_arr("tokenizer.ggml.scores", F32, [0.0] * V)
    kv_arr("tokenizer.ggml.merges", STR, ["w-w"])
    kv("tokenizer.ggml.bos_token_id", U32, 2)
    kv("tokenizer.ggml.eos_token_id", U32, 2)
    kv("tokenizer.ggml.padding_token_id", U32, 1)
    kv("tokenizer.ggml.add_bos_token", BOOL, True)
    kv("tokenizer.ggml.add_eos_token", BOOL, False)
    # trivial template: chat endpoint replays the last user turn verbatim
    kv("tokenizer.chat_template", STR,
       "{% for m in messages %}{{ m['content'] }}{% endfor %}")
    kv("llama.attention.head_count", U32, NH)
    kv("llama.attention.head_count_kv", U32, NKV)
    kv("llama.attention.layer_norm_rms_epsilon", F32, 1e-6)
    kv("llama.block_count", U32, NL)
    kv("llama.context_length", U32, 128)
    kv("llama.embedding_length", U32, H)
    kv("llama.feed_forward_length", U32, FF)
    kv("llama.rope.freq_base", F32, 10000.0)

    def w_str(b: bytearray, s: str):
        e = s.encode()
        b += struct.pack("<Q", len(e)) + e

    def w_val(b: bytearray, t: int, v):
        if t == STR:
            w_str(b, v)
        elif t == U32:
            b += struct.pack("<I", v)
        elif t == I32:
            b += struct.pack("<i", v)
        elif t == F32:
            b += struct.pack("<f", v)
        elif t == BOOL:
            b += struct.pack("<B", 1 if v else 0)

    kvb = bytearray()
    for k, t, v in kvs:
        w_str(kvb, k)
        kvb += struct.pack("<I", t)
        w_val(kvb, t, v)
    for k, t, vals in arrs:
        w_str(kvb, k)
        kvb += struct.pack("<I", 9) + struct.pack("<I", t) + \
            struct.pack("<Q", len(vals))
        for v in vals:
            w_val(kvb, t, v)

    info = bytearray()
    data = bytearray()
    for gi, (gname, hname) in enumerate(names.items()):
        if gi:
            data += b"\x00" * (-len(data) % 32)
        t = np.ascontiguousarray(tensors[hname].T)  # HF [out,in] -> ggml
        w_str(info, gname)
        dims = list(t.shape)
        info += struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<Q", d)
        info += struct.pack("<I", 0)  # ggml f32
        info += struct.pack("<Q", len(data))
        data += t.astype(np.float32).tobytes()

    n_kv = len(kvs) + len(arrs)
    hdr = b"GGUF" + struct.pack("<I", 3) + \
        struct.pack("<Q", len(names)) + struct.pack("<Q", n_kv)
    pre = hdr + bytes(kvb) + bytes(info)
    data_start = len(pre) + (-len(pre) % 32)
    assert len(pre) <= data_start
    with open(out_path, "wb") as f:
        f.write(pre)
        f.write(b"\x00" * (data_start - len(pre)))
        f.write(bytes(data))


# ------------------------------------------------------------------- serving
def supervisor_starter(binary: str, base_gguf: Path,
                       lora_gguf: Optional[Path]):
    """ServerStarter through the REAL ServingSupervisor.

    The adapter rides cfg.extra_llama_args — the documented last-wins extra
    args channel (serving/run_args.py precedence chain); no serving code path
    is modified or bypassed."""
    from sable.core.config import SableConfig
    from sable.serving.supervisor import ServingSupervisor

    def starter(_gguf_path: str):
        cfg = SableConfig(gguf_path=str(base_gguf), port=0,
                            context_length=128)
        cfg.llama_server_binary = binary
        cfg.llama_auto_setup = False  # hermetic: never clone/build llama.cpp
        if lora_gguf is not None:
            cfg.extra_llama_args = ["--lora", str(lora_gguf)]
        sup = ServingSupervisor(cfg)
        sup.start(ready_timeout=120.0)
        return sup

    return starter


def greedy_reply(base_url: str, model: str, prompt: str,
                 max_tokens: int = 48) -> str:
    r = httpx.post(base_url.rstrip("/") + "/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0}, timeout=120.0)
    r.raise_for_status()
    msg = ((r.json().get("choices") or [{}])[0].get("message")) or {}
    return msg.get("content") or ""


# ------------------------------------------------------------------ the run
def run(work: Optional[Path] = None, keep: bool = False) -> Dict[str, Any]:
    need_heavy_deps()
    binary = find_binary()
    if not binary:
        raise RuntimeError("llama-server not found (no build triggered, by "
                           "design). Run `sable setup` first.")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="wp10_gate_")
    root = work or Path(tmp_ctx.name)
    prev_home = os.environ.get("SABLE_HOME")  # restore on exit (pytest-safe)
    os.environ["SABLE_HOME"] = str(root / "home")  # state stays hermetic

    evidence: Dict[str, Any] = {"llama_server": binary}
    try:
        # 1. train (real streamed worker) -----------------------------------
        ckpt = build_tiny_ckpt(root)
        done = train_streamed(root, ckpt)
        evidence["train"] = {k: done.get(k) for k in
                             ("backend", "train_loss", "gguf_lora_path")}

        # 2. export (real run_export relocation branch) ----------------------
        gguf_out = root / "registered-lora.gguf"
        lora_gguf = Path(export_lora_gguf(root, done["adapter_dir"], gguf_out))
        assert lora_gguf.read_bytes()[:4] == b"GGUF"
        evidence["export"] = {"gguf_path": str(lora_gguf)}

        # 3. base GGUF + adapter-effect probe ---------------------------------
        base_gguf = root / "tiny_base.gguf"
        build_base_gguf(ckpt, base_gguf)
        prompt = "w5 w12"
        sup_a = supervisor_starter(binary, base_gguf, None)("<unused>")
        try:
            reply_base = greedy_reply(sup_a.handle.base_url,
                                      sup_a.handle.model_name, prompt)
        finally:
            sup_a.stop()
        sup_b = supervisor_starter(binary, base_gguf, lora_gguf)("<unused>")
        try:
            reply_adapted = greedy_reply(sup_b.handle.base_url,
                                         sup_b.handle.model_name, prompt)
        finally:
            sup_b.stop()
        if reply_base == reply_adapted:
            raise RuntimeError("adapter had NO effect on greedy output — "
                               "golden would be vacuous; cannot close gate")
        evidence["probe"] = {"prompt": prompt,
                             "reply_base": reply_base[:120],
                             "reply_adapted": reply_adapted[:120],
                             "changed": True}

        # 4. minimal deterministic golden --------------------------------------
        # Gate check semantics are case-insensitive SUBSTRING checks
        # (validate._check_task). This tiny random-init word-level model
        # decodes without spaces, and this adapter drives greedy output to
        # special/whitespace tokens (empty visible reply), so pick whichever
        # tripwire distinguishes base vs adapted: a substring ONLY the adapted
        # reply has (expect_contains) else one ONLY the base reply has
        # (expect_not_contains — the honest deterministic assertion that the
        # adapter measurably removed the base's characteristic continuation).
        # Both kinds run through the real gate code unchanged; the negative
        # control below proves the golden is not vacuous either way.
        low_base, low_adapt = reply_base.lower(), reply_adapted.lower()
        adapted_only = ([w for w in reply_adapted.split()
                         if w.lower() not in low_base]
                        or ([reply_adapted.strip()] if reply_adapted.strip()
                            else []))
        base_only = ([w for w in reply_base.split()
                      if w.lower() not in low_adapt]
                     or ([reply_base.strip()] if reply_base.strip() else []))
        if adapted_only:
            golden = {"prompt": prompt, "expect_contains": [adapted_only[0]]}
        elif base_only:
            golden = {"prompt": prompt,
                      "expect_not_contains": [base_only[0]]}
        else:  # pragma: no cover - guarded by the changed-output check above
            raise RuntimeError("no adapter-distinguishing substring found")
        evidence["golden"] = golden

        from sable.training.validate import ValidationConfig, validate_model

        vcfg = ValidationConfig(min_pass_rate=1.0, golden_tasks=[golden],
                                max_tokens=48)
        # POSITIVE: gate through the real starter serving base + adapter ------
        report = validate_model(
            str(lora_gguf), vcfg,
            server_starter=supervisor_starter(binary, base_gguf, lora_gguf))
        evidence["gate_with_adapter"] = {
            "ok": report.get("ok"), "pass_rate": report.get("pass_rate"),
            "reply_preview": (report.get("results") or [{}])[0]
            .get("reply_preview", "")[:120]}

        # NEGATIVE CONTROL: same golden against base WITHOUT adapter ---------
        report_neg = validate_model(
            str(lora_gguf), vcfg,
            server_starter=supervisor_starter(binary, base_gguf, None))
        evidence["gate_without_adapter"] = {
            "ok": report_neg.get("ok"),
            "pass_rate": report_neg.get("pass_rate")}

        if not report.get("ok"):
            raise RuntimeError(f"golden gate FAILED with adapter: {report}")
        if report_neg.get("ok"):
            raise RuntimeError("golden PASSED without the adapter — golden is "
                               "vacuous, gate closure would be dishonest")

        # 5. normal registry promotion path -----------------------------------
        from sable.registry import ModelRegistry

        reg = ModelRegistry(root / "registry.db")
        m = reg.register_model("sable-tiny-base", actor="wp10-gate")
        reg.set_artifacts(m.model_id, gguf_path=str(lora_gguf), quant="lora")
        reg.set_provenance(m.model_id, {"version": 1, "backend": "streamed",
                                        "source": "experiments/wp10_gate"})
        reg.set_validation(m.model_id, report, passed=True, actor="wp10-gate")
        reg.promote(m.model_id, actor="wp10-gate")
        active = reg.active_model()
        reg.close()
        if active is None or active.model_id != m.model_id:
            raise RuntimeError("promotion did not make the model active")
        evidence["promotion"] = {"model_id": m.model_id,
                                 "status": "promoted", "active": True}
        return evidence
    finally:
        if prev_home is None:
            os.environ.pop("SABLE_HOME", None)
        else:
            os.environ["SABLE_HOME"] = prev_home
        if not keep:
            tmp_ctx.cleanup()


def main() -> int:
    keep = "--keep" in sys.argv
    work = Path(sys.argv[sys.argv.index("--work") + 1]) \
        if "--work" in sys.argv else None
    try:
        evidence = run(work=work, keep=keep)
    except Exception as e:  # noqa: BLE001 - report honestly
        print(f"WP10 GOLDEN GATE PROOF: FAILED — {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    print("WP10 GOLDEN GATE PROOF: PASS")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
