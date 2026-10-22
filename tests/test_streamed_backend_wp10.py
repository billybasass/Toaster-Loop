"""WP10 steps 2–3: streamed-LoRA backend integration + artifact pipeline.

Covers (per the WP):
  * training.backend selection — default reproduces today's path, "streamed"
    routes the worker to the streamed runner (stubbed), lenient mistypes.
  * per-turn -100 loss masking incl. system/tool turns and consecutive
    assistant turns; stricter-than-airllm divergence case.
  * end-to-end streamed training on a tiny synthetic llama checkpoint built
    from safetensors shards written here (no network): splitting progress in
    the job log, train_metrics.csv schema, decreasing loss, GGUF-LoRA file.
  * GGUF-LoRA artifact: independent minimal reader asserts header/metadata/
    tensor names; export relocation + streamed lineage extraction.

Heavy deps (torch/transformers/tokenizers/safetensors) are [training]-extra:
tests skip cleanly when absent. Nothing downloads; every checkpoint and
tokenizer is synthesized in tmp dirs.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest


# ------------------------------------------------------------- backend select
def test_parse_backend_defaults_and_lenient_fallback():
    from sable.training.streamed.runner import parse_backend

    assert parse_backend(None) == "peft"          # absent => today's path
    assert parse_backend("") == "peft"
    assert parse_backend("peft") == "peft"
    assert parse_backend("standard") == "peft"    # alias of the old behavior
    assert parse_backend("streamed") == "streamed"
    assert parse_backend("STREAMED ") == "streamed"

    warns: list[str] = []
    assert parse_backend("Streamd", warn=warns.append) == "peft"
    assert "falling back" in warns[0]
    warns.clear()
    assert parse_backend(42, warn=warns.append) == "peft"
    assert warns  # mistyped non-strings warn too


def test_training_config_carries_backend():
    from sable.training.scheduler import TrainingConfig

    assert TrainingConfig().backend is None       # default: inject nothing
    tc = TrainingConfig.from_yaml({"training": {"backend": "streamed"}})
    assert tc.backend == "streamed"


def test_run_sft_routes_streamed_without_touching_peft_path(tmp_path, monkeypatch):
    from sable.training import worker

    calls = []

    def fake_streamed(cfg, log_path, dry_run):
        calls.append((cfg, dry_run))
        return {"adapter_dir": "x", "train_loss": 1.0, "eval_loss": None,
                "backend": "streamed"}

    monkeypatch.setattr(worker, "run_streamed_sft", fake_streamed)
    cfg = {"training": {"backend": "streamed"}}
    out = worker.run_sft(cfg, str(tmp_path / "job.log"), False)
    assert out["backend"] == "streamed"
    assert len(calls) == 1


def test_run_sft_default_never_calls_streamed(tmp_path, monkeypatch):
    from sable.training import worker

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("streamed backend invoked for default config")

    monkeypatch.setattr(worker, "run_streamed_sft", boom)
    # No training.backend key => old path. It fails fast on the empty config
    # (no dataset/model); what matters is the streamed spy was not called.
    with pytest.raises(Exception):
        worker.run_sft({}, str(tmp_path / "job.log"), False)


def test_resolved_train_yaml_backend_plumbing(tmp_path):
    from sable.training.pipeline import _resolved_train_yaml

    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "train.yaml").write_text(
        "model: REPLACE_WITH_BASE_MODEL_PATH_OR_HF_ID\n"
        "data:\n  local_dataset: [REPLACE]\ntraining:\n  output_dir: out\n")

    p_default, _ = _resolved_train_yaml(dataset, tmp_path / "run", None)
    text = Path(p_default).read_text()
    assert "backend" not in text                 # default: file untouched

    p_streamed, applied = _resolved_train_yaml(
        dataset, tmp_path / "run2", None, backend="streamed")
    import yaml

    data = yaml.safe_load(Path(p_streamed).read_text())
    assert data["training"]["backend"] == "streamed"
    assert applied.get("training.backend") is None  # not an override entry


def test_parse_worker_backend_lineage(tmp_path):
    from sable.training.pipeline import parse_worker_backend

    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "progress", "step": 1}) + "\n"
        + json.dumps({"event": "streamed_sft_done", "backend": "streamed",
                      "gguf_lora_path": "/x/lora.gguf"}) + "\n")
    assert parse_worker_backend(str(log)) == {
        "backend": "streamed", "gguf_lora_path": "/x/lora.gguf"}

    peft_log = tmp_path / "peft.log"
    peft_log.write_text(json.dumps({"event": "sft_done",
                                    "adapter_dir": "/a"}) + "\n")
    assert parse_worker_backend(str(peft_log)) == {}   # additive: absent => {}
    assert parse_worker_backend(None) == {}


# ------------------------------------------------------------------- masking
def _tiny_chat_tokenizer(tmp_path):
    """Deterministic word-level chat tokenizer built HERE (no downloads)."""
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["[UNK]", "[PAD]", "[SYS]", "[/SYS]", "[USR]", "[/USR]",
                "[TOOL]", "[/TOOL]", "[ASST]", "[/ASST]"]
    words = specials + ["hello", "world", "be", "nice", "fine", "thanks",
                        "tell", "me", "more", "ok", "done", "a", "b", "c"]
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   pad_token="[PAD]")
    fast.chat_template = (
        "{% for m in messages %}"
        "{% if m.role == 'system' %}[SYS]{{ m.content }}[/SYS]"
        "{% elif m.role == 'user' %}[USR]{{ m.content }}[/USR]"
        "{% elif m.role == 'tool' %}[TOOL]{{ m.content }}[/TOOL]"
        "{% else %}[ASST]{{ m.content }}[/ASST]{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}[ASST]{% endif %}")
    out = tmp_path / "tok"
    fast.save_pretrained(str(out))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(out), use_fast=True)


def _trainable_strings(tok, enc):
    return [tok.decode([i]) for i, l in zip(enc["input_ids"], enc["labels"])
            if l != -100]


def test_mask_multi_turn_system_tool(tmp_path):
    pytest.importorskip("transformers")
    from sable.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "fine thanks"},
        {"role": "user", "content": "tell me more"},
        {"role": "assistant", "content": "ok done"},
    ]
    enc = mask_conversation(tok, msgs, 128)
    assert enc is not None
    assert len(enc["input_ids"]) == len(enc["labels"])
    trainable = _trainable_strings(tok, enc)
    # ONLY assistant content + the assistant end token are trainable; every
    # system/user/tool span and every header is -100.
    assert trainable == ["fine", "thanks", "[/ASST]", "ok", "done", "[/ASST]"]


def test_mask_consecutive_assistant_turns(tmp_path):
    pytest.importorskip("transformers")
    from sable.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "tool", "content": "c"},
        {"role": "assistant", "content": "fine thanks"},
    ]
    enc = mask_conversation(tok, msgs, 128)
    assert enc is not None
    trainable = _trainable_strings(tok, enc)
    # consecutive same-role assistant turns each keep their own span; the
    # tool turn stays fully masked.
    assert trainable == ["a", "[/ASST]", "b", "[/ASST]",
                         "fine", "thanks", "[/ASST]"]


def test_mask_stricter_than_airllm(tmp_path):
    """airllm's messages loader masks only up to the LAST assistant turn —
    everything after earlier assistant turns (the follow-up user/tool spans)
    would be TRAINED. Ours must keep those spans masked."""
    pytest.importorskip("transformers")
    from sable.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "fine thanks"},   # intermediate
        {"role": "user", "content": "tell me more"},       # airllm trains this
        {"role": "tool", "content": "be nice done"},       # ...and this
        {"role": "assistant", "content": "ok done"},
    ]
    enc = mask_conversation(tok, msgs, 128)
    assert enc is not None
    ids, labels = enc["input_ids"], enc["labels"]
    trainable = _trainable_strings(tok, enc)
    assert trainable == ["fine", "thanks", "[/ASST]", "ok", "done", "[/ASST]"]

    # The divergence concretely: 'tell' and 'me' sit between the two
    # assistant spans; under airllm's last-turn-only rule their labels would
    # be non-masked, here they must be -100.
    tell_ids = {tok.convert_tokens_to_ids(w) for w in ("tell", "me", "more")}
    for i, t in enumerate(ids):
        if t in tell_ids:
            assert labels[i] == -100


def test_mask_requires_assistant_turn(tmp_path):
    pytest.importorskip("transformers")
    from sable.training.streamed.masking import mask_conversation

    tok = _tiny_chat_tokenizer(tmp_path)
    enc = mask_conversation(tok, [{"role": "user", "content": "hello"}], 64)
    assert enc is None  # nothing trainable => caller skips the row


def test_sharegpt_role_mapping():
    from sable.training.streamed.masking import sharegpt_to_messages

    # Roles as emitted by curate/sharegpt.py, plus ShareGPT's raw
    # function_call; genuinely unknown roles are dropped.
    msgs = sharegpt_to_messages([
        {"from": "system", "value": "s"},
        {"from": "human", "value": "h"},
        {"from": "gpt", "value": "g"},
        {"from": "tool", "value": "t"},
        {"from": "function_call", "value": "f"},
        {"from": "nonsense", "value": "x"},
    ])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant",
                                         "tool", "assistant"]


# ------------------------------------------------------- streamed training e2e
def _build_tiny_llama_ckpt(root: Path):
    """Synthetic 2-layer llama checkpoint + tokenizer, written locally."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = _tiny_chat_tokenizer(root)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=len(tok), max_position_embeddings=128,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    ckpt = root / "base"
    model.save_pretrained(str(ckpt))      # safetensors shard(s) on disk
    tok.save_pretrained(str(ckpt))
    return ckpt, tok


@pytest.fixture(scope="module")
def streamed_run(tmp_path_factory):
    """Run the full streamed worker once; share artifacts across tests."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    pytest.importorskip("tokenizers")

    root = tmp_path_factory.mktemp("wp10_streamed")
    ckpt, tok = _build_tiny_llama_ckpt(root)

    rows = [{"conversations": [
        {"from": "system", "value": "be nice"},
        {"from": "human", "value": "hello world"},
        {"from": "gpt", "value": "fine thanks"},
        {"from": "human", "value": "tell me more"},
        {"from": "gpt", "value": "ok done"}]}] * 4
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

    from sable.training.worker import main as worker_main

    log = root / "job.log"
    rc = worker_main(["--job", "wp10", "--kind", "sft",
                      "--config", str(train_yaml), "--log-path", str(log)])
    return {"rc": rc, "root": root, "out": out,
            "log_lines": [json.loads(x) for x in
                          log.read_text().splitlines() if x.strip()]}


def test_streamed_e2e_worker_success(streamed_run):
    assert streamed_run["rc"] == 0
    events = [r["event"] for r in streamed_run["log_lines"]]
    assert "streamed_sft_done" in events and "done" in events
    done = [r for r in streamed_run["log_lines"]
            if r["event"] == "streamed_sft_done"][0]
    assert done["backend"] == "streamed"
    assert Path(done["adapter_dir"]).is_dir()


def test_streamed_e2e_splitting_progress_in_job_log(streamed_run):
    split = [r for r in streamed_run["log_lines"]
             if r.get("phase") == "splitting_checkpoint"]
    assert split, "splitting progress must stream to the job log"
    assert split[0]["message"].startswith("splitting checkpoint: shard ")
    assert split[-1]["shard"] == split[-1]["total"]


def test_streamed_e2e_metrics_csv_schema(streamed_run):
    csv_path = streamed_run["out"] / "train_metrics.csv"
    rows = csv_path.read_text().splitlines()
    assert rows[0] == "step,loss,peak_vram_gb,elapsed_s"
    # CPU-only box: peak stays an honest 0.0 (documented in the runner).
    progress = [r for r in streamed_run["log_lines"]
                if r["event"] == "progress"]
    assert len(rows) - 1 == len(progress)
    first_step, loss, peak, elapsed = rows[1].split(",")
    assert first_step == "1" and float(peak) == 0.0 and float(elapsed) >= 0


def test_streamed_e2e_loss_decreases(streamed_run):
    losses = [r["loss"] for r in streamed_run["log_lines"]
              if r["event"] == "progress"]
    assert len(losses) >= 6
    assert losses[-1] < losses[0] * 0.9          # clear downward trend
    assert losses[-1] < losses[-2] <= losses[0]  # not diverging


def test_streamed_e2e_adapter_artifacts(streamed_run):
    adapter = streamed_run["out"] / "adapter"
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert cfg["backend"] == "streamed"
    assert cfg["format"] == "sable-lora-state-dict"
    assert (adapter / "lora_state.safetensors").is_file()
    gguf = streamed_run["out"] / "lora.gguf"
    assert gguf.is_file()
    assert gguf.read_bytes()[:4] == b"GGUF"


# ------------------------------------------------------------------- artifact
def _parse_gguf(path: Path):
    """Minimal independent GGUF reader (stdlib only) for assertions."""
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    def s(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode()

    def v(f, t):
        if t == 8:
            return s(f)
        if t == 9:
            (at,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            return [v(f, at) for _ in range(n)]
        (x,) = struct.unpack(fmt[t], f.read(SZ[t]))
        return x

    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        (ver,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        kv = {}
        for _ in range(n_kv):
            k = s(f)
            (t,) = struct.unpack("<I", f.read(4))
            kv[k] = v(f, t)
        tensors = []
        for _ in range(n_tensors):
            name = s(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            (dt,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, dims, dt, off))
        size = path.stat().st_size
    return {"version": ver, "kv": kv, "tensors": tensors, "size": size}


def test_gguf_lora_artifact_structure(streamed_run):
    art = _parse_gguf(streamed_run["out"] / "lora.gguf")
    assert art["version"] == 3
    kv = art["kv"]
    assert kv["general.architecture"] == "llama"
    assert kv["general.type"] == "adapter"
    assert kv["adapter.type"] == "lora"
    assert isinstance(kv["adapter.lora.alpha"], float)

    names = [t[0] for t in art["tensors"]]
    # llama.cpp LoRA naming: blk.<i>.<role>.weight.lora_a/_b, complete pairs.
    assert "blk.0.attn_q.weight.lora_a" in names
    assert "blk.1.ffn_down.weight.lora_b" in names
    bases = {n.rsplit(".lora_", 1)[0] for n in names}
    for b in bases:
        assert f"{b}.lora_a" in names and f"{b}.lora_b" in names
    # ggml F32 type id (0) — NOT the GGUF value-type id — every tensor.
    assert all(t[2] == 0 for t in art["tensors"])
    # llama.cpp strictly validates offsets: aligned + non-overlapping and
    # inside the file.
    prev_end = 0
    for (_n, dims, _dt, off) in sorted(art["tensors"], key=lambda x: x[3]):
        assert off % 32 == 0 or off == 0
        assert off >= prev_end
        nbytes = 4
        for d in dims:
            nbytes *= d
        prev_end = off + nbytes
    assert prev_end <= art["size"]


def test_map_state_to_gguf_contract_rules():
    from sable.training.streamed.gguf_lora import map_state_to_gguf

    good_A = [[1.0, 2.0], [3.0, 4.0]]          # [r=2, in=2]
    good_B = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # [out=3, r=2]
    state = {
        "model.layers.0.self_attn.q_proj.lora_A": good_A,
        "model.layers.0.self_attn.q_proj.lora_B": good_B,
    }
    tensors, ranks = map_state_to_gguf(state)
    assert sorted(t.name for t in tensors) == [
        "blk.0.attn_q.weight.lora_a", "blk.0.attn_q.weight.lora_b"]
    assert ranks["attn_q"] == 2

    with pytest.raises(ValueError, match="half|partner"):
        map_state_to_gguf({"model.layers.0.self_attn.q_proj.lora_A": good_A})
    with pytest.raises(ValueError, match="contract"):
        map_state_to_gguf({"model.layers.0.norm.weight": good_A})
    with pytest.raises(ValueError, match="mapping"):
        map_state_to_gguf({"model.layers.9.self_attn.x_proj.lora_A": good_A,
                           "model.layers.9.self_attn.x_proj.lora_B": good_B})
    with pytest.raises(ValueError, match="empty"):
        map_state_to_gguf({})


def test_export_relocates_streamed_lora_without_llama_cpp(tmp_path):
    """The streamed GGUF-LoRA artifact flows through run_export's existing
    contract unchanged — and BEFORE the llama.cpp requirement check (no
    merge/quantize needed for an adapter-only artifact)."""
    from sable.training import worker

    out_dir = tmp_path / "out"
    adapter = out_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps(
        {"backend": "streamed", "format": "sable-lora-state-dict"}))
    payload = b"GGUF" + bytes(200)
    (out_dir / "lora.gguf").write_bytes(payload)

    gguf_out = tmp_path / "registered-model-q4_k_m.gguf"
    cfg = {"_adapter_dir": str(adapter), "_gguf_out": str(gguf_out),
           "training": {}}
    res = worker.run_export(cfg, str(tmp_path / "export.log"), False)
    assert res["method"] == "gguf-lora"
    assert Path(res["gguf_path"]).read_bytes() == payload

    # A peft adapter (no backend marker) must NOT take this branch: it hits
    # the pre-existing llama.cpp requirement error instead.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "adapter_config.json").write_text(json.dumps({"r": 8}))
    cfg2 = {"_adapter_dir": str(plain), "_gguf_out": str(tmp_path / "x.gguf"),
            "training": {}}
    with pytest.raises(RuntimeError):
        worker.run_export(cfg2, None, False)


def test_adapter_is_streamed():
    from sable.training.worker import adapter_is_streamed

    assert adapter_is_streamed(Path("/nonexistent")) is False
