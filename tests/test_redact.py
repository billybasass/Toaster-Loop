"""Backlog item #1 tests — secret redaction at record time + read blocking.

Three layers under test:
  1. core/redact.py golden patterns (every vendor family + idempotence +
     benign-text checks).
  2. store/recorder.py: the durable JSONL is redacted while the live
     in-memory conversation stays raw (full agent run through the fake
     server), plus the config escape hatch and its warning.
  3. core/tools.py: read_file/search_files refuse sensitive credential
     paths; `tools.block_sensitive_files: false` re-enables reads.

Plus the curater-level guarantee: a secret that passed the recorder never
appears in curated sft.jsonl output.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer  # noqa: E402

import sable.cli as cli  # noqa: E402
from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.config import load_config  # noqa: E402
from sable.core.redact import redact_messages, redact_text  # noqa: E402
from sable.core.tools import read_file, search_files  # noqa: E402
from sable.store import TraceRecorder  # noqa: E402

# --------------------------------------------------------------- golden set

# (family label, fixture containing the secret)
GOLDEN_SECRETS = [
    ("openai", "OPENAI key is sk-abcDEF456ghiJKL789mnoPQR"),
    ("openai-project", "use sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789 now"),
    ("anthropic", "anthropic: sk-ant-api03-ZZZZZzzzzXXXXyyyyYYYYzzzz1234"),
    ("github-pat", "ghp_AbCdEfGhIjKlMnOpQrStUvWx has repo scope"),
    ("github-oauth", "gho_AbCdEfGhIjKlMnOpQrStUvWx"),
    ("github-fine-grained",
     "github_pat_11ABCDEF0xYz9_abcdefghijklmnopqrstuvwx"),
    ("aws-akid", "AWS key AKIAIOSFODNN7EXAMPLE found"),
    ("aws-temp", "temporary ASIAIOSFODNN7EXAMPLE plus secret"),
    ("huggingface", "hf_AbCdEfGhIjKlMnOpQrStUvWx pushed the model"),
    ("slack", "slack token xoxb-1234567890abcdef-abcdefghij posted"),
    ("google", "AIzaSyA-bCdEfGhIjKlMnOpQrStUvWxYz012345 queried"),
    ("stripe-live-sk", "charge with sk_live_51Hxxxxxxxxxxxxxxxxxxxxxxabc"),
    ("stripe-live-pk", "form uses pk_live_51Hyyyyyyyyyyyyyyyyyyyyyydef"),
    ("gitlab", "glpat-abcdef123456789ABCDF revoked"),
    ("npm", "npm_abcdef1234567890ABCDEF published"),
    ("telegram",
     "bot 123456789:AAHabcdefGHIijklmnopQRSTuvwxYZ12345678 replied"),
    ("discord-webhook",
     "posted to https://discord.com/api/webhooks/"
     "123456789012345678/AbCdEfGhIjKlMnOpQrStUvWxYz123456"),
    ("jwt",
     "auth eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
     ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
     ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"),
    ("kv-env", 'export OPENAI_API_KEY="sk-not-matched-but-secret-value"'),
    ("kv-password", "login with password = hunter2secret here"),
    ("kv-authorization", "Authorization: Bearer abcdefghijklmnop1234"),
    ("json-body", '{"user": "u", "api_key": "opaque-token-value-xyz"}'),
]

PEM_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1P\n"
    "NB8KcKO9vyNcRy7\n"
    "-----END RSA PRIVATE KEY-----"
)


@pytest.mark.parametrize("family,fixture", GOLDEN_SECRETS,
                         ids=[f for f, _ in GOLDEN_SECRETS])
def test_golden_secrets_are_redacted(family, fixture):
    out = redact_text(fixture)
    assert "[REDACTED" in out, f"{family}: no masking applied to {fixture!r}"
    # the secret itself must be gone: strip prose and check token survival
    for tok in fixture.split():
        if len(tok) >= 12 and any(c.isdigit() for c in tok):
            assert tok not in out, f"{family}: token survived: {tok!r}"


def test_pem_block_fully_replaced():
    out = redact_text("here is the key:\n" + PEM_BLOCK + "\ndone")
    assert "[REDACTED_PRIVATE_KEY]" in out
    assert "MIIBOgIBAAJ" not in out
    assert "BEGIN RSA PRIVATE KEY" not in out


@pytest.mark.parametrize("family,fixture", GOLDEN_SECRETS + [("pem", PEM_BLOCK)],
                         ids=[f for f, _ in GOLDEN_SECRETS] + ["pem"])
def test_redaction_is_idempotent(family, fixture):
    once = redact_text(fixture)
    assert redact_text(once) == once


@pytest.mark.parametrize("benign", [
    "the task skipped step 3, disk usage on /dev/sda1 is fine",
    "git commit -m 'fix sk handling in the dispatcher'",
    "hex color #a1b2c3 and pointer id 0xdeadbeef",
    "def token():\n    return 42",
    "api_key = os.environ.get('OPENAI_API_KEY')",
    "tokenizer = 'cl100k_base'",
    "",
    "plain prose with no secrets whatsoever, skipping nothing.",
])
def test_benign_text_untouched(benign):
    assert redact_text(benign) == benign


def test_redact_messages_does_not_mutate_input():
    msgs = [
        {"role": "tool", "content": "secret sk-abcDEF456ghiJKL789mno here"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "write_file",
                "arguments": '{"path": "x", "content": "token: s3cr3tvalue-abc"}'}},
        ]},
    ]
    out = redact_messages(msgs)
    assert "sk-abcDEF456ghiJKL789mno" in msgs[0]["content"]  # raw kept
    assert "sk-abcDEF456ghiJKL789mno" not in out[0]["content"]
    assert "[REDACTED" in out[1]["tool_calls"][0]["function"]["arguments"]
    assert "s3cr3tvalue-abc" not in out[1]["tool_calls"][0]["function"]["arguments"]
    # original tool-call args untouched
    assert "s3cr3tvalue-abc" in msgs[1]["tool_calls"][0]["function"]["arguments"]


# ------------------------------------------------------------ recorder layer

class _SecretLeakingHandler(BaseHTTPRequestHandler):
    """First turn: read_file on a secret file. After tool result: echo it."""

    def log_message(self, format, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        tool_msg = next((m for m in messages if m.get("role") == "tool"), None)
        if tool_msg is None:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps(
                        {"path": self.server.secret_path})  # type: ignore[attr-defined]
                    }}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant",
                   "content": f"Answer with raw secret: {tool_msg['content'][:60]}"}
            finish = "stop"
        body = json.dumps({
            "id": "c", "object": "chat.completion", "model": "fake-model",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_response(404)
        self.end_headers()


SECRET_IN_FILE = "sk-TESTabcDEF456ghiJKL789mnoTOPSECRET"


@pytest.fixture()
def sable_home(tmp_path, monkeypatch):
    home = tmp_path / "uf-home"
    monkeypatch.setenv("SABLE_HOME", str(home))
    monkeypatch.delenv("SABLE_CONFIG", raising=False)
    return home


def test_recorded_jsonl_redacted_live_memory_raw(sable_home, tmp_path):
    """End-to-end: tool result contains a secret -> JSONL clean, memory raw."""
    secret_file = tmp_path / "notes.txt"
    secret_file.write_text(f"api value {SECRET_IN_FILE} end\n")

    srv = FakeOpenAIServer(handler_cls=_SecretLeakingHandler)
    srv._httpd.secret_path = str(secret_file)  # type: ignore[attr-defined]
    with srv:
        rec = TraceRecorder(home=sable_home)
        client = SableAgentClient(
            base_url=srv.base_url, model="fake-model", recorder=rec)
        result = client.run("read the secret file")
        assert result.completed

    # live conversation keeps the raw secret (agent must still function)
    joined_memory = json.dumps(result.messages, default=str)
    assert SECRET_IN_FILE in joined_memory

    # durable buffer contains no trace of it
    matches = list((sable_home / "buffer").rglob(f"{result.trace_id}.jsonl"))
    assert len(matches) == 1
    text = matches[0].read_text()
    assert SECRET_IN_FILE not in text
    assert "[REDACTED" in text

    # SQLite index summary too
    from sable.store.db import TraceStore
    store = TraceStore(sable_home / "buffer" / "sable.db")
    try:
        row = store.get_trace(result.trace_id or "")
        assert row is not None and SECRET_IN_FILE not in (row.task_summary or "")
    finally:
        store.close()


def test_recorder_redaction_escape_hatch_logs_warning(sable_home, caplog):
    class CfgOff:
        recording_redact_secrets = False

    with caplog.at_level("WARNING", logger="sable.store"):
        rec = TraceRecorder(cfg=CfgOff(), home=sable_home)
    assert any("redaction is DISABLED" in r.message for r in caplog.records)
    row = rec.record(task=f"log {SECRET_IN_FILE}", messages=[
        {"role": "user", "content": f"key {SECRET_IN_FILE}"},
        {"role": "assistant", "content": "ok"},
    ], backend="t", model_alias="m", base_url="", completed=True)
    assert row is not None
    text = Path(row.jsonl_path).read_text()
    # explicitly disabled => verbatim (documented escape-hatch behavior)
    assert SECRET_IN_FILE in text


def test_recorder_redaction_on_by_default(sable_home):
    rec = TraceRecorder(home=sable_home)  # no cfg
    assert rec.redact_secrets is True
    row = rec.record(task="t", messages=[
        {"role": "user", "content": f"key {SECRET_IN_FILE}"},
        {"role": "assistant", "content": "done"},
    ], backend="t", model_alias="m", base_url="", completed=True)
    assert row is not None and SECRET_IN_FILE not in Path(
        row.jsonl_path).read_text()


def test_config_yaml_recording_redact_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("SABLE_RECORDING", raising=False)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("recording:\n  redact_secrets: false\n")
    assert load_config(str(cfg_file)).recording_redact_secrets is False
    cfg_file2 = tmp_path / "cfg2.yaml"
    cfg_file2.write_text("tools:\n  block_sensitive_files: false\n")
    assert load_config(str(cfg_file2)).tools["block_sensitive_files"] is False


# ---------------------------------------------------------------- tools

def test_read_file_refuses_dotenv(sable_home, tmp_path):
    env = tmp_path / "project" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("SECRET=abc\n")
    out = read_file(str(env))
    assert out.startswith("error: blocked: sensitive credential path")
    assert "block_sensitive_files" in out


def test_read_file_refuses_ssh_key(sable_home, tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".ssh").mkdir(parents=True)
    (fake_home / ".ssh" / "id_rsa").write_text("PRIVATE\n")
    monkeypatch.setenv("HOME", str(fake_home))
    out = read_file("~/.ssh/id_rsa")
    assert out.startswith("error: blocked: sensitive credential path")


@pytest.mark.parametrize("rel", [
    ".aws/credentials", ".gnupg/pubring.kbx", "secrets/.netrc",
    "db/.pgpass", "app/credentials.json", ".env.production",
])
def test_read_file_refuses_credential_paths(sable_home, tmp_path, rel):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x\n")
    assert read_file(str(p)).startswith("error: blocked:")


def test_read_file_allows_normal_files(sable_home, tmp_path):
    good = tmp_path / "app.py"
    good.write_text("print('hi')\n")
    assert "print('hi')" in read_file(str(good))
    # public keys and .env.example are fine
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA...\n")
    assert "ssh-ed25519" in read_file(str(tmp_path / "id_ed25519.pub"))
    (tmp_path / ".env.example").write_text("PLACEHOLDER=1\n")
    assert "PLACEHOLDER" in read_file(str(tmp_path / ".env.example"))


def test_read_file_config_off_allows_reads(sable_home, tmp_path):
    env = tmp_path / ".env"
    env.write_text("SECRET=abc\n")
    out = read_file(str(env), block_sensitive=False)
    assert "SECRET=abc" in out


def test_search_files_skips_dotenv_contents(sable_home, tmp_path):
    (tmp_path / ".env").write_text("API_TOKEN=sk-abcdef123456789secret\n")
    (tmp_path / "app.py").write_text("API_TOKEN = 'from-app-py'\n")
    out = search_files("API_TOKEN", path=str(tmp_path))
    assert "app.py" in out
    assert ".env:" not in out
    assert "sk-abcdef123456789secret" not in out


def test_search_files_name_mode_skips_blocked(sable_home, tmp_path):
    (tmp_path / ".netrc").write_text("machine x login y password z\n")
    (tmp_path / "readme.md").write_text("hi\n")
    out = search_files(".netrc", path=str(tmp_path), mode="name")
    # blocked files never appear as result paths (the no-match notice echoing
    # the pattern is fine)
    assert str(tmp_path / ".netrc") not in out
    assert "no files matching" in out


def test_search_files_config_off_scans_dotenv(sable_home, tmp_path):
    (tmp_path / ".env").write_text("API_TOKEN=sk-abcdef123456789secret\n")
    out = search_files("API_TOKEN", path=str(tmp_path), block_sensitive=False)
    assert ".env" in out


def test_create_runner_threads_block_flag(tmp_path, monkeypatch):
    """tools.block_sensitive_files reaches the bound tool impls."""
    from sable.core.agent_runner import create_runner
    from sable.core.config import SableConfig

    monkeypatch.setenv("SABLE_HOME", str(tmp_path))
    cfg = SableConfig(tools={"block_sensitive_files": False})
    client = create_runner(cfg, base_url="http://127.0.0.1:1/v1", model="m")
    assert client.block_sensitive_files is False
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    assert "A=1" in client.tool_impls["read_file"](str(env))

    cfg2 = SableConfig()  # default on
    client2 = create_runner(cfg2, base_url="http://127.0.0.1:1/v1", model="m")
    assert client2.tool_impls["read_file"](str(env)).startswith("error: blocked:")


# ------------------------------------------------- curater-level guarantee

def test_secret_never_reaches_curated_sft(sable_home):
    """A secret recorded through the recorder never appears in sft.jsonl.

    Extends the curate test family (test_curate.py) with the backlog #1
    guarantee: record-time redaction is sufficient because curation reads
    only the durable JSONL.
    """
    rec = TraceRecorder(home=sable_home)
    row = rec.record(
        task=f"look up key {SECRET_IN_FILE}",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"what is the key {SECRET_IN_FILE}?"},
            {"role": "assistant", "content": "I looked it up for you."},
        ],
        backend="sable", model_alias="test-model", base_url="",
        completed=True,
    )
    assert row is not None
    assert SECRET_IN_FILE not in Path(row.jsonl_path).read_text()

    assert cli.main(["curate", "--name", "secrecy"]) == 0
    ds_dirs = list((sable_home / "datasets").rglob("sft.jsonl"))
    assert len(ds_dirs) == 1
    sft = ds_dirs[0].read_text()
    assert SECRET_IN_FILE not in sft
    assert "[REDACTED" in sft  # the trace made it in, masked
