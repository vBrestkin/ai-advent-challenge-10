import io
import json
import re
from pathlib import Path

import httpx
import pytest
from pytest_bdd import given, when, then, scenarios, parsers

from day1.cli import main

scenarios("features/cli.feature")
SECRET = "test-secret-do-not-log"


@pytest.fixture
def state(tmp_path):
    return dict(root=tmp_path, env={"POLZA_API_KEY": SECRET, "LLM_MODEL": "test/model"},
                args=["Привет, мир!"], calls=[], kind="ok", out=io.StringIO(), err=io.StringIO())


@given("API готов вернуть ответ")
def ready(state):
    pass


@given(parsers.parse('API отвечает сбоем "{kind}"'))
def failure(state, kind):
    state["kind"] = kind


@given(parsers.parse('настройки содержат ошибку "{kind}"'))
def bad_config(state, kind):
    if kind == "missing_key":
        state["env"].pop("POLZA_API_KEY")
    elif kind == "missing_model":
        state["env"].pop("LLM_MODEL")
    else:
        state["args"] = {"empty_prompt": ["  "], "missing_prompt": [],
                         "unknown_argument": ["--unknown"],
                         "invalid_limit": ["hello", "--max-tokens", "0"],
                         "invalid_timeout": ["hello", "--read-timeout", "-1"]}[kind]


@given("запрос прерывается пользователем")
def interrupted(state):
    state["kind"] = "interrupt"


def handler(state, request):
    state["calls"].append(request)
    kind = state["kind"]
    errors = {"dns": httpx.ConnectError, "refused": httpx.ConnectError,
              "tls": httpx.ConnectError, "reset": httpx.ReadError,
              "connect_timeout": httpx.ConnectTimeout, "read_timeout": httpx.ReadTimeout,
              "write_timeout": httpx.WriteTimeout, "pool_timeout": httpx.PoolTimeout,
              "protocol": httpx.RemoteProtocolError}
    if kind in errors:
        raise errors[kind](f"{kind}: {SECRET}", request=request)
    if kind == "interrupt":
        raise KeyboardInterrupt
    if kind == "unexpected":
        raise RuntimeError(SECRET)
    if kind.isdigit():
        return httpx.Response(int(kind), json={"error": {"message": SECRET}})
    if kind == "invalid_json":
        return httpx.Response(200, text="broken " + SECRET)
    if kind == "invalid_shape":
        return httpx.Response(200, json=[])
    if kind == "empty_choices":
        return httpx.Response(200, json={"choices": []})
    body = {"choices": [{"finish_reason": "stop", "message": {"content": "Здравствуйте!"}}]}
    choice = body["choices"][0]
    if kind == "empty_content":
        choice["message"]["content"] = None
    if kind == "refusal":
        choice["finish_reason"] = "content_filter"
    if kind == "truncated":
        choice["finish_reason"] = "length"
    if kind == "provider_error":
        choice["finish_reason"] = "error"
    return httpx.Response(200, json=body)


@when("пользователь запускает CLI с запросом")
def run(state):
    with httpx.Client(transport=httpx.MockTransport(lambda r: handler(state, r))) as client:
        state["code"] = main(state["args"], env=state["env"], log_dir=state["root"],
                             client=client, stdout=state["out"], stderr=state["err"])


def log_text(state):
    files = list(state["root"].glob("*.log"))
    assert len(files) == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-f0-9]+\.log", files[0].name)
    text = files[0].read_text(encoding="utf-8")
    assert SECRET not in text + state["out"].getvalue() + state["err"].getvalue()
    return text


@then("ответ появляется в консоли и журнале")
def success(state):
    assert state["code"] == 0
    assert state["out"].getvalue() == "Здравствуйте!\n"
    text = log_text(state)
    assert "Привет, мир!" in text and "Здравствуйте!" in text
    assert len(state["calls"]) == 1
    request = state["calls"][0]
    assert str(request.url) == "https://polza.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer " + SECRET
    payload = json.loads(request.content)
    assert payload["model"] == "test/model"
    assert payload["max_tokens"] == 4096
    assert payload["messages"] == [{"role": "user", "content": "Привет, мир!"}]
    assert payload["stream"] is False
    assert request.extensions["timeout"]["connect"] == 10
    assert request.extensions["timeout"]["read"] == 60


@then("программа сообщает об ошибке в консоли и журнале без повторного запроса")
def failed(state):
    assert state["code"] == 1
    assert "Ошибка" in state["err"].getvalue()
    assert "Ошибка" in log_text(state)
    assert len(state["calls"]) == 1
    assert state["out"].getvalue() == ""


@then("API не вызывается, а ошибка появляется в консоли и журнале")
def invalid(state):
    assert state["code"] == 2
    assert not state["calls"]
    assert "Ошибка" in state["err"].getvalue()
    assert "Ошибка" in log_text(state)


@then("программа завершается с кодом 130 и записывает прерывание")
def cancelled(state):
    assert state["code"] == 130
    assert "Прервано" in state["err"].getvalue()
    assert "Прервано" in log_text(state)


def test_each_run_gets_own_log(state):
    run(state)
    run(state)
    assert len(list(state["root"].glob("*.log"))) == 2


def test_no_api_call_when_log_cannot_be_created(state):
    path = state["root"] / "file"
    path.write_text("occupied")
    state["root"] = path
    run(state)
    assert state["code"] == 1
    assert not state["calls"]
    assert "журнал" in state["err"].getvalue()


def test_secret_in_prompt_and_response_is_redacted(state):
    state["args"] = ["Ключ " + SECRET]
    run(state)
    assert SECRET not in log_text(state)


def test_help_is_logged_without_api_call(state):
    state["args"] = ["--help"]
    run(state)
    assert state["code"] == 0
    assert not state["calls"]
    assert "usage:" in state["out"].getvalue()
    assert "usage:" in log_text(state)


def test_failed_response_body_is_logged_and_redacted(state):
    state["kind"] = "invalid_json"
    run(state)
    assert "broken [REDACTED]" in log_text(state)


@pytest.mark.parametrize("body", [None, {}, {"choices": None}, {"choices": [{}]},
    {"choices": [{"finish_reason": "stop", "message": "wrong"}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": ["wrong"]}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "", "refusal": "no"}}]}])
def test_malformed_response_never_succeeds(state, body):
    def respond(request):
        state["calls"].append(request)
        return httpx.Response(200, content=json.dumps(body))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        code = main(state["args"], env=state["env"], log_dir=state["root"],
                    client=client, stdout=state["out"], stderr=state["err"])
    assert code == 1
    assert "Ошибка" in log_text(state)
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("args", [["hi", "--read-timeout", "nan"],
    ["hi", "--connect-timeout", "inf"], ["hi", "--max-tokens", "abc"]])
def test_invalid_numeric_settings(state, args):
    state["args"] = args
    run(state)
    invalid(state)


def test_dotenv_loaded_and_environment_overrides_it(state, monkeypatch):
    import day1.cli as cli
    (state["root"] / ".env").write_text("POLZA_API_KEY=from-file\nLLM_MODEL=from-file\n")
    monkeypatch.setattr(cli, "PROJECT_DIR", state["root"])
    monkeypatch.setenv("POLZA_API_KEY", SECRET)
    monkeypatch.setenv("LLM_MODEL", "from-environment")
    with httpx.Client(transport=httpx.MockTransport(lambda r: handler(state, r))) as client:
        assert main(["hi"], log_dir=state["root"], client=client,
                    stdout=state["out"], stderr=state["err"]) == 0
    assert json.loads(state["calls"][0].content)["model"] == "from-environment"
    assert state["calls"][0].headers["authorization"] == "Bearer " + SECRET


def test_module_entrypoint_logs_help(tmp_path):
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "day1", "--help"],
        env={**os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert not result.stderr
