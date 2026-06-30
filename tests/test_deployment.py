from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_is_non_root_and_has_healthcheck():
    dockerfile = _read("Dockerfile")

    assert "FROM python:3.11-slim" in dockerfile
    assert "USER careerpilot" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "127.0.0.1:8001/health" in dockerfile
    assert 'CMD ["python", "-m", "uvicorn"' in dockerfile


def test_dockerignore_excludes_secrets_and_private_files():
    dockerignore = _read(".dockerignore")

    assert ".env" in dockerignore
    assert "data/resume" in dockerignore
    assert "*.pdf" in dockerignore
    assert "*.zip" in dockerignore
    assert "!.env.example" in dockerignore


def test_compose_exposes_service_and_does_not_hardcode_keys():
    compose = _read("docker-compose.yml")

    assert "careerpilot:" in compose
    assert "${CAREERPILOT_PORT:-8001}:8001" in compose
    assert 'DEEPSEEK_API_KEY: "${DEEPSEEK_API_KEY:-}"' in compose
    assert 'DASHSCOPE_API_KEY: "${DASHSCOPE_API_KEY:-}"' in compose
    assert "your_deepseek_api_key" not in compose
    assert "sk-" not in compose
    assert "healthcheck:" in compose


def test_ci_covers_python39_tests_and_docker_smoke_test():
    workflow = _read(".github/workflows/ci.yml")

    assert 'python-version: ["3.9", "3.11"]' in workflow
    assert "actions/checkout@v4" in workflow
    assert "actions/setup-python@v5" in workflow
    assert "pytest -q" in workflow
    assert "docker build --tag careerpilot-langgraph:ci ." in workflow
    assert "http://127.0.0.1:8001/health" in workflow
