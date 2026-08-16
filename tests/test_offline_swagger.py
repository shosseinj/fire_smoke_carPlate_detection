from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
MAIN_SOURCE = PROJECT_ROOT / "app" / "main.py"
SWAGGER_ASSETS = PROJECT_ROOT / "app" / "web" / "swagger-ui"


def test_swagger_uses_only_local_assets() -> None:
    source = MAIN_SOURCE.read_text(encoding="utf-8")

    assert "docs_url=None" in source
    assert 'swagger_js_url="/docs-assets/swagger-ui-bundle.js"' in source
    assert 'swagger_css_url="/docs-assets/swagger-ui.css"' in source
    assert 'swagger_favicon_url="/docs-assets/favicon.svg"' in source
    assert "cdn.jsdelivr.net" not in source
    assert "fastapi.tiangolo.com/img/favicon.png" not in source


def test_vendored_swagger_assets_are_present() -> None:
    expected_minimum_sizes = {
        "swagger-ui-bundle.js": 1_000_000,
        "swagger-ui.css": 100_000,
        "favicon.svg": 100,
        "LICENSE": 1_000,
    }

    for filename, minimum_size in expected_minimum_sizes.items():
        asset = SWAGGER_ASSETS / filename
        assert asset.is_file()
        assert asset.stat().st_size >= minimum_size

import pytest

pytestmark = pytest.mark.unit
