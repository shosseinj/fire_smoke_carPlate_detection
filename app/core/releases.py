from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.api.project_info_schemas import ProjectInfoResponse, ReleaseInfo

RELEASE_DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "releases.json"


class ReleaseDataError(RuntimeError):
    ...


@lru_cache(maxsize=1)
def get_project_info() -> ProjectInfoResponse:
    try:
        raw_data = json.loads(RELEASE_DATA_FILE.read_text(encoding="utf-8"))
        return ProjectInfoResponse.model_validate(raw_data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ReleaseDataError(f"Unable to load release data from {RELEASE_DATA_FILE}: {exc}") from exc


def get_current_release() -> ReleaseInfo:
    project_info = get_project_info()
    return next(
        release
        for release in project_info.releases
        if release.version == project_info.project.current_version
    )


def get_release(version: str) -> ReleaseInfo | None:
    return next(
        (release for release in get_project_info().releases if release.version == version),
        None,
    )