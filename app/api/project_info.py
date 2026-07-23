from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.project_info_schemas import CurrentReleaseResponse, ProjectInfoResponse, ReleaseInfo
from app.core.releases import get_current_release, get_project_info, get_release

router = APIRouter(prefix="/api/v1/project-info", tags=["Project Information"])


@router.get(
    "",
    response_model=ProjectInfoResponse,
    summary="Get complete project information",
)
def read_project_info() -> ProjectInfoResponse:
    return get_project_info()


@router.get(
    "/current",
    response_model=CurrentReleaseResponse,
    summary="Get the current project release",
)
def read_current_release() -> CurrentReleaseResponse:
    project_info = get_project_info()
    return CurrentReleaseResponse(
        project=project_info.project,
        release=get_current_release(),
    )


@router.get(
    "/releases",
    response_model=list[ReleaseInfo],
    summary="List project releases",
)
def list_releases() -> list[ReleaseInfo]:
    return get_project_info().releases


@router.get(
    "/releases/{version}",
    response_model=ReleaseInfo,
    summary="Get one project release",
)
def read_release(version: str) -> ReleaseInfo:
    release = get_release(version)
    if release is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project release {version!r} was not found.",
        )
    return release