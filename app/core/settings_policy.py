from __future__ import annotations

from typing import Any


SETTING_FIELDS: tuple[str, ...] = (
    "enable_processing", "process_fire", "process_plate",
    "counts_for_attendance", "margin_level", "draw_box",
    "draw_face", "draw_skeleton", "draw_zones",
    "face_rec_score", "face_det_score", "human_det_score",
    "confirmation_threshold",
)


def resolve_camera_setting(
    camera_value: Any,
    general_value: Any,
    *,
    force: bool = False,
) -> Any:
    if force:
        return general_value
    if camera_value is None:
        return general_value
    return camera_value


def resolve_all_camera_settings(
    camera_settings: dict[str, Any],
    general_settings: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in SETTING_FIELDS:
        result[field] = resolve_camera_setting(
            camera_settings.get(field),
            general_settings.get(field),
            force=force,
        )
    if result.get("confirmation_threshold") is not None and result.get("face_rec_score") is not None:
        if result["confirmation_threshold"] < result["face_rec_score"]:
            result["confirmation_threshold"] = result["face_rec_score"]
    return result
