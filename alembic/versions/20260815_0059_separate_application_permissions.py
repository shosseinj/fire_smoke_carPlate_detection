"""Replace broad application grants with independent application permissions.

Revision ID: 20260815_0059
Revises: 20260815_0058
"""

from alembic import op
import sqlalchemy as sa

revision = "20260815_0059"
down_revision = "20260815_0058"
branch_labels = None
depends_on = None

READ_APPS = (
    "buildings", "sections", "cameras", "rooms", "sources", "personnel",
    "personnel_images", "employee_types", "shifts", "holidays", "requests",
    "personnel_requests", "car_plates", "plate_settings", "plate_logs",
    "fire_smoke", "fire_logs", "detection_logs", "humans", "faces",
    "recordings", "static_videos", "models", "general_settings", "diagnostics",
    "processor_tests", "broadcast", "video_wall", "results", "broadcast_gpu",
    "import_progress", "developer",
)

WRITE_GRANTS = {
    "rooms": ("create", "edit", "delete", "assign"),
    "sources": ("create", "edit", "delete", "enable", "assign"),
    "personnel": ("create", "edit", "delete", "import", "export"),
    "personnel_images": ("create", "delete"),
    "employee_types": ("create", "edit", "delete"),
    "shifts": ("create", "edit", "delete", "assign"),
    "holidays": ("create", "edit", "delete", "import", "export"),
    "requests": ("create", "delete", "approve", "cancel"),
    "personnel_requests": ("create", "edit", "delete", "approve", "reject", "generate"),
    "car_plates": ("create", "edit", "delete"),
    "plate_settings": ("edit", "delete"),
    "plate_logs": ("create", "edit", "delete", "download"),
    "fire_smoke": ("configure",), "fire_logs": ("create", "edit", "delete", "download"),
    "detection_logs": ("create", "edit", "delete", "import", "export", "download"),
    "faces": ("configure", "enroll", "delete"),
    "recordings": ("create", "cancel", "download", "delete"),
    "static_videos": ("create", "edit", "delete", "retry"),
    "models": ("configure", "upload", "convert"),
    "general_settings": ("edit", "reset"), "diagnostics": ("execute",),
    "processor_tests": ("execute",), "frames": ("execute",),
    "extract_frames": ("execute",), "broadcast": ("control",),
    "import_progress": ("create", "delete"), "live_branch": ("execute",),
    "developer": ("execute",),
}

def _copy(source_actions: tuple[str, ...], application: str, action: str) -> None:
    op.get_bind().execute(sa.text("""
        INSERT INTO user_permission_grants
            (user_id, application, action, scope_type, scope_id, assigned_by)
        SELECT DISTINCT user_id, :application, :action, 'global', 0, assigned_by
        FROM user_permission_grants
        WHERE application = 'application' AND action = ANY(:source_actions)
        ON CONFLICT DO NOTHING
    """), {"application": application, "action": action, "source_actions": list(source_actions)})

def upgrade() -> None:
    op.drop_constraint("ck_user_permission_grants_target", "user_permission_grants", type_="check")
    op.create_check_constraint(
        "ck_user_permission_grants_target", "user_permission_grants",
        "(scope_type = 'global' AND scope_id = 0) OR "
        "(scope_type IN ('building', 'section', 'camera', 'room') AND scope_id > 0)",
    )
    for app in READ_APPS:
        _copy(("read", "manage", "system"), app, "read")
    for app, actions in WRITE_GRANTS.items():
        for action in actions:
            _copy(("manage", "system"), app, action)
    op.execute("DELETE FROM user_permission_grants WHERE application = 'application'")

def downgrade() -> None:
    for action in ("read", "manage", "system"):
        op.execute(sa.text("""
            INSERT INTO user_permission_grants (user_id, application, action, scope_type, scope_id)
            SELECT DISTINCT user_id, 'application', :action, 'global', 0
            FROM user_permission_grants ON CONFLICT DO NOTHING
        """).bindparams(action=action))
    op.execute("DELETE FROM user_permission_grants WHERE scope_type = 'room'")
    op.drop_constraint("ck_user_permission_grants_target", "user_permission_grants", type_="check")
    op.create_check_constraint(
        "ck_user_permission_grants_target", "user_permission_grants",
        "(scope_type = 'global' AND scope_id = 0) OR "
        "(scope_type IN ('building', 'section', 'camera') AND scope_id > 0)",
    )
