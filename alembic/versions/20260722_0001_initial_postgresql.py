"""Initial PostgreSQL schema.

Revision ID: 20260722_0001
Revises: None
Create Date: 2026-07-22
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '20260722_0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text('CREATE TABLE buildings (\n\tid SERIAL NOT NULL, \n\tname TEXT NOT NULL, \n\taddress TEXT, \n\tdescription TEXT, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id)\n)'))
    op.execute(sa.text('CREATE INDEX idx_buildings_name ON buildings (name)'))
    op.execute(sa.text("CREATE TABLE cameras (\n\tcamera_id TEXT NOT NULL, \n\tname TEXT NOT NULL, \n\tenabled INTEGER DEFAULT '1' NOT NULL, \n\ttasks_json TEXT DEFAULT '[]' NOT NULL, \n\tsource_uri TEXT, \n\tframe_width INTEGER DEFAULT '640' NOT NULL, \n\tframe_height INTEGER DEFAULT '640' NOT NULL, \n\tsection_id INTEGER, \n\tmetadata_json TEXT DEFAULT '{}' NOT NULL, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (camera_id), \n\tCONSTRAINT ck_cameras_enabled CHECK (enabled IN (0, 1))\n)"))
    op.execute(sa.text('CREATE INDEX idx_cameras_enabled ON cameras (enabled)'))
    op.execute(sa.text('CREATE TABLE face_embeddings (\n\tid TEXT NOT NULL, \n\tperson TEXT NOT NULL, \n\tref_img_id TEXT, \n\tembedding BYTEA NOT NULL, \n\tdimension INTEGER NOT NULL, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id)\n)'))
    op.execute(sa.text('CREATE INDEX idx_face_embeddings_person ON face_embeddings (person)'))
    op.execute(sa.text('CREATE INDEX idx_face_embeddings_ref_img ON face_embeddings (ref_img_id)'))
    op.execute(sa.text('CREATE TABLE face_quality_settings (\n\tsingleton SERIAL NOT NULL, \n\tquality_threshold FLOAT NOT NULL, \n\tblur_threshold FLOAT NOT NULL, \n\tmin_face_width INTEGER NOT NULL, \n\tmin_face_height INTEGER NOT NULL, \n\tmin_eye_distance FLOAT NOT NULL, \n\tmax_abs_yaw FLOAT NOT NULL, \n\tmax_abs_pitch FLOAT NOT NULL, \n\tmax_abs_roll FLOAT NOT NULL, \n\trequire_landmarks INTEGER NOT NULL, \n\tPRIMARY KEY (singleton), \n\tCONSTRAINT ck_face_quality_singleton CHECK (singleton = 1)\n)'))
    op.execute(sa.text("CREATE TABLE fire_smoke_logs (\n\tid SERIAL NOT NULL, \n\tcamera TEXT NOT NULL, \n\ttime TIMESTAMP WITH TIME ZONE NOT NULL, \n\tincident_id TEXT, \n\tseverity TEXT NOT NULL, \n\tfire_count INTEGER NOT NULL, \n\tsmoke_count INTEGER NOT NULL, \n\tfire_confidence FLOAT NOT NULL, \n\tsmoke_confidence FLOAT NOT NULL, \n\twindow_seconds FLOAT NOT NULL, \n\tsnapshot_url TEXT DEFAULT '' NOT NULL, \n\tdetails_json TEXT DEFAULT '{}' NOT NULL, \n\tPRIMARY KEY (id)\n)"))
    op.execute(sa.text('CREATE INDEX idx_fire_smoke_logs_camera ON fire_smoke_logs (camera)'))
    op.execute(sa.text('CREATE INDEX idx_fire_smoke_logs_incident ON fire_smoke_logs (incident_id)'))
    op.execute(sa.text('CREATE INDEX idx_fire_smoke_logs_severity ON fire_smoke_logs (severity)'))
    op.execute(sa.text('CREATE INDEX idx_fire_smoke_logs_time ON fire_smoke_logs (time)'))
    op.execute(sa.text('CREATE TABLE fire_smoke_settings (\n\tid SERIAL NOT NULL, \n\twindow_seconds FLOAT NOT NULL, \n\tlow_count INTEGER NOT NULL, \n\tmedium_count INTEGER NOT NULL, \n\thigh_count INTEGER NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT ck_fire_smoke_settings_singleton CHECK (id = 1)\n)'))
    op.execute(sa.text("CREATE TABLE holidays (\n\tid SERIAL NOT NULL, \n\tname TEXT NOT NULL, \n\tdate_value DATE NOT NULL, \n\tdescription TEXT, \n\tholiday_type TEXT DEFAULT 'national' NOT NULL, \n\tevery_year INTEGER DEFAULT '0' NOT NULL, \n\tis_active INTEGER DEFAULT '1' NOT NULL, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id)\n)"))
    op.execute(sa.text('CREATE INDEX idx_holidays_active ON holidays (is_active)'))
    op.execute(sa.text('CREATE INDEX idx_holidays_date ON holidays (date_value)'))
    op.execute(sa.text("CREATE TABLE human_logs (\n\tid SERIAL NOT NULL, \n\tsession_id TEXT NOT NULL, \n\tcamera TEXT NOT NULL, \n\ttrack_id INTEGER NOT NULL, \n\tname TEXT DEFAULT 'Unknown' NOT NULL, \n\tfirst_seen TIMESTAMP WITH TIME ZONE NOT NULL, \n\tlast_seen TIMESTAMP WITH TIME ZONE NOT NULL, \n\trecognition_score FLOAT DEFAULT '0' NOT NULL, \n\tref_img_id TEXT, \n\tsnapshot_url TEXT DEFAULT '' NOT NULL, \n\tvideo_url TEXT DEFAULT '' NOT NULL, \n\tface_video_url TEXT DEFAULT '' NOT NULL, \n\tsnapshot_quality FLOAT DEFAULT '0' NOT NULL, \n\tbest_face_quality FLOAT DEFAULT '0' NOT NULL, \n\tfull_frame_video_frames INTEGER DEFAULT '0' NOT NULL, \n\taccepted_face_frames INTEGER DEFAULT '0' NOT NULL, \n\tpersonnel_id INTEGER, \n\tcounts_for_attendance INTEGER DEFAULT '1' NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT uq_human_session_camera_track UNIQUE (session_id, camera, track_id)\n)"))
    op.execute(sa.text('CREATE INDEX idx_human_logs_camera ON human_logs (camera)'))
    op.execute(sa.text('CREATE INDEX idx_human_logs_last_seen ON human_logs (last_seen)'))
    op.execute(sa.text('CREATE INDEX idx_human_logs_name ON human_logs (name)'))
    op.execute(sa.text('CREATE TABLE model_conversion_jobs (\n\tjob_id TEXT NOT NULL, \n\trole TEXT NOT NULL, \n\tsource_model TEXT NOT NULL, \n\toutput_directory TEXT NOT NULL, \n\tstatus TEXT NOT NULL, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\toptions_json TEXT NOT NULL, \n\tartifacts_json TEXT NOT NULL, \n\terrors_json TEXT NOT NULL, \n\tPRIMARY KEY (job_id)\n)'))
    op.execute(sa.text("CREATE TABLE model_general_settings (\n\tid SERIAL NOT NULL, \n\tfire_smoke_model TEXT NOT NULL, \n\tvehicle_detector_model TEXT NOT NULL, \n\tplate_detector_model TEXT NOT NULL, \n\tpreferred_format TEXT NOT NULL, \n\tallow_onnx_fallback INTEGER NOT NULL, \n\tallow_pt_fallback INTEGER NOT NULL, \n\texport_imgsz INTEGER NOT NULL, \n\texport_batch_size INTEGER NOT NULL, \n\texport_workspace_gb FLOAT NOT NULL, \n\texport_half INTEGER NOT NULL, \n\texport_dynamic INTEGER NOT NULL, \n\texport_timeout_seconds INTEGER DEFAULT '300' NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT ck_model_general_singleton CHECK (id = 1)\n)"))
    op.execute(sa.text("CREATE TABLE personnel (\n\tid SERIAL NOT NULL, \n\tfname TEXT NOT NULL, \n\tlname TEXT NOT NULL, \n\tnational_code TEXT NOT NULL, \n\temployee_type TEXT DEFAULT 'unknown' NOT NULL, \n\tdegree TEXT, \n\tshift_id INTEGER, \n\tlast_seen TIMESTAMP WITH TIME ZONE, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (national_code)\n)"))
    op.execute(sa.text('CREATE INDEX idx_personnel_name ON personnel (lname, fname)'))
    op.execute(sa.text('CREATE INDEX idx_personnel_national_code ON personnel (national_code)'))
    op.execute(sa.text('CREATE INDEX idx_personnel_shift ON personnel (shift_id)'))
    op.execute(sa.text('CREATE TABLE plate_camera_settings (\n\tcamera_id TEXT NOT NULL, \n\tvehicle_confidence FLOAT, \n\tplate_confidence FLOAT, \n\tocr_confidence FLOAT, \n\tmin_vehicle_width_pixels INTEGER, \n\tmin_vehicle_height_pixels INTEGER, \n\tmin_vehicle_area_ratio FLOAT, \n\tvehicle_crop_padding_ratio FLOAT, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (camera_id)\n)'))
    op.execute(sa.text('CREATE TABLE plate_general_settings (\n\tid SERIAL NOT NULL, \n\tvehicle_confidence FLOAT NOT NULL, \n\tplate_confidence FLOAT NOT NULL, \n\tocr_confidence FLOAT NOT NULL, \n\tmin_vehicle_width_pixels INTEGER NOT NULL, \n\tmin_vehicle_height_pixels INTEGER NOT NULL, \n\tmin_vehicle_area_ratio FLOAT NOT NULL, \n\tvehicle_crop_padding_ratio FLOAT NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT ck_plate_general_singleton CHECK (id = 1)\n)'))
    op.execute(sa.text("CREATE TABLE plate_logs (\n\tid SERIAL NOT NULL, \n\tcamera TEXT NOT NULL, \n\ttime TIMESTAMP WITH TIME ZONE NOT NULL, \n\tplate TEXT NOT NULL, \n\tsnapshot_url TEXT DEFAULT '' NOT NULL, \n\tdetails_json TEXT DEFAULT '{}' NOT NULL, \n\tPRIMARY KEY (id)\n)"))
    op.execute(sa.text('CREATE INDEX idx_plate_logs_camera ON plate_logs (camera)'))
    op.execute(sa.text('CREATE INDEX idx_plate_logs_plate ON plate_logs (plate)'))
    op.execute(sa.text('CREATE INDEX idx_plate_logs_time ON plate_logs (time)'))
    op.execute(sa.text('CREATE TABLE revoked_tokens (\n\tjti TEXT NOT NULL, \n\trevoked_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\texpires_at_utc TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (jti)\n)'))
    op.execute(sa.text('CREATE INDEX idx_revoked_expires ON revoked_tokens (expires_at_utc)'))
    op.execute(sa.text("CREATE TABLE users (\n\tid SERIAL NOT NULL, \n\tusername TEXT NOT NULL, \n\tpassword_hash TEXT NOT NULL, \n\trole TEXT DEFAULT 'viewer' NOT NULL, \n\tis_active INTEGER DEFAULT '1' NOT NULL, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\temail TEXT, \n\tfull_name TEXT, \n\tlast_login_utc TIMESTAMP WITH TIME ZONE, \n\tlogin_attempts INTEGER DEFAULT '0' NOT NULL, \n\tlocked_until_utc TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (id), \n\tUNIQUE (username)\n)"))
    op.execute(sa.text('CREATE INDEX idx_users_email ON users (email)'))
    op.execute(sa.text('CREATE INDEX idx_users_username ON users (username)'))
    op.execute(sa.text("CREATE TABLE work_shifts (\n\tid SERIAL NOT NULL, \n\tshift_name TEXT NOT NULL, \n\tshift_type TEXT DEFAULT 'morning' NOT NULL, \n\tstart_time TIME WITHOUT TIME ZONE DEFAULT TIME '08:00' NOT NULL, \n\tend_time TIME WITHOUT TIME ZONE DEFAULT TIME '16:00' NOT NULL, \n\ttimezone_name VARCHAR(64) DEFAULT 'Asia/Tehran' NOT NULL, \n\tmax_minutes_delay INTEGER DEFAULT '15' NOT NULL, \n\tmax_minutes_early INTEGER DEFAULT '15' NOT NULL, \n\tmax_overtime_hours FLOAT DEFAULT '2.0' NOT NULL, \n\tworks_saturday INTEGER DEFAULT '0' NOT NULL, \n\tworks_sunday INTEGER DEFAULT '0' NOT NULL, \n\tworks_monday INTEGER DEFAULT '1' NOT NULL, \n\tworks_tuesday INTEGER DEFAULT '1' NOT NULL, \n\tworks_wednesday INTEGER DEFAULT '1' NOT NULL, \n\tworks_thursday INTEGER DEFAULT '1' NOT NULL, \n\tworks_friday INTEGER DEFAULT '0' NOT NULL, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id)\n)"))
    op.execute(sa.text("CREATE TABLE personnel_images (\n\tid SERIAL NOT NULL, \n\tpersonnel_id INTEGER NOT NULL, \n\tstorage_key TEXT NOT NULL, \n\tdescription TEXT, \n\tis_primary INTEGER DEFAULT '0' NOT NULL, \n\tuploaded_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tembedding_id TEXT, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(personnel_id) REFERENCES personnel (id) ON DELETE CASCADE\n)"))
    op.execute(sa.text('CREATE INDEX idx_personnel_images_personnel ON personnel_images (personnel_id)'))
    op.execute(sa.text('CREATE INDEX idx_personnel_images_primary ON personnel_images (personnel_id, is_primary)'))
    op.execute(sa.text("CREATE TABLE personnel_requests (\n\tid SERIAL NOT NULL, \n\tpersonnel_id INTEGER NOT NULL, \n\trequest_type TEXT DEFAULT 'leave' NOT NULL, \n\tstart_date DATE NOT NULL, \n\tend_date DATE NOT NULL, \n\treason TEXT, \n\tstatus TEXT DEFAULT 'pending' NOT NULL, \n\tapproved_by INTEGER, \n\trejection_reason TEXT, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(personnel_id) REFERENCES personnel (id) ON DELETE CASCADE\n)"))
    op.execute(sa.text('CREATE INDEX idx_requests_dates ON personnel_requests (start_date, end_date)'))
    op.execute(sa.text('CREATE INDEX idx_requests_personnel ON personnel_requests (personnel_id)'))
    op.execute(sa.text('CREATE INDEX idx_requests_status ON personnel_requests (status)'))
    op.execute(sa.text('CREATE TABLE sections (\n\tid SERIAL NOT NULL, \n\tbuilding_id INTEGER, \n\tname TEXT NOT NULL, \n\tdescription TEXT, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(building_id) REFERENCES buildings (id) ON DELETE SET NULL\n)'))
    op.execute(sa.text('CREATE INDEX idx_sections_building ON sections (building_id)'))
    op.execute(sa.text('CREATE INDEX idx_sections_name ON sections (name)'))
    op.execute(sa.text('CREATE TABLE rooms (\n\tid SERIAL NOT NULL, \n\tsection_id INTEGER, \n\tname TEXT NOT NULL, \n\tdescription TEXT, \n\tpolygon_json TEXT, \n\tcreated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tupdated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(section_id) REFERENCES sections (id) ON DELETE SET NULL\n)'))
    op.execute(sa.text('CREATE INDEX idx_rooms_name ON rooms (name)'))
    op.execute(sa.text('CREATE INDEX idx_rooms_section ON rooms (section_id)'))
    op.execute(sa.text('CREATE TABLE detection_room_matches (\n\tid SERIAL NOT NULL, \n\tdetection_type TEXT NOT NULL, \n\tdetection_event_id INTEGER NOT NULL, \n\troom_id INTEGER NOT NULL, \n\tpersonnel_id INTEGER, \n\tcamera_id TEXT, \n\tmatched_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(room_id) REFERENCES rooms (id) ON DELETE CASCADE\n)'))
    op.execute(sa.text('CREATE INDEX idx_matches_detection ON detection_room_matches (detection_type, detection_event_id)'))
    op.execute(sa.text('CREATE INDEX idx_matches_personnel ON detection_room_matches (personnel_id)'))
    op.execute(sa.text('CREATE INDEX idx_matches_room ON detection_room_matches (room_id)'))
    op.execute(sa.text('CREATE TABLE personnel_room_access (\n\tid SERIAL NOT NULL, \n\tpersonnel_id INTEGER NOT NULL, \n\troom_id INTEGER NOT NULL, \n\tgranted_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tgranted_by TEXT, \n\tPRIMARY KEY (id), \n\tCONSTRAINT uq_personnel_room_access UNIQUE (personnel_id, room_id), \n\tFOREIGN KEY(room_id) REFERENCES rooms (id) ON DELETE CASCADE\n)'))
    op.execute(sa.text('CREATE INDEX idx_access_personnel ON personnel_room_access (personnel_id)'))
    op.execute(sa.text('CREATE INDEX idx_access_room ON personnel_room_access (room_id)'))


def downgrade() -> None:
    op.execute(sa.text('DROP TABLE IF EXISTS "personnel_room_access" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "detection_room_matches" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "rooms" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "sections" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "personnel_requests" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "personnel_images" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "work_shifts" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "users" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "revoked_tokens" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "plate_logs" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "plate_general_settings" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "plate_camera_settings" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "personnel" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "model_general_settings" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "model_conversion_jobs" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "human_logs" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "holidays" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "fire_smoke_settings" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "fire_smoke_logs" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "face_quality_settings" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "face_embeddings" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "cameras" CASCADE'))
    op.execute(sa.text('DROP TABLE IF EXISTS "buildings" CASCADE'))
