# Detection-log media storage update

## Deployment

Run the new PostgreSQL migration before starting the updated application:

```bash
alembic upgrade head
```

Migration `20260728_0039_detection_media_storage.py` adds the face-thumbnail and video-readiness fields and converts historical `/media/...` values into root-relative storage keys.

## Media contract

Detection-log rows now store private root-relative keys such as:

```text
detected_faces/<file>.jpg
face_thumbnails/<file>.jpg
body_images/<file>.jpg
full_frame_images/<file>.jpg
human_videos/<file>.mp4
human_face_videos/<file>.mp4
```

Raw storage keys are not returned by detection-log API responses. Responses may embed only `face_thumbnail`; all original media is accessed through authenticated API routes:

```text
GET /api/v1/logs/{log_id}/face
GET /api/v1/logs/{log_id}/body
GET /api/v1/logs/{log_id}/snapshot
GET /api/v1/logs/{log_id}/video
GET /api/v1/logs/{log_id}/face-video
```

Each route displays inline by default. Add `?download=true` to request an attachment. Video routes support HTTP byte ranges for browser playback and seeking.

`GET /api/v1/logs/filter` preserves the existing `include_thumbnails` query option. When it is false, `face_thumbnail` is `null`, but available protected media URLs are still returned.

## Privacy and safety

The compatibility `/media` mount remains available for non-person media such as fire/smoke and plate evidence. Direct access to personnel and person-detection directories is blocked; clients must use authenticated API routes.

All detection media paths are resolved against `SAVED_MEDIA_PATH`. Absolute paths outside that root, remote URLs, and traversal keys are rejected.

## Real-time behavior

- `body_image` is now the tracked person's body crop.
- The highest-quality face observed for the track is saved as the original face image.
- A compressed JPEG face thumbnail (maximum 224 pixels) is saved and embedded when requested.
- The full snapshot is the complete camera frame.
- Video URLs are published only after the MP4 writer is released and the file is readable.
- Media keys survive idle writer cleanup before the final track event.
- Replaced and deleted application-managed detection media is cleaned safely within `SAVED_MEDIA_PATH`.

## Included checks

```bash
PYTHONPATH=. python -m pytest -q tests
python -m compileall -q app alembic
```

The archive includes focused tests for storage-key safety, private static-media blocking, thumbnail size/encoding, protected response URLs, video readiness, and byte-range delivery.
