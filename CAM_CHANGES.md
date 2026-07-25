# Cam API and hierarchy changes

## Hierarchy

The location hierarchy is now:

`building -> section -> cam -> room`

- `cam.section_id` references `sections.id`.
- `rooms.cam_id` references `cam.id`.
- New rooms must provide `camera_id` through the room API.
- The existing `rooms.section_id` column is retained as a compatibility/cache field for existing zone and reporting code. It is derived from the selected cam and synchronized when a cam moves to another section.
- Existing rooms remain valid after migration and can be assigned to a cam later.

`camera_number` is unique within a section.

## Cam fields

- `camera_name`
- `camera_number`
- `width`
- `high`
- `source_type`: `usb`, `rtsp`, or `other`
- `section_id`
- `url`

## Endpoints

- `POST /api/v1/cams` — create
- `GET /api/v1/cams` — list; supports `section_id` and `source_type` filters
- `GET /api/v1/cams/{cam_id}` — read one
- `PATCH /api/v1/cams/{cam_id}` — update
- `DELETE /api/v1/cams/{cam_id}` — delete; blocked while rooms reference the cam
- `POST /api/v1/cams/health-check` — accepts `{ "url": "..." }` and returns `status`, `message`, and `snapshot`

## Room integration

- Room create/update uses `camera_id` as the parent cam reference.
- `PATCH /rooms/{room_id}/assign-camera/{cam_id}` assigns a room to a cam.
- `GET /rooms/?camera_id={cam_id}` lists rooms belonging to a cam.
- The old source-registry assignment endpoint remains available for backward compatibility but is hidden from OpenAPI.

## Migration

Run:

```bash
alembic upgrade head
```

The new head revision is `20260725_0023`.
