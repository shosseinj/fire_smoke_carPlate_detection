# send_recent_detections implementation report

## Implemented

- Added `app/core/recent_detection_service.py`.
- Loads the latest known and unknown detection logs independently, then merges them in descending `detection_time` order.
- Uses `WEBSOCKET_RECENT_DETECTIONS_LIMIT` as the per-group limit, with the default yielding up to 50 known plus 50 unknown detections.
- Resolves personnel names, room names, personnel reference images, general thresholds, and camera overrides.
- Reproduces legacy `unknown` / `unsure` / `known` classification boundaries.
- Concatenates the detected face with the selected reference image for `unsure` and `known` detections.
- JPEG-encodes output at quality 70 and returns raw Base64, matching the old payload.
- Formats detection time as Jalali in `Asia/Tehran`.
- Sends the legacy message immediately after WebSocket acceptance and before live frame delivery:

```json
{
  "type": "recent_detections",
  "detections": [],
  "count": 0
}
```

As in the old project, no message is sent when no usable detection images are available.

## Validation

- Python compilation passed for the modified files and the full `app` package.
- Classification boundary checks passed.
- Jalali/timezone conversion check passed.
- Full application import was not possible in the execution environment because `bcrypt` is not installed.
