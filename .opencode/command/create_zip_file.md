---
description: Create a lightweight ZIP containing only the project files needed for URL, request, and output-structure comparison.
agent: create_zip_file
---

Create a lightweight I/O-analysis ZIP archive using the requested filename: `$ARGUMENTS`.

Follow the `create-zip-file` skill and the `create_zip_file` agent instructions. The ZIP must be created in the current project directory, must not include heavy runtime assets or generated data, and must contain the source needed to inspect routes, requests, WebSocket messages, schemas, templates, and outputs. Redact secrets in temporary copies only, verify the final archive, and report its path, size, and entry count.
