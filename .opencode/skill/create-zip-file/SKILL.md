---
name: create-zip-file
description: Use when the user asks to create a lightweight ZIP source snapshot for an AI agent to compare project URLs, requests, WebSocket messages, and response/output structures.
compatibility: Windows PowerShell 5.1 and standard ZIP tooling
---

# Create Lightweight I/O Archive

Create the requested ZIP in the current project directory. The archive is for static I/O analysis, not standalone execution.

## Required Input

- Take the ZIP filename from the user's argument.
- Preserve the requested filename exactly. If no `.zip` suffix is supplied, add `.zip`.
- Reject paths, rooted names, empty names, and names containing `..`; the output must be a file in the current project directory.
- Never overwrite a non-ZIP file. Overwriting an existing ZIP with the requested name is allowed only after stating that it will be replaced.

## Include

Start by inspecting the project rather than assuming its framework. Include the smallest set of text/source files needed to understand external behavior:

- Application entry points and route registration
- HTTP and WebSocket route/controller/handler implementations
- Request, response, validation, serialization, and DTO/schema definitions
- Services and data-access/model code directly used by those handlers
- HTML/templates or frontend source that defines request calls or rendered output
- API, contract, integration, and route-registry tests
- Runtime configuration, dependency manifests, migration/schema definitions, and concise endpoint/change notes
- A short archive scope note in the archive staging area when useful

For this Python/FastAPI project, this normally means the application Python files, templates, relevant JSON/config files, migrations, and `tests/*.py`, while omitting unrelated model-inference implementation.

## Exclude

Do not include:

- `.git`, `.opencode` runtime state, caches, `__pycache__`, logs, temporary files, or generated reports
- Virtual environments, dependency directories, wheels, lockfile caches, and build output
- Databases, Qdrant/vector stores, uploads, videos, images, fonts, and model weights (`.pt`, `.onnx`, `.engine`, etc.)
- Bundled/minified static dependencies such as Swagger UI unless the user explicitly asks for them
- Existing ZIP archives, including the output archive itself
- Secrets, credentials, private keys, tokens, or environment-specific connection data
- README or seed-data files when they contain credentials and are not required for I/O shape analysis

## Security

Inspect selected text files for secrets before archiving. Redact credential literals only in temporary staging copies; never modify the project source. At minimum handle database/RTSP URL passwords, API keys, JWT secrets, and hard-coded default passwords. Verify the final archive does not contain the original secret values.

## Implementation

1. Inspect the repository and identify the runtime/API surface.
2. Build a temporary staging directory outside the project, preserving relative paths.
3. Copy only the selected files into staging.
4. Apply redaction to staging copies only.
5. Create the requested ZIP from staging, never by recursively compressing the whole project.
6. Verify the ZIP exists, contains the expected route/schema/test files, excludes heavy/generated entries, and report its file count and size.
7. Remove the temporary staging directory. Leave only the requested ZIP in the project as a generated artifact.

Use PowerShell-compatible commands on Windows. Do not use destructive Git commands or alter unrelated user changes.
