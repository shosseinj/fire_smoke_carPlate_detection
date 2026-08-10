# Protected Components

During this phase everything is protected EXCEPT files explicitly allowed in `.workflow/phase.json`.

## Always protected

- Git ref/branch: `ai-branch`
- AI/inference code unless explicitly approved
- Existing unrelated HTTP/WebSocket/API endpoints
- Existing live stream behavior unrelated to persistence
- GPU inference path
- Authentication and authorization behavior
- Existing response schemas/contracts
- Database behavior unrelated to video persistence

## Rule

If implementation requires touching a protected area:

1. Do not make the change yet.
2. Explain exactly why it appears necessary.
3. Show the relevant Git diff or code dependency.
4. Wait for explicit user approval before adding that path to the allowed list.
