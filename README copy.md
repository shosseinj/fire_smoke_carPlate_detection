# OpenCode Live Redis -> MinIO Workflow

A small Git-enforced workflow for diagnosing and implementing asynchronous live frame/video persistence without disturbing `ai-branch` or unrelated real-time code.

## What it gives you

- OpenCode project rules
- diagnosis-first workflow
- Git snapshot of `live` and `ai-branch`
- explicit allow-list of files that may change
- automatic warning for every other changed file
- four simple OpenCode commands

## Install

Copy the contents of this folder into the ROOT of your existing repository.
Do not copy the outer folder itself if that would create a nested project.

Your repository should then contain:

```text
repo/
  AGENTS.md
  PROJECT_STATE.md
  DECISIONS.md
  PROTECTED_COMPONENTS.md
  opencode.json
  tools/
    start_phase.py
    check_changes.py
  .opencode/commands/
    phase-status.md
    diagnose-video-save.md
    implement-video-save.md
    verify-live.md
```

Commit the workflow files if you want them permanently versioned.

## Step 0 — start safely

First switch to your `live` branch yourself:

```bash
git checkout live
```

Then start in diagnosis-only mode:

```bash
python tools/start_phase.py
```

No application file is allowed to change yet.
The script records:
- current `live` HEAD
- current `ai-branch` HEAD

Check protection:

```bash
python tools/check_changes.py
```

## Step 1 — diagnose before editing

Start OpenCode in the repository and run:

```text
/diagnose-video-save
```

OpenCode should trace:

```text
frame -> enqueue -> Redis -> worker -> video/frame creation -> MinIO
```

It should tell you the FIRST broken boundary.
Do not ask it to redesign everything yet.

## Step 2 — allow only the files needed for the first fix

After diagnosis, suppose OpenCode proves the problem is in:

```text
services/video_queue.py
workers/video_worker.py
```

Restart/update the phase allow-list:

```bash
python tools/start_phase.py --allow services/video_queue.py workers/video_worker.py
```

Now only those application files may change.

If the real paths are directories, you can allow a directory:

```bash
python tools/start_phase.py --allow services/video_persistence workers/video_persistence
```

Keep the allow-list as small as possible.

## Step 3 — implement ONE fix

In OpenCode:

```text
/implement-video-save
```

The command tells OpenCode to:
- change only the confirmed failing boundary
- avoid unrelated refactors
- keep MinIO/video I/O off the live/GPU path
- show the Git diff
- run the protection checker

## Step 4 — inspect what happened

Run:

```text
/verify-live
```

You should see PASS/FAIL/UNKNOWN for the queue, worker, MinIO path, live behavior, and Git protection.

You can also manually run:

```bash
git diff --stat
git diff
python tools/check_changes.py
```

If the checker prints an unexpected file, DO NOT continue to the next step until you understand why it changed.

## Step 5 — continue boundary by boundary

If Redis enqueue now works but the worker does not consume, diagnose that boundary next.
Then allow only the worker file(s), implement one fix, verify again.

Repeat:

```text
DIAGNOSE -> ALLOW SMALL SCOPE -> IMPLEMENT ONE FIX -> GIT CHECK -> VERIFY
```

## Example session

```bash
git checkout live
python tools/start_phase.py
opencode
```

Then inside OpenCode:

```text
/diagnose-video-save
```

Suppose result:

```text
frame production: PASS
enqueue call: PASS
Redis message: PASS
worker consumption: FAIL
first broken boundary: worker is reading a different Redis stream name
```

Exit or use another terminal and allow only the worker config/file:

```bash
python tools/start_phase.py --allow workers/video_worker.py config/redis.py
```

Then OpenCode:

```text
/implement-video-save
/verify-live
```

Expected Git protection report:

```text
Allowed/expected changed files:
  OK   workers/video_worker.py

Unexpected/protected changed files:
  (none)

ai-branch unchanged: YES
CHECK PASSED
```

If instead you see:

```text
FAIL api/live_endpoint.py
FAIL ai/inference.py
```

stop. OpenCode must report why these changed before anything else is done.

## Important real-time rule

Do not solve the problem by putting video encoding or MinIO upload directly into the GPU/live frame loop.
The intended shape is:

```text
live path -> quick enqueue/publish -> continue real-time work
                    |
                    v
                Redis
                    |
                    v
             async worker
                    |
                    v
              encode/upload
                    |
                    v
                  MinIO
```

The exact Redis data structure and worker style should follow your existing project rather than introducing unnecessary new infrastructure.

## Commands

- `/phase-status` — show scope and Git protection state
- `/diagnose-video-save` — read-only root-cause tracing
- `/implement-video-save` — make one approved minimal fix
- `/verify-live` — verify function + real-time safety + Git changes
