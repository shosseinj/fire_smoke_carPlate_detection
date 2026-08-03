# AI Branch Protection

The existing AI branch is protected.

All agents, commands, subtasks, and workflows must preserve the AI branch structurally and behaviorally.

## Non-negotiable rules

Do not modify:

- AI GStreamer elements
- AI element order
- AI caps
- AI resolution
- AI frame rate
- AI batch size
- AI queues
- AI latency settings
- AI memory type
- AI source routing
- AI inference inputs or outputs
- AI appsink behavior
- AI model configuration
- AI preprocessing or postprocessing

The live branch must be implemented only as an independent sibling branch from the post-decode NVMM tee.

The live branch must not consume frames from the AI appsink or any CPU-side AI frame path.

## Shared-file rule

Editing a shared pipeline file is allowed only when necessary to attach or manage the live branch.

Such edits must not alter the AI branch.

Before changing a shared file:

1. Record the AI pipeline structure.
2. Identify the exact live-branch-only change.
3. Add or run AI-isolation tests.

After changing a shared file:

1. Compare AI branch behavior before and after.
2. Confirm the AI branch element chain is unchanged.
3. Confirm AI caps, FPS, resolution, queues, and memory type are unchanged.
4. Run AI regression tests.
5. Reject the change if AI behavior changed.

## Approval gate

Any proposed change that alters the AI branch requires explicit user approval before implementation.

Without explicit approval, the agent must stop and report:

AI BRANCH CHANGE REQUIRED — USER APPROVAL NEEDED