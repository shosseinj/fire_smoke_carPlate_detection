# Backend and AI Parallel OpenCode Agents Design

## Goal

Replace the repository's mixed and partly obsolete OpenCode workflows with one beginner-friendly workflow for implementing backend and AI features quickly. The workflow must use parallel work only when tasks have non-overlapping ownership and must preserve unrelated working behavior.

## Current Problems

- The active `safe-lead -> explorer -> implementer -> reviewer` command is fully sequential.
- All custom agents use `bash: ask`, so routine searches, diffs, and targeted tests repeatedly interrupt execution.
- The general implementer can edit any project file and has no explicit ownership protocol for concurrent work.
- The lead can see and invoke unrelated built-in subagents because its task permissions are not restricted.
- `README copy.md` describes an older Redis-to-MinIO workflow and references the removed `PROTECTED_COMPONENTS.md` file.
- Validators remain for several removed OpenCode packages, including minimal-safe, Excel migration, GPU broadcast, and live-branch workflows. They fail against the current repository and do not validate the active agents.

## Agent Set

The project will expose four custom roles.

### `dev-lead`

The primary agent is the only coordinator. It translates the user's feature request into a small task graph, launches backend and AI investigation in parallel when both domains are relevant, assigns exclusive file ownership before edits, waits for results, resolves integration ordering, launches review, and reports evidence.

It may invoke only `backend-agent`, `ai-agent`, and `reviewer`. It must not delegate merely to create activity; a backend-only or AI-only request uses only the relevant specialist.

### `backend-agent`

This subagent investigates and implements backend work: FastAPI routes, services, persistence, database models and migrations, schemas, authentication, configuration, storage, queues, and backend tests.

It may edit only files explicitly assigned by `dev-lead`. It must report any required AI-owned or shared-file change instead of editing outside its assignment.

### `ai-agent`

This subagent investigates and implements AI and video work: processors, inference, model loading and export, tracking, DeepStream/GStreamer integration, frame routing, AI configuration, performance diagnostics, and AI tests.

It may edit only files explicitly assigned by `dev-lead`. It must report any required backend-owned or shared-file change instead of editing outside its assignment.

### `reviewer`

This read-only subagent reviews the combined diff, ownership compliance, API and AI integration, regression risk, concurrency, cleanup, error handling, and targeted test coverage. The lead may launch two reviewer tasks in parallel for a large cross-domain feature: one for correctness and one for tests/integration. Review findings use `BLOCKER`, `WARNING`, or `OK` severity.

## Parallel Workflow

The single user entry point is `/develop <feature request>`.

1. `dev-lead` inspects the request and current Git state without modifying unrelated files.
2. If both domains are involved, it launches `backend-agent` and `ai-agent` concurrently in investigation mode. If only one domain is involved, it launches only that agent.
3. Each specialist returns relevant paths, dependencies, risks, proposed ownership, and targeted validation.
4. The lead creates an ownership map. Every editable file has exactly one owner. Shared integration files are assigned to one specialist or retained for a later sequential integration step.
5. Specialists with disjoint ownership implement concurrently. Tasks that need the same file run sequentially.
6. Each specialist runs targeted tests for its owned change and returns changed files, commands, results, and unresolved integration needs.
7. The lead inspects the combined diff and performs any explicitly assigned integration step.
8. The reviewer checks the complete change. Independent review tasks may run concurrently when their scopes do not overlap.
9. The lead resolves blockers, reruns relevant validation, and reports the final outcome, changed files, test evidence, and remaining risks.

## Ownership Contract

Every implementation task must include:

- a concise objective;
- an explicit list of owned files or directories;
- a list of files that must not be edited;
- expected interfaces or outputs;
- targeted validation commands;
- a reminder that other agents may be working in the same repository and existing edits must not be reverted.

An agent must stop and report back if it discovers that completing its task requires an unassigned file. The lead may then reassign ownership or serialize the work. Agents must never reset, discard, overwrite, or reformat changes they do not own.

## Permissions

The configuration will use OpenCode's current `permission` syntax.

- The lead can edit integration and workflow files and can invoke only the three custom subagents.
- Backend and AI agents can edit their explicitly assigned project files, but their prompts enforce the ownership contract because OpenCode permissions cannot dynamically express a per-task file list.
- The reviewer cannot edit files.
- Routine read-only commands, repository searches, `git status`, `git diff`, and targeted test commands are allowed without repeated approval.
- Destructive filesystem operations, dependency installation or upgrades, Git commit/push/reset operations, external-directory access, and broad deployment actions remain denied or require explicit user approval.
- Environment and secret files retain protected read behavior. Agents must not print secrets.
- `subagent_depth` remains `1`; specialists cannot recursively create more agents.

## Repository Cleanup

The active workflow will have one source of truth: `opencode.json`, four agent files, one `/develop` command, and one validator.

The implementation will identify and remove or clearly retire only files that exclusively validate or document removed OpenCode packages. This includes the stale `README copy.md` protection workflow and validators whose required agents/state files no longer exist. Production application scripts, tests, feature documentation, and runtime configuration are outside this cleanup unless they exclusively serve an obsolete agent package.

No `PROTECTED_COMPONENTS.md` mechanism will be recreated. Protection comes from Git-state preservation, explicit ownership, constrained permissions, narrow diffs, and review.

## Validation

A unified validator will verify:

- `opencode.json` parses and selects `dev-lead`;
- exactly the intended custom agent and command files exist;
- all four agents parse through the installed OpenCode CLI;
- agent modes and edit permissions match their roles;
- the lead's task permission allows only the three project subagents;
- `subagent_depth` is `1`;
- the `/develop` command references only valid agents and contains the ownership and parallelism gates;
- stale agent names and `PROTECTED_COMPONENTS.md` references are absent from active workflow documentation;
- safe commands do not prompt unnecessarily while destructive commands remain protected.

The implementation will also run the unified validator, OpenCode resolved-config inspection, agent listing, relevant Python tests for the validator, and a final Git diff scope audit.

## Success Criteria

- The user needs only `/develop <request>` and does not need to manually coordinate agents.
- Backend and AI investigation starts in parallel for cross-domain features.
- Backend and AI implementation runs in parallel only with disjoint ownership.
- Small single-domain changes avoid unnecessary agents.
- No two agents edit the same file concurrently.
- Routine investigation and targeted tests do not repeatedly request shell approval.
- Obsolete OpenCode workflow references no longer create false validation failures.
- The final response always states what changed, what was tested, what failed, and what remains risky.
