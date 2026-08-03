# OpenCode Live-Branch Workflow Design

## Goal

Reorganize the decode-tee GPU live-branch OpenCode workflow so each requirement has one authoritative owner while preserving the production feature, the protected AI branch, useful multi-agent behavior, and unrelated worktree changes.

## Authority model

`.agentic/live-branch/config.yaml` is the only owner of mutable product and runtime values. The canonical wall profile is represented as numeric `width: 320` and `height: 320` fields. URLs, route fragments, feature flags, heartbeat intervals, shutdown grace, latency targets, report paths, and canonical agent names also live only in this configuration.

Existing values in `.agentic/live-branch/config.yaml` are immutable unless the user explicitly requests changing them. Workflow installation, normalization, and validation must preserve those values byte-for-byte. For this refactor, the user's explicit `320*320` instruction confirms the existing wall dimensions; it does not authorize changing any other configuration value.

Durable rules are divided by responsibility across the ten required `.opencode/instructions/*.md` files. They describe invariants and how to resolve configuration keys, but do not copy mutable values. The AI-protection instruction remains the absolute boundary for production work.

`.agentic/live-branch/spec.md` is a scope and authority index. It links to the configuration and focused instructions instead of restating their requirements. State, architecture map, AI baseline, validation matrix, and final report are evidence artifacts; observations must identify the governing config key and must not become normative sources.

## Agents and commands

The orchestrator remains a small coordinator: load the shared sources, enforce stage ordering, dispatch canonical agents, collect gates, and update evidence. Each specialized agent reads the same configuration and instruction index, then applies only its focused role instructions. Agent prompts contain no mutable product values.

Commands are thin workflow entry points. Each selects a canonical agent and action without embedding the full feature specification. The five required command filenames replace the two obsolete decode-tee-prefixed command names.

Frontmatter uses syntax accepted by the installed OpenCode 1.18.11 CLI. Permissions are restricted by role, use only supported action values, and the orchestrator may delegate only to configured workflow agents.

## Validation

`scripts/validate_live_branch_config.py` validates YAML structure, types, bounds, canonical agent names, paths, URL/route values, timing relationships, and the 320 by 320 wall profile.

`scripts/validate_opencode_live_branch_workflow.py` validates the complete installed workflow: required and obsolete files, frontmatter shape, supported modes/actions, permission mappings, command agents, orchestrator delegation, shared-source references, instruction/config references, unknown agents, stale filenames, and forbidden duplicated mutable values. It invokes the local OpenCode CLI to prove that every agent and the resolved project configuration parse.

Tests construct temporary workflow fixtures and demonstrate failures for missing files, stale references, conflicting values, invalid frontmatter, unknown agent names, and malformed permissions before the validator implementation is completed.

The PowerShell installer copies or normalizes the canonical repository workflow and runs both validators. It does not rewrite an existing `.agentic/live-branch/config.yaml` and does not modify production application files.

## Execution and safety

Existing modifications in `app/api/live_branch.py`, `app/core/deepstream_ingestor.py`, and `app/web/dashboard.html` are user-owned and remain unstaged and unchanged. No production live-branch or AI-branch code is in scope. Changes are committed locally in logical groups; pushing, merging, rebasing, resetting, and cleaning are forbidden.

Verification consists of validator unit tests, both validators against the repository, Python compilation, PowerShell installer validation, OpenCode agent parsing, resolved-config parsing, stale/conflict searches, and a final staged-file scope audit.
