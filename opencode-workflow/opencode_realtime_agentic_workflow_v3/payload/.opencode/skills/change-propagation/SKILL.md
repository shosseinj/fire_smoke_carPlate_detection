---
name: change-propagation
description: Trace and implement a requested option or feature through all required project layers, including inferred endpoint and downstream changes
compatibility: opencode
metadata:
  domain: real-time-software
  workflow: feature-integration
---
# Change propagation

## Use this skill when

Use this for any new option, behavior, detector setting, model parameter, API field, UI control, event field, processing mode, deployment setting, or migrated feature.

## Required trace

Find the most similar existing behavior and trace:

```text
Definition -> default -> validation -> external input -> serialization -> service propagation -> worker/queue -> runtime consumer -> result/event -> storage -> API/WebSocket -> UI/downstream consumer -> logs/metrics -> tests -> deployment
```

Not every project has every step. Discover the real path.

## Endpoint inference

When behavior can be controlled externally, determine the correct interface: REST request body/query, WebSocket message, CLI, environment variable, configuration file, UI control, or another project-specific boundary.

Add or update the interface even when the user did not say "update the endpoint." Do not expose strictly internal behavior unnecessarily; document the verified reason.

## Compatibility

Prefer safe defaults, unchanged behavior when omitted, validation errors for invalid values, compatible serialization, and old-client support. Identify unavoidable breaking changes explicitly.

## Acceptance cases

At minimum consider omitted/default, enabled, disabled or alternate, boundary, invalid, input propagation, runtime consumption, downstream output, old behavior regression, and performance impact.

## Completion evidence

A feature is incomplete when no real execution path reaches it, its output is not consumed, connected sections were not checked, or representative runtime validation is missing without an honest status.
