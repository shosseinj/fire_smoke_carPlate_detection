# OpenCode Real-Time Agentic Workflow — Excel Catalog Edition

This package installs an OpenCode workflow for real-time applications such as fire, smoke, vehicle, and license-plate detection.

It does **not** need the previous project repository. Previous options and routes are supplied through an Excel catalog and then selected step by step in prompts.

The workflow enforces these behaviors:

1. Only the rows, routes, options, or application areas selected in the current prompt are implemented.
2. A selected option is propagated through every necessary layer, even when the prompt names only one endpoint or setting.
3. Unselected catalog items remain `PENDING`.
4. Prompt details enrich the selected Excel rows and override ambiguous catalog text.
5. Missing behavior is inferred only from verified new-project patterns and recorded as an assumption.
6. The agent never claims previous-project behavioral parity when no previous source implementation was provided.
7. Every important project section is registered in a project-wide testing and evaluation system.
8. A feature is not complete until connectivity, regression, deployment, and representative real-time behavior are evaluated honestly.

## Included Excel catalog

The package includes the supplied `app_names_and_urls.xlsx` workbook. You can install using that bundled file or provide another `.xlsx` file with `--options-excel`.

Supported columns include:

- `App Name`, `Application`, `Subsystem`, or `Category`
- `URL`, `Route`, or `Endpoint`
- `Option`, `Feature`, `Name`, or `Setting`
- `Method`
- `Description`
- `Expected Behavior`
- `Default Value`
- `Allowed Values`
- `Notes`
- `Priority`

At minimum, a row needs a route/URL or an option/feature name.

## Requirements

- Python 3.9 or newer
- OpenCode
- Target new-project directory
- An `.xlsx` options catalog; the bundled workbook is used by default

## Install on Windows

Use the bundled Excel catalog:

```powershell
Expand-Archive .\opencode_realtime_agentic_workflow_v3.zip -DestinationPath .\opencode-workflow
cd .\opencode-workflow\opencode_realtime_agentic_workflow_v3
python .\installer\install.py --target "C:\path\to\new-project"
```

Or provide your own Excel file:

```powershell
python .\installer\install.py `
  --target "C:\path\to\new-project" `
  --options-excel "C:\path\to\previous_project_options.xlsx"
```

Validate and start:

```powershell
cd "C:\path\to\new-project"
python .\scripts\validate_agentic_workflow.py
opencode
```

Initial setup:

```text
/bootstrap-agentic
```

Then migrate only one selected scope:

```text
/migrate-selected-options App: Cameras. Route: /api/v1/cameras/{camera_id}/effective-settings. It returns the resolved camera settings after applying project defaults and camera overrides.
```

Review progress:

```text
/migration-status all
```

Use `STEP_BY_STEP_PROMPT.md` for a reusable detailed prompt.

## Reimport a changed Excel catalog

From the target project:

```powershell
python .\scripts\import_option_catalog.py `
  --excel "C:\path\to\updated_previous_project_options.xlsx"
```

Matching item progress is preserved by default.

## Installed state files

- `.agentic/migration/source/previous-project-options.xlsx`
- `.agentic/migration/legacy-endpoint-catalog.json`
- `.agentic/migration/current-scope.json`
- `.agentic/migration/legacy-option-map.json`
- `.agentic/migration/migration-history.json`
- `.agentic/evaluation/section-registry.json`

The installer merges a marked block into `AGENTS.md`, installs OpenCode agents, commands, and skills, and preserves conflicting workflow files unless `--force` is used.
