# Windows PowerShell installation

## 1. Install OpenCode when needed

```powershell
npm install -g opencode-ai
opencode --version
```

## 2. Extract the package

```powershell
Expand-Archive `
  -Path .\opencode_realtime_agentic_workflow_v3.zip `
  -DestinationPath .\opencode-workflow
```

## 3. Install into the new project

The package already contains the Excel file you supplied, so no previous-project path is needed:

```powershell
cd .\opencode-workflow\opencode_realtime_agentic_workflow_v3

python .\installer\install.py `
  --target "C:\Users\YOUR_USER\Desktop\new-project"
```

To use another Excel file:

```powershell
python .\installer\install.py `
  --target "C:\Users\YOUR_USER\Desktop\new-project" `
  --options-excel "C:\Users\YOUR_USER\Desktop\previous_project_options.xlsx"
```

## 4. Validate and start

```powershell
cd "C:\Users\YOUR_USER\Desktop\new-project"
python .\scripts\validate_agentic_workflow.py
opencode
```

## 5. Initial setup only

```text
/bootstrap-agentic
```

Paste `RUN_PROMPT.md` when you want a repository and evaluation baseline without implementing all catalog options.

## 6. Migrate selected options step by step

```text
/migrate-selected-options App: Authentication. Routes: /api/v1/auth/login and /api/v1/auth/me. Login accepts username and password, returns access and refresh tokens, and /me returns the authenticated profile.
```

Use a new prompt for the next selection. OpenCode must leave all unmentioned catalog entries `PENDING`.

## 7. Review status

```text
/migration-status all
/evaluate-project
```

## 8. Import an updated Excel file later

```powershell
python .\scripts\import_option_catalog.py `
  --excel "C:\path\to\updated_options.xlsx"
```

## Upgrade existing workflow files

```powershell
python .\installer\install.py `
  --target "C:\path\to\new-project" `
  --options-excel "C:\path\to\previous_project_options.xlsx" `
  --force
```

Conflicting files are backed up under `.agentic/install-backups/<timestamp>/`.
