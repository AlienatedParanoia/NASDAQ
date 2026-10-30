<#
.SYNOPSIS
    Registers (or removes) the Windows Scheduled Task that runs the daily
    forward-trading runner, `python -m bots.run_daily`, once a day.

.DESCRIPTION
    Fires daily at 06:30 SGT. That one fixed local-clock time is only correct
    if THIS MACHINE's own clock is set to Singapore time (SGT has no DST,
    which is the whole point of the choice - see docs/plans/phase5-forward-
    runner.md). If this machine runs a different timezone, change -At below
    to whatever local wall-clock time is comfortably after both 04:00 and
    05:00 SGT the next morning (NYSE's close as seen from Singapore, EDT and
    EST respectively) and update this comment to say what timezone is
    actually in effect here.

    Runs whether or not you are logged in (LogonType S4U: no password is
    stored, and the task gets no network credentials - fine for Yahoo, which
    needs none). Registering an S4U task needs an elevated (Run as
    Administrator) PowerShell; the script refuses up front without one.

    Idempotent: re-running replaces the existing task's action, trigger,
    settings, principal and description in place (Register-ScheduledTask
    -Force) rather than erroring or creating a duplicate.

    Exit codes of the runner itself, as shown in Task Scheduler's "Last Run
    Result": 0 fine / nothing to do, 1 a bot failed, 2 a config or argument
    error before any bot ran, 3 waiting for price data (retries next run).

    Bakes in two paths when run - the project root and the venv's python.exe -
    both printed before anything is registered, because a task that silently
    points at the wrong interpreter is tedious to diagnose later.

.PARAMETER Remove
    Unregister the task instead of creating/updating it.

.EXAMPLE
    powershell -File scripts\install_task.ps1               # from an admin prompt
    powershell -File scripts\install_task.ps1 -WhatIf      # preview only
    powershell -File scripts\install_task.ps1 -Remove       # unregister
#>
# Keep this file pure ASCII: Windows PowerShell 5.1 reads a BOM-less script as
# ANSI, and a UTF-8 em-dash decodes to a curly quote that breaks parsing.
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Remove
)

$TaskName = "nasdaq-advisor-daily-run"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "Project root : $ProjectRoot"
Write-Host "Python       : $PythonExe"
Write-Host "Task name    : $TaskName"

if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "No such task ($TaskName) - nothing to remove."
        exit 0
    }
    if ($PSCmdlet.ShouldProcess($TaskName, "Unregister scheduled task")) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed task $TaskName."
    }
    exit 0
}

$IsAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $IsAdmin) {
    if ($WhatIfPreference) {
        Write-Warning "Not elevated: the real run must be from an Administrator PowerShell (S4U task)."
    } else {
        Write-Error (
            "Registering a task that runs while you are logged out (S4U) needs " +
            "an elevated PowerShell. Right-click PowerShell -> Run as " +
            "administrator, then re-run this script."
        )
        exit 1
    }
}

if (-not (Test-Path $PythonExe)) {
    Write-Error (
        "No venv python at $PythonExe - create the venv and install " +
        "requirements first (see CLAUDE.md's Setup section) before " +
        "registering this task."
    )
    exit 1
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe -Argument "-m bots.run_daily" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At "06:30"
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 1)
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$verb = if ($existing) { "Update" } else { "Register" }
$description = (
    "nasdaq-advisor: daily forward paper-trading runner. Fake money only - " +
    "see CLAUDE.md. Exit code 0 = fine/nothing to do, 1 = a bot failed " +
    "(check data\advisor.log), 2 = a config/data error before any bot ran, " +
    "3 = waiting for price data (retries next run)."
)

if ($PSCmdlet.ShouldProcess($TaskName, "$verb scheduled task (daily 06:30 -> python -m bots.run_daily)")) {
    # -Force replaces an existing task wholesale, so the principal and
    # description are updated too - Set-ScheduledTask cannot change either
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
        -Settings $Settings -Principal $Principal -Description $description -Force | Out-Null
    $done = if ($existing) { "Updated" } else { "Registered" }
    Write-Host "$done task ${TaskName}: daily 06:30, $PythonExe -m bots.run_daily (cwd $ProjectRoot)"
}
