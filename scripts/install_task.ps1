<#
.SYNOPSIS
    Registers (or removes) the Windows Scheduled Task that runs the daily
    forward-trading runner, `python -m bots.run_daily`, once a day.

.DESCRIPTION
    Fires daily at 06:30 SGT. That one fixed local-clock time is only correct
    if THIS MACHINE's own clock is set to Singapore time (SGT has no DST,
    which is the whole point of the choice — see docs/plans/phase5-forward-
    runner.md). If this machine runs a different timezone, change -At below
    to whatever local wall-clock time is comfortably after both 04:00 and
    05:00 SGT the next morning (NYSE's close as seen from Singapore, EDT and
    EST respectively) and update this comment to say what timezone is
    actually in effect here.

    Idempotent: re-running updates the existing task's action/trigger/settings
    in place rather than erroring or creating a duplicate.

    Bakes in two paths when run — the project root and the venv's python.exe —
    both printed before anything is registered, because a task that silently
    points at the wrong interpreter is tedious to diagnose later.

.PARAMETER Remove
    Unregister the task instead of creating/updating it.

.EXAMPLE
    powershell -File scripts\install_task.ps1
    powershell -File scripts\install_task.ps1 -WhatIf      # preview only
    powershell -File scripts\install_task.ps1 -Remove       # unregister
#>
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
        Write-Host "No such task ($TaskName) — nothing to remove."
        exit 0
    }
    if ($PSCmdlet.ShouldProcess($TaskName, "Unregister scheduled task")) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed task $TaskName."
    }
    exit 0
}

if (-not (Test-Path $PythonExe)) {
    Write-Error (
        "No venv python at $PythonExe — create the venv and install " +
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
    -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$verb = if ($existing) { "Update" } else { "Register" }
$description = (
    "nasdaq-advisor: daily forward paper-trading runner. Fake money only — " +
    "see CLAUDE.md. Exit code 0 = fine/nothing to do, 1 = a bot failed " +
    "(check data\advisor.log), 2 = a config/data error before any bot ran."
)

if ($PSCmdlet.ShouldProcess($TaskName, "$verb scheduled task (daily 06:30 -> python -m bots.run_daily)")) {
    if ($existing) {
        Set-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings | Out-Null
    } else {
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
            -Settings $Settings -Principal $Principal -Description $description | Out-Null
    }
    Write-Host "${verb}d task ${TaskName}: daily 06:30, $PythonExe -m bots.run_daily (cwd $ProjectRoot)"
}
