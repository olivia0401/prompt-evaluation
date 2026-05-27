# Live progress monitor for the experiment runner.
# Reads outputs/results.jsonl and displays progress, cost, ETA, and per-model
# / per-task breakdowns. Re-renders every -RefreshSeconds.
#
# Run in a SECOND PowerShell window while run_experiment.py is going:
#   .\scripts\monitor.ps1                       # 10s refresh, auto-detect total
#   .\scripts\monitor.ps1 -Expected 7176        # show % against expected total
#   .\scripts\monitor.ps1 -RefreshSeconds 5     # faster refresh

param(
    [int]$RefreshSeconds = 10,
    [int]$Expected = 0,
    [string]$JsonlPath = "outputs/results.jsonl"
)

$DONE_STATUSES = @("ok", "ok_length_violation")
$FAIL_STATUSES = @("parse_fail", "refused", "truncated", "api_error", "rate_limited", "timeout")

$ScriptStart = Get-Date

function Format-Duration {
    param([TimeSpan]$Span)
    if ($Span.TotalHours -ge 1) {
        return "{0:N1}h" -f $Span.TotalHours
    } elseif ($Span.TotalMinutes -ge 1) {
        return "{0:N1}m" -f $Span.TotalMinutes
    } else {
        return "{0:N0}s" -f $Span.TotalSeconds
    }
}

function Read-Results {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return @()
    }
    $records = @()
    Get-Content -Path $Path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line) {
            try {
                $records += ($line | ConvertFrom-Json)
            } catch {
                # skip malformed lines (partial fsync race)
            }
        }
    }
    return $records
}

while ($true) {
    Clear-Host
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host " Prompt Eval - Live Monitor" -ForegroundColor Cyan
    Write-Host " File: $JsonlPath" -ForegroundColor Gray
    Write-Host " Refresh: ${RefreshSeconds}s   Started: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host "================================================================" -ForegroundColor Cyan

    $records = Read-Results -Path $JsonlPath
    $total = $records.Count

    if ($total -eq 0) {
        Write-Host ""
        Write-Host "No results yet. Waiting for $JsonlPath ..." -ForegroundColor Yellow
        Start-Sleep -Seconds $RefreshSeconds
        continue
    }

    $done = ($records | Where-Object { $DONE_STATUSES -contains $_.status }).Count
    $failed = ($records | Where-Object { $FAIL_STATUSES -contains $_.status }).Count
    $budgetTrip = ($records | Where-Object { $_.status -eq "budget_exceeded" }).Count

    $totalCost = ($records | Measure-Object -Property cost_usd -Sum).Sum
    if ($null -eq $totalCost) { $totalCost = 0 }

    $totalIn = ($records | Measure-Object -Property input_tokens -Sum).Sum
    $totalOut = ($records | Measure-Object -Property output_tokens -Sum).Sum
    $totalReason = ($records | Measure-Object -Property reasoning_tokens -Sum).Sum

    # Progress + ETA
    Write-Host ""
    if ($Expected -gt 0) {
        $pct = [math]::Round(($total / $Expected) * 100, 1)
        Write-Host (" Progress: {0,5} / {1,-5}  ({2,5}%)" -f $total, $Expected, $pct) -ForegroundColor Green

        $elapsed = (Get-Date) - $ScriptStart
        if ($total -gt 0 -and $elapsed.TotalSeconds -gt 0) {
            $rate = $total / $elapsed.TotalSeconds
            $remaining = $Expected - $total
            if ($rate -gt 0 -and $remaining -gt 0) {
                $etaSec = $remaining / $rate
                $etaSpan = [TimeSpan]::FromSeconds($etaSec)
                Write-Host (" Rate:     {0,5:N2} calls/s   ETA: {1}" -f $rate, (Format-Duration $etaSpan)) -ForegroundColor Gray
            }
        }
    } else {
        Write-Host (" Total recorded: $total (no --Expected set)") -ForegroundColor Green
    }

    Write-Host ""
    Write-Host " Status breakdown:" -ForegroundColor White
    $records | Group-Object status | Sort-Object Count -Descending | ForEach-Object {
        $color = if ($DONE_STATUSES -contains $_.Name) { "Green" }
                 elseif ($_.Name -eq "budget_exceeded") { "Red" }
                 elseif ($_.Name -eq "rate_limited") { "Yellow" }
                 else { "DarkYellow" }
        Write-Host ("   {0,-22} {1,6}" -f $_.Name, $_.Count) -ForegroundColor $color
    }

    Write-Host ""
    Write-Host " Cost & tokens:" -ForegroundColor White
    Write-Host ("   Total cost:        `${0:N4}" -f $totalCost) -ForegroundColor Cyan
    Write-Host ("   Input tokens:      {0:N0}" -f $totalIn)
    Write-Host ("   Output tokens:     {0:N0}" -f $totalOut)
    Write-Host ("   Reasoning tokens:  {0:N0}" -f $totalReason) -ForegroundColor Magenta
    if ($totalOut -gt 0) {
        $reasonPct = [math]::Round(($totalReason / $totalOut) * 100, 1)
        Write-Host ("   (reasoning is {0}% of output — watch this!)" -f $reasonPct) -ForegroundColor Magenta
    }

    Write-Host ""
    Write-Host " By model:" -ForegroundColor White
    $records | Group-Object model_key | Sort-Object Count -Descending | ForEach-Object {
        $cost = ($_.Group | Measure-Object -Property cost_usd -Sum).Sum
        if ($null -eq $cost) { $cost = 0 }
        $okCount = ($_.Group | Where-Object { $DONE_STATUSES -contains $_.status }).Count
        Write-Host ("   {0,-10} {1,5} calls  ok={2,5}  cost=`${3,7:N4}" -f $_.Name, $_.Count, $okCount, $cost)
    }

    Write-Host ""
    Write-Host " By task (top 10):" -ForegroundColor White
    $records | Group-Object task | Sort-Object Count -Descending | Select-Object -First 10 | ForEach-Object {
        Write-Host ("   {0,-22} {1,5}" -f $_.Name, $_.Count)
    }

    # Most recent entries
    Write-Host ""
    Write-Host " Last 3 entries:" -ForegroundColor White
    $records | Select-Object -Last 3 | ForEach-Object {
        $ts = if ($_.timestamp) { $_.timestamp.Substring(11, 8) } else { "-" }
        $brief = if ($_.brief_id) { $_.brief_id.PadRight(15).Substring(0, 15) } else { "?" }
        Write-Host ("   $ts  $brief  $($_.task)  $($_.model_key)  -> $($_.status)") -ForegroundColor DarkGray
    }

    Write-Host ""
    Write-Host " Ctrl+C to exit. Next refresh in ${RefreshSeconds}s..." -ForegroundColor DarkGray

    Start-Sleep -Seconds $RefreshSeconds
}
