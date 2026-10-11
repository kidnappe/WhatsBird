$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($p in $procs) {
    if ($p.CommandLine -like '*national_pipeline*' -or $p.CommandLine -like '*build_dataset_gbif*') {
        Write-Output ("killing " + $p.ProcessId + " : " + $p.CommandLine.Substring(0, [Math]::Min(80, $p.CommandLine.Length)))
        Stop-Process -Id $p.ProcessId -Force
    }
}
Write-Output "done"
