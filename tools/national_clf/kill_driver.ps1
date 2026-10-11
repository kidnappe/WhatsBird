$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($p in $procs) {
    if ($p.CommandLine -like '*national_pipeline_v2*') {
        Write-Output ("killing driver " + $p.ProcessId)
        Stop-Process -Id $p.ProcessId -Force
    }
}
Write-Output "done"
