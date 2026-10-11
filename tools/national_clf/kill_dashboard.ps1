$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($p in $procs) {
    if ($p.CommandLine -like '*train_dashboard*') {
        Write-Output ("killing dashboard " + $p.ProcessId)
        Stop-Process -Id $p.ProcessId -Force
    }
}
Write-Output "done"
