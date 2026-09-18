$ErrorActionPreference = 'Stop'
Start-Transcript -Path 'C:\Project\Model\wsl_setup.log' -Force
try {
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All -NoRestart
} finally {
    Stop-Transcript
}
