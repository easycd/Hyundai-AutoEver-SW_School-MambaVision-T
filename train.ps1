# Run from PowerShell without activating a Linux shell.
$ErrorActionPreference = 'Stop'
wsl -d Ubuntu-22.04 -u root --cd /mnt/c/Project/Model -- /opt/hand-mamba/bin/python train_mambavision.py @args
exit $LASTEXITCODE
