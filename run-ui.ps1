$ErrorActionPreference = 'Stop'
$projectPath = '/mnt/c/Project/Model'
$pythonPath = '/opt/hand-mamba/bin/python'

Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    '-NoProfile',
    '-Command',
    'Start-Sleep -Seconds 3; Start-Process http://localhost:8501'
)

wsl -d Ubuntu-22.04 -u root --cd $projectPath -- $pythonPath -m streamlit run app.py `
    --server.address 0.0.0.0 `
    --server.port 8501 `
    --server.headless true
exit $LASTEXITCODE
