[CmdletBinding()]
param(
    [switch]$TrainerOnly,
    [switch]$SkipTrainer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if ($TrainerOnly -and $SkipTrainer) {
    throw '-TrainerOnly and -SkipTrainer cannot be used together.'
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python312Version = '3.12.10'
$PipVersion = '26.2.1'
$HatchlingVersion = '1.32.0'
$PythonInstallerUrl = "https://www.python.org/ftp/python/$Python312Version/python-$Python312Version-amd64.exe"
$PythonInstallerSha256 = '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'
$PipWheelUrl = 'https://files.pythonhosted.org/packages/f3/6e/1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/pip-26.2.1-py3-none-any.whl'
$PipWheelSha256 = '71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e'

Set-Location $ProjectRoot

function Write-Section {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host ''
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host "> $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath"
    }
}

function Test-Python312 {
    param([Parameter(Mandatory = $true)][string]$FilePath)

    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        return $false
    }

    try {
        $Version = & $FilePath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return (($LASTEXITCODE -eq 0) -and ($Version.Trim() -eq '3.12'))
    }
    catch {
        return $false
    }
}

function Find-Python312 {
    if ($env:QDM_PYTHON312 -and (Test-Python312 $env:QDM_PYTHON312)) {
        return (Resolve-Path -LiteralPath $env:QDM_PYTHON312).Path
    }

    $KnownPaths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    if (${env:ProgramFiles(x86)}) {
        $KnownPaths += (Join-Path ${env:ProgramFiles(x86)} 'Python312\python.exe')
    }

    foreach ($Candidate in $KnownPaths) {
        if (Test-Python312 $Candidate) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }

    $PythonCommand = Get-Command python3.12.exe -ErrorAction SilentlyContinue
    if ($PythonCommand -and (Test-Python312 $PythonCommand.Source)) {
        return $PythonCommand.Source
    }

    $RegistryPaths = @(
        'HKCU:\Software\Python\PythonCore\3.12\InstallPath',
        'HKLM:\Software\Python\PythonCore\3.12\InstallPath',
        'HKLM:\Software\WOW6432Node\Python\PythonCore\3.12\InstallPath'
    )
    foreach ($RegistryPath in $RegistryPaths) {
        if (Test-Path $RegistryPath) {
            $InstallRoot = (Get-Item $RegistryPath).GetValue('')
            if ($InstallRoot) {
                $Candidate = Join-Path $InstallRoot 'python.exe'
                if (Test-Python312 $Candidate) {
                    return (Resolve-Path -LiteralPath $Candidate).Path
                }
            }
        }
    }

    $Launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Launcher) {
        try {
            $Candidate = & $Launcher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
            if (($LASTEXITCODE -eq 0) -and (Test-Python312 $Candidate.Trim())) {
                return $Candidate.Trim()
            }
        }
        catch {
            # Some Microsoft Store launchers do not discover a normal per-user install.
        }
    }

    return $null
}

function Install-Python312 {
    Write-Section "Installing Python $Python312Version for the current user"

    if (-not [Environment]::Is64BitOperatingSystem) {
        throw 'The CUDA trainer requires 64-bit Windows.'
    }

    $InstallerPath = Join-Path ([System.IO.Path]::GetTempPath()) "qdm-python-$Python312Version-$PID.exe"
    try {
        Write-Host "Downloading $PythonInstallerUrl"
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $PythonInstallerUrl -OutFile $InstallerPath
        }
        catch {
            # Some Windows certificate stores fail with SEC_E_NO_CREDENTIALS.
            # Prefer any existing Python's independent TLS stack as a fallback.
            Write-Warning 'PowerShell HTTPS download failed; trying an alternate downloader.'
            $DownloadPython = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($DownloadPython) {
                Invoke-Native $DownloadPython.Source @(
                    '-c',
                    'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])',
                    $PythonInstallerUrl,
                    $InstallerPath
                )
            }
            else {
                # On a machine without Python, curl may fetch without Schannel
                # verification only here; the official Sigstore SHA-256 and
                # Authenticode signature are verified before execution.
                $Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
                if (-not $Curl) {
                    throw 'Python download failed and no alternate downloader is available.'
                }
                Invoke-Native $Curl.Source @(
                    '--fail', '--location', '--insecure',
                    '--silent', '--show-error',
                    '--output', $InstallerPath,
                    $PythonInstallerUrl
                )
            }
        }

        $ActualHash = (Get-FileHash -LiteralPath $InstallerPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $PythonInstallerSha256) {
            throw "Python installer checksum mismatch. Expected $PythonInstallerSha256, got $ActualHash."
        }

        $Signature = Get-AuthenticodeSignature -FilePath $InstallerPath
        if (($Signature.Status -ne 'Valid') -or ($Signature.SignerCertificate.Subject -notmatch 'Python Software Foundation')) {
            throw "Python installer signature validation failed: $($Signature.Status)"
        }

        $Install = Start-Process -FilePath $InstallerPath -ArgumentList @(
            '/quiet',
            'InstallAllUsers=0',
            'Include_launcher=1',
            'InstallLauncherAllUsers=0',
            'Include_pip=1',
            'Include_test=0',
            'PrependPath=0'
        ) -Wait -PassThru -WindowStyle Hidden

        if ($Install.ExitCode -ne 0) {
            throw "Python installer failed with exit code $($Install.ExitCode)."
        }
    }
    finally {
        Remove-Item -LiteralPath $InstallerPath -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-Venv {
    param(
        [Parameter(Mandatory = $true)][string]$BootstrapPython,
        [Parameter(Mandatory = $true)][string]$VenvPath
    )

    $VenvPython = Join-Path $VenvPath 'Scripts\python.exe'
    if (Test-Python312 $VenvPython) {
        Write-Host "Reusing Python 3.12 environment: $VenvPath"
        return $VenvPython
    }

    if (Test-Path -LiteralPath $VenvPath) {
        Write-Host "Recreating incompatible or incomplete environment: $VenvPath"
        Invoke-Native $BootstrapPython @('-m', 'venv', '--clear', $VenvPath)
    }
    else {
        Write-Host "Creating environment: $VenvPath"
        Invoke-Native $BootstrapPython @('-m', 'venv', $VenvPath)
    }

    if (-not (Test-Python312 $VenvPython)) {
        throw "Virtual environment was not created correctly: $VenvPath"
    }
    return $VenvPython
}

function Ensure-Pip {
    param([Parameter(Mandatory = $true)][string]$Python)

    $InstalledVersion = & $Python -c "import pip; print(pip.__version__)"
    if (($LASTEXITCODE -eq 0) -and ($InstalledVersion.Trim() -eq $PipVersion)) {
        Write-Host "pip $PipVersion is already installed."
        return
    }

    # Do not use the bundled pip 25.0.1 for network access: on some Windows
    # systems it stalls before dependency resolution. Python's HTTPS client
    # downloads a pinned wheel, which is verified before local installation.
    $WheelDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "qdm-pip-$PID-$([Guid]::NewGuid().ToString('N'))"
    $WheelPath = Join-Path $WheelDirectory "pip-$PipVersion-py3-none-any.whl"
    New-Item -ItemType Directory -Path $WheelDirectory | Out-Null
    try {
        Invoke-Native $Python @(
            '-c',
            'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])',
            $PipWheelUrl,
            $WheelPath
        )
        $ActualHash = (Get-FileHash -LiteralPath $WheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $PipWheelSha256) {
            throw "pip wheel checksum mismatch. Expected $PipWheelSha256, got $ActualHash."
        }
        Invoke-Native $Python @(
            '-m', 'pip', 'install',
            '--disable-pip-version-check',
            '--no-deps',
            '--force-reinstall',
            $WheelPath
        )
    }
    finally {
        Remove-Item -LiteralPath $WheelDirectory -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Install-Application {
    param([Parameter(Mandatory = $true)][string]$BootstrapPython)

    Write-Section 'Installing Qwen Dataset Manager'
    $ApplicationPython = Ensure-Venv $BootstrapPython (Join-Path $ProjectRoot '.venv')
    Ensure-Pip $ApplicationPython
    Invoke-Native $ApplicationPython @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-cache-dir',
        '-r', (Join-Path $ProjectRoot 'requirements.txt')
    )
    Invoke-Native $ApplicationPython @('-m', 'pip', 'check')
    Invoke-Native $ApplicationPython @('-c', "import flask, PIL, app; print('Application import check passed.')")
}

function Install-Trainer {
    param([Parameter(Mandatory = $true)][string]$BootstrapPython)

    Write-Section 'Installing CUDA trainer'

    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        throw 'Git is required because AI Toolkit pins a diffusers Git revision. Install Git for Windows and rerun this installer.'
    }
    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        throw 'An NVIDIA GPU and current NVIDIA driver are required. Use -SkipTrainer to install only the dataset manager.'
    }

    & nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        throw 'nvidia-smi failed. Update or reinstall the NVIDIA driver.'
    }

    # Windows PowerShell 5.1 can return a null Get-PSDrive.Free value for a
    # perfectly healthy filesystem, which compares as zero. DriveInfo reports
    # the real filesystem capacity consistently across PowerShell versions.
    $ProjectDrive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($ProjectRoot))
    $FreeSpaceGB = [math]::Round($ProjectDrive.AvailableFreeSpace / 1GB, 2)
    Write-Host "Free space on $($ProjectDrive.Name): $FreeSpaceGB GB"
    if ($ProjectDrive.AvailableFreeSpace -lt 10GB) {
        Write-Warning 'Less than 10 GB is free on the project drive. CUDA dependencies and training outputs need substantial disk space.'
    }

    $TrainerPython = Ensure-Venv $BootstrapPython (Join-Path $ProjectRoot 'trainer\.venv')
    Ensure-Pip $TrainerPython

    Invoke-Native $TrainerPython @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-cache-dir',
        'torch==2.13.0',
        'torchvision==0.28.0',
        'torchaudio==2.11.0',
        '--index-url', 'https://download.pytorch.org/whl/cu130'
    )

    # Installing the build backend first and disabling build isolation avoids a
    # reproducible hang while pip prepares the pinned diffusers revision.
    Invoke-Native $TrainerPython @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-cache-dir',
        "hatchling==$HatchlingVersion"
    )
    Invoke-Native $TrainerPython @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-cache-dir',
        '--no-build-isolation',
        '-r', (Join-Path $ProjectRoot 'trainer\ai_toolkit\requirements.txt')
    )

    Invoke-Native $TrainerPython @('-m', 'pip', 'check')
    Invoke-Native $TrainerPython @(
        '-c',
        "import torch, diffusers, transformers, bitsandbytes, peft; assert torch.cuda.is_available(), 'PyTorch installed, but CUDA is not available'; print('CUDA trainer ready:', torch.__version__, torch.cuda.get_device_name(0))"
    )
    Invoke-Native $TrainerPython @((Join-Path $ProjectRoot 'trainer\ai_toolkit\run.py'), '--help')
}

try {
    Write-Host 'Qwen Dataset Manager full installer' -ForegroundColor Green
    Write-Host "Project: $ProjectRoot"

    $Python312 = Find-Python312
    if (-not $Python312) {
        Install-Python312
        $Python312 = Find-Python312
    }
    if (-not $Python312) {
        throw "Python $Python312Version was installed but could not be located. Set QDM_PYTHON312 to python.exe and rerun."
    }
    Write-Host "Using Python 3.12: $Python312"

    if (-not $TrainerOnly) {
        Install-Application $Python312
    }
    if (-not $SkipTrainer) {
        Install-Trainer $Python312
    }

    Write-Section 'Installation complete'
    if ($TrainerOnly) {
        Write-Host 'CUDA trainer is ready. Restart Qwen Dataset Manager.' -ForegroundColor Green
    }
    else {
        Write-Host 'Run run.cmd, then open http://127.0.0.1:5001' -ForegroundColor Green
    }
}
catch {
    Write-Host ''
    Write-Host "INSTALLATION FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Fix the reported prerequisite/network issue and rerun the installer; completed steps are reused.' -ForegroundColor Yellow
    exit 1
}
