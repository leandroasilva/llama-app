param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$OutputDir = "build/windows"
)

$ErrorActionPreference = "Stop"

$Version = $Version.TrimStart('v')
$name = "llama-gtk-$Version-windows-x86_64"
$outputRoot = (New-Item -ItemType Directory -Force -Path $OutputDir).FullName
$zip = Join-Path $outputRoot "$name.zip"
if (-not (Test-Path $zip)) {
    throw "missing portable package: $zip"
}

$tempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
$extractRoot = Join-Path $tempRoot "llama-gtk-installer-src"
if (Test-Path $extractRoot) {
    Remove-Item -Recurse -Force $extractRoot
}
Expand-Archive -Path $zip -DestinationPath $extractRoot -Force

$bundle = Join-Path $extractRoot $name
if (-not (Test-Path $bundle)) {
    throw "missing extracted bundle: $bundle"
}

$iscc = $null
if (${env:ProgramFiles(x86)}) {
    $iscc = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"
}
if (-not $iscc -or -not (Test-Path $iscc)) {
    $command = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($command) { $iscc = $command.Source }
}
if (-not $iscc -or -not (Test-Path $iscc)) {
    throw "Inno Setup compiler (ISCC.exe) not found; install it with 'choco install innosetup'"
}

$env:LLAMA_VERSION = $Version
$env:LLAMA_SOURCE_DIR = (Resolve-Path $bundle).Path
$env:LLAMA_OUTPUT_DIR = $outputRoot

& $iscc (Join-Path $PSScriptRoot "installer.iss")
if ($LASTEXITCODE -ne 0) {
    throw "ISCC failed with exit code $LASTEXITCODE"
}

$setup = Join-Path $outputRoot "$name-setup.exe"
if (-not (Test-Path $setup)) {
    throw "installer was not produced: $setup"
}

$hash = (Get-FileHash -Algorithm SHA256 -Path $setup).Hash.ToLowerInvariant()
Set-Content -Path "${setup}.sha256" -Value "$hash  ${name}-setup.exe" -Encoding ascii
Write-Host "Windows installer written to $setup"
