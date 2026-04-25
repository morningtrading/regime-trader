# setup_aliases.ps1 - installe menummh et menutr dans le profil PowerShell
# Lance en PowerShell (pas besoin d'admin)

$menuDir = $PSScriptRoot

# Convertir le chemin Windows en chemin bash (C:\ -> /c/)
$menuDirBash = $menuDir -replace '\\', '/'
$menuDirBash = $menuDirBash -replace '^([A-Z]):', { '/'+$args[0].ToLower() }

# Creer le profil s'il n'existe pas
if (-not (Test-Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}

$content = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
if (-not $content) { $content = "" }

# Supprimer les anciennes entrees
$content = $content -replace '(?m)^function menummh \{[^\}]*\}\r?\n?', ''
$content = $content -replace '(?m)^function menutr \{[^\}]*\}\r?\n?', ''
$content | Set-Content $PROFILE -Encoding UTF8

# Ajouter les nouvelles fonctions
$additions = @"

# regime-trader aliases (auto-installe par setup_aliases.ps1)
function menummh { bash "$menuDirBash/regime_trader.sh" }
function menutr  { bash "$menuDirBash/tailscale_transfer.sh" }
"@

Add-Content -Path $PROFILE -Value $additions -Encoding UTF8

Write-Host "Aliases installes dans $PROFILE" -ForegroundColor Green
Write-Host "Lance: . `$PROFILE" -ForegroundColor Yellow
