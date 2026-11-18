# Install cloudflared on Windows using winget.
$ErrorActionPreference = "Stop"
if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id Cloudflare.cloudflared --accept-source-agreements --accept-package-agreements
    cloudflared --version
} else {
    Write-Host "Install cloudflared from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    exit 1
}
