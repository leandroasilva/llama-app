#!/bin/bash
# setup-secrets.sh - Helper script to configure GitHub Actions secrets for Llama app
# Run this script and follow the prompts

set -e

echo "=== Llama App - GitHub Actions Secrets Setup ==="
echo ""
echo "This script will help you configure the Apple Developer secrets needed for CI."
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it with: brew install gh"
    exit 1
fi

# Check if logged in
if ! gh auth status &> /dev/null; then
    echo "Error: Not logged in to GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

echo "Step 1: Export your Developer ID certificate from Keychain Access"
echo "-------------------------------------------------------------------"
echo "1. Open Keychain Access (Spotlight: Keychain Access)"
echo "2. Select 'login' keychain and 'Certificates' category"
echo "3. Find 'Developer ID Application: Your Name (TEAMID)'"
echo "4. Right-click → Export 'Developer ID Application...'"
echo "5. Save as 'certificate.p12' and set a password"
echo ""
read -p "Press Enter when you have the .p12 file ready..."

echo ""
echo "Step 2: Encode the certificate"
echo "-------------------------------"
read -p "Enter the path to your certificate.p12 file: " CERT_PATH

if [ ! -f "$CERT_PATH" ]; then
    echo "Error: File not found: $CERT_PATH"
    exit 1
fi

CERT_BASE64=$(base64 -i "$CERT_PATH" | tr -d '\n')
echo "✓ Certificate encoded ($(echo -n "$CERT_BASE64" | wc -c | tr -d ' ') bytes)"

echo ""
echo "Step 3: Get your Team ID"
echo "-------------------------"
echo "Find your Team ID at: https://developer.apple.com/account → Membership"
echo "It's a 10-character code like: A1B2C3D4E5"
echo ""
read -p "Enter your Team ID: " TEAM_ID

if [ ${#TEAM_ID} -ne 10 ]; then
    echo "Warning: Team ID should be 10 characters. You entered: ${#TEAM_ID}"
fi

echo ""
echo "Step 4: Apple ID for notarization"
echo "----------------------------------"
read -p "Enter your Apple ID email: " APPLE_ID

echo ""
echo "Step 5: App-Specific Password"
echo "------------------------------"
echo "Generate one at: https://appleid.apple.com → Sign-In and Security → App-Specific Passwords"
echo ""
read -p "Enter your app-specific password: " APPLE_PASSWORD

echo ""
echo "Step 6: Certificate password"
echo "-----------------------------"
read -s -p "Enter the password you set for the .p12 file: " CERT_PASSWORD
echo ""

echo ""
echo "=== Setting secrets ==="
echo ""

gh secret set APPLE_CERTIFICATE --body "$CERT_BASE64"
echo "✓ APPLE_CERTIFICATE set"

gh secret set APPLE_CERTIFICATE_PASSWORD --body "$CERT_PASSWORD"
echo "✓ APPLE_CERTIFICATE_PASSWORD set"

gh secret set APPLE_TEAM_ID --body "$TEAM_ID"
echo "✓ APPLE_TEAM_ID set"

gh secret set APPLE_ID --body "$APPLE_ID"
echo "✓ APPLE_ID set"

gh secret set APPLE_ID_PASSWORD --body "$APPLE_PASSWORD"
echo "✓ APPLE_ID_PASSWORD set"

echo ""
echo "=== All secrets configured! ==="
echo ""
echo "The next push to main will trigger the release workflow."
echo "View your secrets at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/settings/secrets/actions"
echo ""
echo "To trigger a test build, you can push an empty commit:"
echo "  git commit --allow-empty -m 'ci: trigger release' && git push"
