#!/bin/bash
# export-cert.sh - Export Developer ID certificate as .p12
# Run this script in your terminal - it will prompt for your keychain password

set -e

echo "=== Export Developer ID Certificate ==="
echo ""
echo "This script will export your Developer ID certificate as a .p12 file."
echo "You'll be prompted for your keychain password (or Touch ID)."
echo ""

OUTPUT="/tmp/llama-certificate.p12"
PASSWORD="Llama2026!"

echo "Exporting certificate..."
security export \
  -k ~/Library/Keychains/login.keychain-db \
  -t identities \
  -f pkcs12 \
  -P "$PASSWORD" \
  -o "$OUTPUT"

echo ""
echo "✓ Certificate exported to: $OUTPUT"
echo "✓ Password: $PASSWORD"
echo ""
echo "Now run these commands to set the GitHub secrets:"
echo ""
echo "  base64 -i $OUTPUT | tr -d '\\n' | gh secret set APPLE_CERTIFICATE"
echo "  gh secret set APPLE_CERTIFICATE_PASSWORD --body \"$PASSWORD\""
echo ""
echo "Or manually at: https://github.com/leandroasilva/llama-app/settings/secrets/actions"
