#!/bin/bash

# Start
echo "🏎️ Updating system..."
echo

# Update brew
echo "🍺 Updating Homebrew formulas and casks"
echo
brew update
brew upgrade
brew upgrade --cask
echo

# Update pipx packages
echo "📦 Updating pipx packages"
echo

# Get the list of installed pipx packages
pipx_json_data=$(pipx list --json)
pipx_packages=$(echo "$pipx_json_data" | jq -r '.venvs | keys[]' | sort -u)

# Loop through each package and upgrade it
for pipx_package in $pipx_packages; do
  echo "Upgrading $pipx_package..."
  pipx upgrade "$pipx_package"
  echo
done
echo

# All done.
echo "🌟 All done."
echo
