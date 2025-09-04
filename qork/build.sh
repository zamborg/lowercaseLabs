#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Clean up old build artifacts ---
echo "Cleaning up old build artifacts..."
rm -rf dist
rm -rf build
rm -rf *.egg-info

# --- Build the project ---
echo "Building the project..."
python3 -m build

echo "Build complete. The distribution files are in the dist/ directory."
