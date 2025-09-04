#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Upload to PyPI ---
echo "Uploading the distribution to PyPI..."
python3 -m twine upload dist/*

echo "Upload complete."
