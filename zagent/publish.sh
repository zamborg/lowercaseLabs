#!/bin/bash

# zagent PyPI Publishing Script
# This script builds and publishes the zagent package to PyPI
# 
# NOTE: This script is now deprecated in favor of the Makefile.
# Use 'make release' instead for a better experience.

set -e  # Exit on any error

echo "⚠️  DEPRECATED: This script is deprecated."
echo "🔄 Please use the Makefile instead:"
echo "   make release    # Full release process"
echo "   make cleanup    # Clean build artifacts"
echo "   make build      # Build package"
echo "   make help       # Show all options"
echo ""

read -p "Do you want to continue with this script anyway? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "👍 Great! Run 'make release' instead."
    exit 0
fi

echo "🚀 Publishing zagent to PyPI (using legacy script)"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Make sure you're in the zagent directory."
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade build twine

# Build the package
echo "🔨 Building package..."
python -m build

# Check the built package
echo "🔍 Checking built package..."
python -m twine check dist/*

# Upload to PyPI
echo "📤 Uploading to PyPI..."
echo "Note: You'll need to have your PyPI credentials configured"
echo "Use 'twine configure' or set TWINE_USERNAME and TWINE_PASSWORD environment variables"

read -p "Do you want to upload to PyPI now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m twine upload dist/*
    echo "✅ Package uploaded successfully!"
else
    echo "📋 Package built successfully. You can upload later with:"
    echo "   python -m twine upload dist/*"
fi

echo "🎉 Done!"