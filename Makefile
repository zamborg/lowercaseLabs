.PHONY: cleanup release build check install-deps test help

# Default target
help:
	@echo "Available targets:"
	@echo "  cleanup      - Clean up build artifacts and dist files"
	@echo "  build        - Build the package"
	@echo "  check        - Check the built package"
	@echo "  release      - Build and publish to PyPI (interactive)"
	@echo "  install-deps - Install build dependencies"
	@echo "  test         - Run the test suite with pytest"

# Clean up build artifacts
cleanup:
	@echo "🧹 Cleaning up build artifacts..."
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Cleanup complete"

# Install build dependencies
install-deps:
	@echo "📦 Installing build dependencies..."
	pip install --upgrade build twine

# Build the package
build: cleanup install-deps test
	@echo "🔨 Building package..."
	python -m build

# Check the built package
check: build
	@echo "🔍 Checking built package..."
	python -m twine check dist/*

# Run tests
test:
	@echo "🧪 Running tests with pytest..."
	python -m pytest tests/ -v --tb=short

# Release target - build, check, and publish
release: check
	@echo "📤 Ready to upload to PyPI..."
	@echo "Note: You'll need to have your PyPI credentials configured"
	@echo "Use 'twine configure' or set TWINE_USERNAME and TWINE_PASSWORD environment variables"
	python -m  twine upload dist/*
	@echo "✅ Package uploaded successfully!";
	make cleanup