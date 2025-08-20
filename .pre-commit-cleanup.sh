#!/bin/bash
# Pre-commit hook to clean up before commits

echo "🧹 Running pre-commit cleanup..."

# Remove Python cache files
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove build artifacts
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

# Remove temporary files
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.log" -delete 2>/dev/null || true

# Remove test outputs (but preserve test_code/)
rm -rf test_output/ contextual_index/ code_index/ test_index/ 2>/dev/null || true

echo "✅ Cleanup completed"
