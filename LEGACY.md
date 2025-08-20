# Legacy Files

This directory contains the original monolithic implementation that has been refactored into the modular structure in `src/code_indexer/`.

## Files

- `semantic_pipeline.py` - Original 3,795-line monolithic implementation
- `requirements.txt` - Legacy requirements file (now use pyproject.toml)

## Migration

The functionality has been completely migrated to the new modular structure:
- **From**: `semantic_pipeline.py` (monolithic)
- **To**: `src/code_indexer/` (modular package)

Use the new package instead:
```bash
# Old way
python semantic_pipeline.py --codebase ./src --output ./index

# New way  
code-indexer --codebase ./src --output ./index
# or
python -m code_indexer.apps.cli --codebase ./src --output ./index
```

The old file is kept for reference and comparison purposes.
