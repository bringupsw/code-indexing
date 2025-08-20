# Code Indexer

A semantic code analysis and indexing pipeline for Python codebases.

## Structure

The code has been organized into a clean modular architecture:

```
src/code_indexer/
├── apps/           # Application entry points
│   └── cli/        # Command line interface
│       ├── __init__.py
│       ├── __main__.py
│       └── main.py
├── libs/           # Shared libraries
│   └── common/     # Common components
│       ├── __init__.py
│       ├── ai_agent.py          # Natural language query interface
│       ├── ast_analyzer.py      # AST parsing and semantic extraction
│       ├── contextual_knowledge.py  # Business domain classification
│       ├── embeddings.py        # Vector embedding generation
│       ├── knowledge_graph.py   # SQLite-based knowledge storage
│       ├── models.py           # Core data structures
│       └── pipeline.py         # Main orchestrator
├── __init__.py
└── __main__.py
```

## Usage

### Running the CLI

You can run the code indexer in several ways:

```bash
# Run as a module
python -m code_indexer --help

# Run the CLI directly  
python -m code_indexer.apps.cli --help

# Index a codebase
python -m code_indexer --codebase /path/to/codebase --output ./index

# Search the index
python -m code_indexer --search "authentication" --index ./index

# Ask the AI agent questions
python -m code_indexer --ask "Find all security-critical functions" --index ./index

# Generate reports
python -m code_indexer --report --index ./index --analytics
```

### Creating configuration files

```bash
# Create default configuration files for contextual analysis
python -m code_indexer --create-config --context-config ./my_context
```

## Components

### Core Libraries (`libs/common/`)

- **`models.py`**: Data structures for entities, relations, and query results
- **`contextual_knowledge.py`**: Business domain and team classification logic  
- **`ast_analyzer.py`**: Python AST parsing and semantic entity extraction
- **`embeddings.py`**: Multi-approach vector embedding generation
- **`knowledge_graph.py`**: SQLite-based storage with semantic search
- **`ai_agent.py`**: Natural language query processing
- **`pipeline.py`**: Main orchestrator that coordinates all components

### Applications (`apps/`)

- **`cli/`**: Command-line interface with comprehensive argument parsing

## Key Features

- **Semantic Analysis**: Extracts functions, classes, methods, and variables with metadata
- **Contextual Classification**: Automatically categorizes code by business domain and team
- **Vector Embeddings**: Generates searchable embeddings using multiple approaches
- **Knowledge Graph**: Stores entities and relationships in a queryable database
- **Natural Language Queries**: AI agent interface for asking questions about the codebase
- **Multiple Search Types**: Semantic, graph-based, hybrid, and contextual search
- **Analytics**: Generates comprehensive reports about codebase structure and complexity

## Dependencies

The system relies on:
- `ast` (Python standard library)
- `sqlite3` (Python standard library) 
- `sentence-transformers` (for embeddings)
- `numpy` (for vector operations)
- `yaml` (for configuration files)

Each module is designed to be:
- **Independent**: Can be used separately or together
- **Extensible**: Easy to add new features and analysis types
- **Scalable**: Supports parallel processing for large codebases
- **Persistent**: Stores results for reuse and incremental updates
