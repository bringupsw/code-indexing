# Semantic Code Indexer

A powerful semantic code analysis and indexing pipeline for Python codebases that extracts, embeds, and indexes code entities for intelligent search and analysis.

## 🚀 Features

- **Semantic Analysis**: Deep AST parsing to extract functions, classes, methods, and variables with rich metadata
- **Contextual Classification**: Automatically categorizes code by business domain, team ownership, and architectural patterns
- **Vector Embeddings**: Multi-approach embedding generation for semantic similarity search
- **Knowledge Graph**: SQLite-based storage with efficient querying and graph traversal
- **Natural Language Queries**: AI-powered interface for asking questions about your codebase
- **Multiple Search Types**: Semantic, graph-based, hybrid, and contextual search capabilities
- **Comprehensive Analytics**: Detailed reports about codebase structure, complexity, and quality metrics

## 📦 Installation

### From PyPI (when published)

```bash
# Minimal installation (basic functionality)
pip install semantic-code-indexer

# With ML models for real embeddings
pip install semantic-code-indexer[ml]

# With enhanced UX features
pip install semantic-code-indexer[enhanced]

# Full installation
pip install semantic-code-indexer[all]
```

### From Source

```bash
git clone https://github.com/bringupsw/code-indexing.git
cd code-indexing
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/bringupsw/code-indexing.git
cd code-indexing

# Basic installation
pip install -e .

# Development installation with all tools
pip install -e ".[dev,test,docs]"

# Full installation with ML models
pip install -e ".[all,dev,test]"
```

## 🎯 Quick Start

### 1. Index a Codebase

```bash
# Index your Python codebase using the new modular structure
code-indexer --codebase /path/to/your/project --output ./index

# Or using the module directly
python -m code_indexer.apps.cli --codebase /path/to/project --output ./index --context-config ./my_context
```

### 2. Search the Index

```bash
# Semantic search
code-indexer --search "authentication login" --index ./index

# Contextual search with filters
code-indexer --search "database" --index ./index --domain "Data" --team "Backend Team"
```

### 3. Ask Natural Language Questions

```bash
# Query the AI agent
code-indexer --ask "What are the most complex functions?" --index ./index
code-indexer --ask "Find all security-critical code" --index ./index
code-indexer --ask "Who owns the payment processing logic?" --index ./index
```

### 4. Generate Reports

```bash
# Comprehensive analytics report
code-indexer --report --index ./index --analytics --save-results report.json
```

## 🔧 Configuration

Create contextual configuration files to customize analysis:

```bash
# Generate default configuration files
code-indexer --create-config --context-config ./my_context
```

This creates:
- `business_domains.yaml` - Define your business domains and keywords
- `teams.yaml` - Configure team ownership patterns
- `architectural_patterns.yaml` - Define architectural patterns to detect
- `annotations.yaml` - Configure quality and security annotations

## 💻 Python API

You can also use the indexer programmatically:

```python
from code_indexer.libs.common import CodebaseSemanticPipeline

# Initialize pipeline
pipeline = CodebaseSemanticPipeline(
    output_dir="./index",
    context_config_dir="./context_config"
)

# Process a codebase
results = pipeline.process_codebase("/path/to/codebase")

# Search the index
search_results = pipeline.search("authentication", search_type="hybrid")

# Query with AI agent
response = pipeline.query_agent("Find all database-related functions")

# Generate analytics
analytics = pipeline.get_contextual_analytics()
```

## � Project Structure

```
src/code_indexer/
├── apps/           # Application entry points
│   └── cli/        # Command line interface
├── libs/           # Shared libraries
│   └── common/     # Common components
│       ├── models.py              # Core data structures
│       ├── contextual_knowledge.py # Business domain classification
│       ├── ast_analyzer.py        # AST parsing and extraction
│       ├── embeddings.py          # Vector embedding generation
│       ├── knowledge_graph.py     # Knowledge storage and search
│       ├── ai_agent.py            # Natural language interface
│       └── pipeline.py            # Main orchestrator
## 📚 Dependencies

### Core Dependencies (Required)
- **numpy**: Vector operations and mathematical computations
- **PyYAML**: Configuration file parsing

### Optional Dependencies
- **[ml]**: `sentence-transformers`, `transformers`, `torch`, `scikit-learn` - For real ML model embeddings
- **[enhanced]**: `tqdm`, `rich` - For progress bars and enhanced CLI experience  
- **[dev]**: `pytest`, `black`, `isort`, `flake8`, `mypy`, `pre-commit` - Development tools
- **[test]**: `pytest`, `pytest-cov`, `pytest-mock` - Testing frameworks
- **[docs]**: `sphinx`, `sphinx-rtd-theme`, `myst-parser` - Documentation generation

The package works out-of-the-box with just the core dependencies using mock embeddings. Install optional groups as needed for enhanced functionality.
```

### Example Configuration

**business_domains.yaml**:
```yaml
Authentication:
  patterns: ["auth", "login", "user", "password"]
  importance: 1.8

Payment:
  patterns: ["payment", "billing", "charge"]
  importance: 2.0
```

## 📖 Usage Examples

### Command Line Interface

```bash
# Basic indexing
python semantic_pipeline.py --codebase ./your_code --output ./index

# Advanced indexing with custom settings
python semantic_pipeline.py --codebase ./your_code --output ./index \
  --workers 8 --exclude-tests

# With custom context configuration  
python semantic_pipeline.py --codebase ./your_code --output ./index \
  --context-config ./my_context

# AI Agent queries
python semantic_pipeline.py --ask "Find all authentication functions" --index ./index
python semantic_pipeline.py --ask "Who owns the payment code?" --index ./index
python semantic_pipeline.py --ask "Show architectural patterns" --index ./index

# Search with filters
python semantic_pipeline.py --search "database" --index ./index \
  --search-type contextual --domain "Data Access"

# Generate comprehensive reports
python semantic_pipeline.py --report --index ./index \
  --analytics --save-results ./report.json
```

### AI Agent Capabilities

The AI agent understands natural language queries about your codebase:

- **Code Overview**: "What does this code do?", "Explain this codebase"
- **Function Finding**: "Find login functions", "Show authentication code"
- **Security Analysis**: "Find security vulnerabilities", "Show sensitive data handling"
- **Team Ownership**: "Who owns this code?", "Show team responsibilities"
- **Architecture**: "What patterns are used?", "Show system architecture"
- **Quality**: "Find complex functions", "Show code quality issues"

## 📊 Architecture

The pipeline consists of four main phases:

1. **AST Analysis**: Extracts semantic entities and relationships using Python's AST
2. **Embedding Generation**: Creates vector embeddings using hybrid approaches
3. **Knowledge Graph**: Builds SQLite-based graph with entities and relationships
4. **AI Interface**: Provides natural language querying capabilities

## 🤖 AI Agent Features

- **Local Processing**: Runs entirely on your machine, no external API calls
- **Intent Detection**: Uses regex patterns to understand user queries
- **Contextual Analysis**: Provides domain-aware and team-aware responses
- **Multiple Formats**: Supports natural language, structured, and JSON output

## 📁 File Structure

```
semantic-pipeline/
├── semantic_pipeline.py      # Main pipeline implementation
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
└── contextual_index/        # Generated knowledge base (created after indexing)
    └── knowledge.db
```

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/bringupsw/code-indexing.git
cd code-indexing

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/code_indexer --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/

# Run all quality checks
pre-commit run --all-files
```

### Building and Publishing

```bash
# Build the package
python -m build

# Check the build
twine check dist/*

# Publish to PyPI (maintainers only)
twine upload dist/*
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Python's AST module for robust code analysis
- Uses sentence-transformers for high-quality embeddings
- SQLite for efficient knowledge graph storage
- Inspired by modern code intelligence tools and semantic search systems

## 📞 Support

- 📖 [Documentation](src/code_indexer/README.md)
- 🐛 [Issue Tracker](https://github.com/bringupsw/code-indexing/issues)
- 💬 [Discussions](https://github.com/bringupsw/code-indexing/discussions)

For questions, issues, or contributions, please open an issue on GitHub.
