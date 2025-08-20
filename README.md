# Semantic Code Indexing Pipeline

A comprehensive semantic analysis and indexing system for Python codebases that extracts semantic information, generates embeddings, and builds a knowledge graph for intelligent code search and analysis.

## 🚀 Features

- **AST Analysis**: Extract semantic entities (functions, classes, methods) and relationships
- **Embedding Generation**: Create vector embeddings using multiple approaches (text, structure, context)
- **Knowledge Graph**: Build a graph database with entities and relationships
- **AI Agent Interface**: Natural language querying of your codebase
- **Contextual Search**: Business domain and team-aware code search
- **Analytics**: Comprehensive reporting on code quality, complexity, and structure

## 📦 Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd misc

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Index a Codebase
```bash
python semantic_pipeline.py --codebase /path/to/your/code --output ./my_index
```

### 2. Ask Questions About Your Code
```bash
# Get an overview
python semantic_pipeline.py --ask "What does this code do?" --index ./my_index

# Find specific functions
python semantic_pipeline.py --ask "Find authentication functions" --index ./my_index

# Analyze security
python semantic_pipeline.py --ask "Show me security-critical code" --index ./my_index

# Check complexity
python semantic_pipeline.py --ask "What are the most complex functions?" --index ./my_index
```

### 3. Generate Reports
```bash
python semantic_pipeline.py --report --index ./my_index --analytics
```

## 🔧 Configuration

The pipeline works out of the box with intelligent defaults. For advanced usage, you can create custom configuration files:

### Create Custom Configuration
```bash
python semantic_pipeline.py --create-config --context-config ./my_context_config
```

This creates configuration files for:
- Business domains (`business_domains.yaml`)
- Team ownership (`teams.yaml`) 
- Architectural patterns (`architectural_patterns.yaml`)
- Code annotations (`annotations.yaml`)

### 2. Ask Questions About Your Code
```bash
# Get an overview
python semantic_pipeline.py --ask "What does this code do?" --index ./my_index

# Find specific functions
python semantic_pipeline.py --ask "Find authentication functions" --index ./my_index

# Analyze security
python semantic_pipeline.py --ask "Show me security-critical code" --index ./my_index

# Check complexity
python semantic_pipeline.py --ask "What are the most complex functions?" --index ./my_index
```

### 3. Generate Reports
```bash
python semantic_pipeline.py --report --index ./my_index --analytics
```

## 🔧 Configuration

The pipeline automatically creates default configuration when needed. For custom configuration:

```bash
# Create custom configuration files
python semantic_pipeline.py --create-config --context-config ./my_context
```

This creates configuration files for:
- Business domains (`business_domains.yaml`)
- Team ownership (`teams.yaml`) 
- Architectural patterns (`architectural_patterns.yaml`)
- Code annotations (`annotations.yaml`)

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

### Running Tests
```bash
# Create some sample code to analyze
mkdir sample_code
echo 'def hello_world(): """Greets the world.""" print("Hello, World!")' > sample_code/main.py

# Index the sample code
python semantic_pipeline.py --codebase ./sample_code --output ./test_index

# Test AI agent
python semantic_pipeline.py --ask "What does this code do?" --index ./test_index
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source. License details to be determined.

## 🙏 Acknowledgments

- Built with Python's AST module for code analysis
- Uses SQLite for efficient knowledge graph storage
- Inspired by modern code intelligence tools

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub.
