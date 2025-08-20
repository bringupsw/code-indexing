# MCP Server for Semantic Code Indexer

This directory contains the MCP (Model Context Protocol) server that exposes the Semantic Code Indexer capabilities to Claude Desktop and other MCP-compatible clients.

## 🚀 Quick Start

### 1. Install the Package

```bash
# Install the semantic code indexer with MCP support
pip install -e .[mcp]

# Or install manually
pip install -e .
pip install mcp
```

### 2. Test the Installation

```bash
# Test the MCP server
code-indexer-mcp --test

# Or run as module
python -m code_indexer.apps.mcp --test
```

### 3. Configure Claude Desktop

```bash
# Run the setup script
python -m code_indexer.apps.mcp.setup_claude

# Or manually configure Claude Desktop
```

Add this to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "semantic-code-indexer": {
      "command": "code-indexer-mcp"
    }
  }
}
```

### 4. Restart Claude Desktop

Close and reopen Claude Desktop to load the MCP server.

## 🛠️ Available Tools

The MCP server provides these tools to Claude:

### 📂 `index_codebase`
Index a Python codebase for analysis
- **codebase_path**: Path to your Python project
- **output_path**: Where to store the index (optional)
- **use_real_embeddings**: Use ML embeddings for better quality (optional)
- **exclude_tests**: Skip test files (optional, default: true)

### 🔍 `search_code`
Search your indexed codebase
- **query**: Search terms (e.g., "authentication functions")
- **search_type**: semantic, graph, hybrid, or contextual
- **top_k**: Number of results (1-50)
- **index_path**: Specific index to search (optional)
- **domain**: Filter by business domain (optional)

### 🤖 `ask_agent`
Ask natural language questions about your code
- **question**: Any question about your codebase
- **index_path**: Specific index to query (optional)
- **format**: natural, structured, or json response format

### 📊 `get_code_analytics`
Generate comprehensive codebase analytics
- **index_path**: Specific index to analyze (optional)
- **include_quality**: Include code quality metrics
- **include_security**: Include security analysis
- **include_architecture**: Include architectural patterns

### 🔄 `find_similar_code`
Find code similar to a specific function or class
- **entity_name**: Name of the function/class to find similar code for
- **index_path**: Specific index to search (optional)
- **similarity_threshold**: Minimum similarity score (0.0-1.0)
- **max_results**: Maximum results to return

### 📋 `get_entity_details`
Get detailed information about a specific code entity
- **entity_name**: Name of the function, class, or entity
- **index_path**: Specific index to search (optional)
- **include_source**: Include full source code
- **include_dependencies**: Include dependency information

### 📂 `list_indexes`
List all available code indexes
- **directory**: Directory to search for indexes (optional)
- Shows all previously created indexes with their paths and metadata

## 💬 Example Usage with Claude

Once configured, you can have conversations like:

**You**: "Can you analyze my FastAPI project at ~/my-project?"

**Claude**: I'll help you analyze your FastAPI project! Let me start by indexing the codebase.

*[Claude uses the `index_codebase` tool]*

**You**: "What are the most complex functions in my codebase?"

**Claude**: Let me get the code analytics for you.

*[Claude uses `get_code_analytics` tool]*

**You**: "Find all authentication-related code"

**Claude**: I'll search for authentication-related code in your project.

*[Claude uses `search_code` tool with query "authentication"]*

## 🔧 Advanced Configuration

### Multiple Projects

You can maintain separate indexes for different projects:

```bash
# Index different projects
# Claude: "Index my API project at ~/projects/api with output ~/indexes/api"
# Claude: "Index my web app at ~/projects/web with output ~/indexes/web"

# Switch between them
# Claude: "Search for login functions in the API project" (specify index_path)
```

### High-Quality Embeddings

For production-quality semantic search:

```bash
# Install ML dependencies
pip install sentence-transformers torch transformers

# Use in indexing
# Claude: "Index my codebase with real embeddings for better quality"
```

### Development vs Production

```bash
# Development (fast)
# Claude: "Index ~/my-project" (uses mock embeddings)

# Production (high-quality)
# Claude: "Index ~/my-project with real ML embeddings"
```

## 🚨 Troubleshooting

### MCP Server Not Loading
1. Check that `code-indexer-mcp --test` works
2. Verify Claude Desktop configuration
3. Check Claude Desktop logs
4. Restart Claude Desktop

### Command Not Found
```bash
# Make sure the package is installed
pip install -e .

# Check if the command is available
which code-indexer-mcp
```

### Import Errors
```bash
# Install all dependencies
pip install -e .[mcp,ml]

# Test imports
python -c "from code_indexer.apps.mcp.main import main; print('OK')"
```

### Permission Issues
Make sure the installed scripts have execute permissions and are in your PATH.

## 📁 Files

- `main.py`: Main MCP server implementation
- `setup_claude.py`: Claude Desktop configuration script
- `requirements.txt`: MCP-specific dependencies
- `__init__.py`: Package initialization
- `__main__.py`: Module entry point
- `README.md`: This file

## 🔗 Related

- [Main CLI app](../cli/): Command-line interface for the code indexer
- [Core libraries](../../libs/): Core semantic analysis functionality
- [Project root](../../../): Main project documentation

This MCP server turns Claude Desktop into a powerful code analysis assistant! 🎉
