# Semantic Code Indexer MCP Server

This MCP (Model Context Protocol) server exposes the Semantic Code Indexer capabilities to Claude Desktop, allowing you to analyze and search Python codebases directly from Claude conversations.

## 🚀 Quick Setup

### 1. Install MCP Dependencies

```bash
cd /Users/nwakrat/myprojects/misc/mcp_server
pip install -r requirements.txt
```

### 2. Configure Claude Desktop

Add this configuration to your Claude Desktop settings:

**Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "semantic-code-indexer": {
      "command": "python",
      "args": ["/Users/nwakrat/myprojects/misc/mcp_server/server.py"],
      "env": {
        "PYTHONPATH": "/Users/nwakrat/myprojects/misc"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

Close and reopen Claude Desktop to load the new MCP server.

## 🛠️ Available Tools

Once connected, Claude will have access to these tools:

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
- **domain**: Filter by business domain (optional)

### 🤖 `ask_agent`
Ask natural language questions about your code
- **question**: Any question about your codebase
- **format**: natural, structured, or json response format

### 📊 `get_code_analytics`
Generate comprehensive codebase analytics
- **include_quality**: Include code quality metrics
- **include_security**: Include security analysis
- **include_architecture**: Include architectural patterns

### 🔄 `find_similar_code`
Find code similar to a specific function or class
- **entity_name**: Name of the function/class to find similar code for
- **similarity_threshold**: Minimum similarity score (0.0-1.0)
- **max_results**: Maximum results to return

### 📋 `get_entity_details`
Get detailed information about a specific code entity
- **entity_name**: Name of the function, class, or entity
- **include_source**: Include full source code
- **include_dependencies**: Include dependency information

## 💬 Example Conversations with Claude

Once set up, you can have conversations like:

**You**: "Can you analyze my FastAPI project at ~/my-project?"

**Claude**: I'll help you analyze your FastAPI project! Let me start by indexing the codebase.

*[Claude uses the `index_codebase` tool]*

**You**: "What are the most complex functions in my codebase?"

**Claude**: Let me get the code analytics for you.

*[Claude uses `get_code_analytics` tool]*

**You**: "Find all authentication-related code"

**Claude**: I'll search for authentication-related code in your project.

*[Claude uses `search_code` tool with query "authentication"]*

**You**: "Show me functions similar to `login_user`"

**Claude**: I'll find functions similar to `login_user` for you.

*[Claude uses `find_similar_code` tool]*

## 🔧 Advanced Configuration

### Using Real ML Embeddings

For production-quality semantic search, install ML dependencies:

```bash
pip install sentence-transformers torch transformers
```

Then use `use_real_embeddings: true` when indexing:

**You**: "Index my codebase with high-quality embeddings"

**Claude**: *[Uses `index_codebase` with `use_real_embeddings: true`]*

### Custom Index Locations

You can maintain multiple project indexes:

```bash
# Different projects
~/code_indexes/project_a/
~/code_indexes/project_b/
```

## 🚨 Troubleshooting

### MCP Server Not Loading
1. Check Claude Desktop logs: `~/Library/Logs/Claude/mcp.log`
2. Verify Python path is correct in config
3. Ensure all dependencies are installed
4. Restart Claude Desktop

### Import Errors
If you see import errors, ensure the PYTHONPATH is set correctly in the config and all dependencies are installed in the same Python environment.

### Permission Issues
Make sure the MCP server script has execute permissions:
```bash
chmod +x /Users/nwakrat/myprojects/misc/mcp_server/server.py
```

## 🎯 Use Cases

### Code Review & Analysis
- "Analyze the code quality of this project"
- "Find potential security issues"
- "What are the most complex parts of the codebase?"

### Codebase Exploration
- "What does this codebase do?"
- "Find all database-related functions"
- "Show me the authentication flow"

### Development Support
- "Find functions similar to this one"
- "What are the dependencies of this module?"
- "Help me understand this function"

### Architecture Documentation
- "Generate a summary of the architectural patterns used"
- "What are the main business domains in this code?"
- "Show me the team ownership structure"

This MCP server turns Claude Desktop into a powerful code analysis assistant! 🎉
