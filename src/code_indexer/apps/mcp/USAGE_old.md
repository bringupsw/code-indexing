# 🤖 Claude Desktop + Semantic Code Indexer Integration

## What This Does

This MCP (Model Context Protocol) server connects your **Semantic Code Indexer** to **Claude Desktop**, giving Claude the ability to:

- 📂 **Index Python codebases** for semantic analysis
- 🔍 **Search code** using natural language queries  
- 🤖 **Answer questions** about your codebase
- 📊 **Generate analytics** and quality reports
- 🔄 **Find similar code** across your projects
- 📋 **Get detailed info** about specific functions/classes

## 🚀 Quick Start (Already Set Up!)

### ✅ Configuration Complete
Your Claude Desktop is now configured with the MCP server at:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### ✅ Next Steps
1. **Restart Claude Desktop** (important!)
2. Open a new conversation
3. Start using the tools!

## 💬 How to Use with Claude

### 1️⃣ Index Your Codebase

**You say:**
> "Can you analyze my Python project at `/Users/nwakrat/myprojects/misc/src`?"

**Claude will:**
- Use the `index_codebase` tool
- Process all Python files
- Create searchable embeddings
- Give you a summary of what was found

### 2️⃣ Search Your Code

**You say:**
> "Find all authentication-related functions"

**Claude will:**
- Use the `search_code` tool
- Return relevant functions with similarity scores
- Show file locations and descriptions

### 3️⃣ Ask Questions

**You say:**
> "What are the most complex functions in my codebase?"

**Claude will:**
- Use the `ask_agent` tool
- Get contextual analysis
- Provide insights about code complexity

### 4️⃣ Get Analytics

**You say:**
> "Generate a code quality report for my project"

**Claude will:**
- Use the `get_code_analytics` tool
- Analyze quality metrics, security, architecture
- Provide comprehensive insights

### 5️⃣ Find Similar Code

**You say:**
> "Find functions similar to `authenticate_user`"

**Claude will:**
- Use the `find_similar_code` tool
- Calculate semantic similarity
- Show related functions with similarity scores

### 6️⃣ Get Detailed Info

**You say:**
> "Show me the details of the `CodeEmbeddingGenerator` class"

**Claude will:**
- Use the `get_entity_details` tool
- Show source code, documentation, dependencies
- Provide complete entity information

## 🎯 Example Conversation Flow

```
You: "Hi Claude! Can you help me analyze my code?"

Claude: "I'd be happy to help! I can analyze Python codebases using 
semantic indexing. What's the path to your project?"

You: "/Users/nwakrat/myprojects/misc/src"

Claude: "I'll index your codebase now!"
[Uses index_codebase tool]
"✅ Indexed successfully! Found 8 files with 45 functions and 6 classes.
Your code uses real ML embeddings for high-quality analysis."

You: "What are the main components?"

Claude: [Uses search_code and get_code_analytics tools]
"Your codebase has several key components:
- **CLI Application**: Command-line interface in apps/cli/
- **Core Libraries**: AST analysis, embeddings, knowledge graph
- **AI Agent**: Natural language query processing
- **Pipeline**: Main orchestration logic

The most complex functions are in the embeddings module..."

You: "Show me functions similar to generate_embeddings"

Claude: [Uses find_similar_code tool]
"Here are functions similar to generate_embeddings:
1. **process_entities** (similarity: 0.84)
2. **analyze_codebase** (similarity: 0.79)
3. **create_knowledge_graph** (similarity: 0.72)
..."
```

## 🔧 Advanced Usage

### Using Real ML Embeddings

**You say:**
> "Index my codebase with high-quality ML embeddings"

**Claude will:**
- Set `use_real_embeddings: true`
- Use sentence-transformers for better semantic understanding
- Provide higher quality search results

### Multiple Projects

**You say:**
> "Index my FastAPI project at ~/projects/api and my Django project at ~/projects/web separately"

**Claude will:**
- Create separate indexes for each project
- Allow you to switch between them for analysis

### Contextual Filtering

**You say:**
> "Find authentication functions in the security domain with high importance"

**Claude will:**
- Use domain and importance filters
- Provide targeted results

## 🛠️ Technical Details

### Tools Available to Claude

1. **`index_codebase`** - Index Python projects
2. **`search_code`** - Semantic code search
3. **`ask_agent`** - Natural language queries
4. **`get_code_analytics`** - Comprehensive analytics
5. **`find_similar_code`** - Similarity search
6. **`get_entity_details`** - Detailed entity info

### Embedding Modes

- **Mock Embeddings** (fast): Good for development, quick analysis
- **Real ML Embeddings** (high-quality): Best for production analysis

### Storage

- Indexes stored locally (default: `./code_index/`)
- SQLite-based knowledge graph
- Can maintain multiple project indexes

## 🚨 Troubleshooting

### Claude Can't See the Tools

1. **Restart Claude Desktop** (most common fix)
2. Check MCP server logs in Claude's log directory
3. Verify configuration with: `python mcp_server/setup_claude.py --verify`

### Import Errors

1. Ensure Python environment has all dependencies
2. Check PYTHONPATH in the configuration
3. Verify the semantic code indexer is properly installed

### Performance Issues

1. Use mock embeddings for faster indexing during development
2. Switch to real ML embeddings only when you need high-quality results
3. Index smaller codebases first to test

## 🎉 You're All Set!

Your Claude Desktop now has powerful code analysis capabilities! Just:

1. **Restart Claude Desktop**
2. **Start a new conversation** 
3. **Ask Claude to analyze your code**

Example first prompt:
> "Hi Claude! Can you analyze my Python project at `/path/to/my/project` and tell me about its structure and quality?"

Happy coding! 🚀
