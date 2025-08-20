#!/usr/bin/env python3
"""
MCP Server for Semantic Code Indexer

This server exposes the semantic code indexer capabilities to Claude Desktop
via the Model Context Protocol (MCP).
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to the path to import our code indexer
sys.path.append(str(Path(__file__).parent.parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    sys.exit(1)

# Import our semantic code indexer
from src.code_indexer.libs.common.pipeline import CodebaseSemanticPipeline
from src.code_indexer.libs.common.models import CodeEntity
from src.code_indexer.libs.common.embeddings import CodeEmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-code-indexer-mcp")

# Global pipeline instance
pipeline: Optional[CodebaseSemanticPipeline] = None
current_index_path: Optional[str] = None

# MCP Server instance
server = Server("semantic-code-indexer")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for the semantic code indexer."""
    return [
        Tool(
            name="index_codebase",
            description="Index a Python codebase for semantic search and analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "codebase_path": {
                        "type": "string",
                        "description": "Path to the codebase to index"
                    },
                    "output_path": {
                        "type": "string", 
                        "description": "Path where to store the index (optional)",
                        "default": "./code_index"
                    },
                    "use_real_embeddings": {
                        "type": "boolean",
                        "description": "Use real ML embeddings (slower but higher quality)",
                        "default": False
                    },
                    "exclude_tests": {
                        "type": "boolean",
                        "description": "Exclude test files from indexing",
                        "default": True
                    }
                },
                "required": ["codebase_path"]
            }
        ),
        Tool(
            name="search_code",
            description="Search the indexed codebase using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'authentication functions', 'database connections')"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["semantic", "graph", "hybrid", "contextual"],
                        "description": "Type of search to perform",
                        "default": "hybrid"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "domain": {
                        "type": "string",
                        "description": "Filter by business domain (optional)"
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Minimum importance score",
                        "default": 0.0
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="ask_agent",
            description="Ask natural language questions about the codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the codebase"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["natural", "structured", "json"],
                        "description": "Response format",
                        "default": "structured"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="get_code_analytics",
            description="Generate comprehensive analytics about the codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_quality": {
                        "type": "boolean",
                        "description": "Include code quality metrics",
                        "default": True
                    },
                    "include_security": {
                        "type": "boolean", 
                        "description": "Include security analysis",
                        "default": True
                    },
                    "include_architecture": {
                        "type": "boolean",
                        "description": "Include architectural patterns analysis",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="find_similar_code",
            description="Find code entities similar to a given function or class",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the function, class, or entity to find similar code for"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0 to 1.0)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of similar entities to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["entity_name"]
            }
        ),
        Tool(
            name="get_entity_details",
            description="Get detailed information about a specific code entity",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the entity to get details for"
                    },
                    "include_source": {
                        "type": "boolean",
                        "description": "Include full source code",
                        "default": True
                    },
                    "include_dependencies": {
                        "type": "boolean",
                        "description": "Include dependency information",
                        "default": True
                    }
                },
                "required": ["entity_name"]
            }
        )
    ]

def ensure_pipeline(index_path: str = None) -> CodebaseSemanticPipeline:
    """Ensure we have a pipeline instance."""
    global pipeline, current_index_path
    
    if index_path and index_path != current_index_path:
        current_index_path = index_path
        pipeline = None
    
    if pipeline is None:
        output_dir = index_path or "./code_index"
        pipeline = CodebaseSemanticPipeline(
            output_dir=output_dir,
            context_config_dir="./context_config"
        )
        current_index_path = output_dir
        
    return pipeline

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls from Claude."""
    try:
        if name == "index_codebase":
            return await handle_index_codebase(arguments)
        elif name == "search_code":
            return await handle_search_code(arguments)
        elif name == "ask_agent":
            return await handle_ask_agent(arguments)
        elif name == "get_code_analytics":
            return await handle_get_analytics(arguments)
        elif name == "find_similar_code":
            return await handle_find_similar(arguments)
        elif name == "get_entity_details":
            return await handle_get_entity_details(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_index_codebase(args: Dict[str, Any]) -> List[TextContent]:
    """Handle codebase indexing."""
    codebase_path = args["codebase_path"]
    output_path = args.get("output_path", "./code_index")
    use_real_embeddings = args.get("use_real_embeddings", False)
    exclude_tests = args.get("exclude_tests", True)
    
    if not os.path.exists(codebase_path):
        return [TextContent(type="text", text=f"Error: Codebase path '{codebase_path}' does not exist")]
    
    # Set environment variable for embeddings
    if use_real_embeddings:
        os.environ['USE_REAL_EMBEDDINGS'] = 'true'
    else:
        os.environ['USE_REAL_EMBEDDINGS'] = 'false'
    
    try:
        pipeline = ensure_pipeline(output_path)
        
        # Process the codebase
        results = pipeline.process_codebase(
            codebase_path,
            exclude_tests=exclude_tests
        )
        
        embedding_info = pipeline.embedding_generator.get_embedding_info()
        
        summary = f"""✅ **Codebase Indexed Successfully**

📁 **Source**: {codebase_path}
💾 **Index**: {output_path}
🧮 **Embeddings**: {embedding_info['type']} ({embedding_info['dimensions']} dimensions)

📊 **Results**:
- **Files processed**: {results.get('files_processed', 'N/A')}
- **Functions found**: {results.get('functions_found', 'N/A')}
- **Classes found**: {results.get('classes_found', 'N/A')}
- **Total entities**: {results.get('total_entities', 'N/A')}

🎯 **Ready for**:
- Semantic search with `search_code`
- Natural language queries with `ask_agent`
- Code analytics with `get_code_analytics`
- Similarity search with `find_similar_code`

💡 **Tip**: Use `ask_agent` for natural language questions about your code!
"""
        
        return [TextContent(type="text", text=summary)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error indexing codebase: {str(e)}")]

async def handle_search_code(args: Dict[str, Any]) -> List[TextContent]:
    """Handle code search."""
    query = args["query"]
    search_type = args.get("search_type", "hybrid")
    top_k = args.get("top_k", 10)
    domain = args.get("domain")
    min_importance = args.get("min_importance", 0.0)
    
    try:
        pipeline = ensure_pipeline()
        
        # Perform search
        results = pipeline.search(
            query=query,
            search_type=search_type,
            top_k=top_k,
            domain=domain,
            min_importance=min_importance
        )
        
        if not results:
            return [TextContent(type="text", text=f"No results found for query: '{query}'")]
        
        # Format results
        response = f"🔍 **Search Results for**: '{query}'\n"
        response += f"📊 **Search Type**: {search_type} | **Results**: {len(results)}\n\n"
        
        for i, result in enumerate(results[:top_k], 1):
            entity = result.get('entity', {})
            score = result.get('score', 0)
            
            response += f"**{i}. {entity.get('name', 'Unknown')}** (Score: {score:.3f})\n"
            response += f"   📁 File: `{entity.get('file_path', 'N/A')}`\n"
            response += f"   🏷️ Type: {entity.get('type', 'N/A')}\n"
            
            if entity.get('docstring'):
                response += f"   📝 {entity['docstring'][:100]}...\n"
                
            if entity.get('business_domain'):
                response += f"   🏢 Domain: {entity['business_domain']}\n"
                
            response += "\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching code: {str(e)}")]

async def handle_ask_agent(args: Dict[str, Any]) -> List[TextContent]:
    """Handle natural language questions."""
    question = args["question"]
    format_type = args.get("format", "structured")
    
    try:
        pipeline = ensure_pipeline()
        
        # Query the AI agent
        response = pipeline.query_agent(question, format_type=format_type)
        
        formatted_response = f"🤖 **AI Agent Response**\n\n**Question**: {question}\n\n**Answer**:\n{response}"
        
        return [TextContent(type="text", text=formatted_response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error querying agent: {str(e)}")]

async def handle_get_analytics(args: Dict[str, Any]) -> List[TextContent]:
    """Handle analytics generation."""
    include_quality = args.get("include_quality", True)
    include_security = args.get("include_security", True)
    include_architecture = args.get("include_architecture", True)
    
    try:
        pipeline = ensure_pipeline()
        
        # Generate analytics
        analytics = pipeline.get_contextual_analytics()
        
        response = "📊 **Codebase Analytics**\n\n"
        
        # Basic metrics
        if analytics.get('basic_metrics'):
            metrics = analytics['basic_metrics']
            response += "**📈 Basic Metrics**:\n"
            response += f"- Total files: {metrics.get('total_files', 'N/A')}\n"
            response += f"- Total functions: {metrics.get('total_functions', 'N/A')}\n"
            response += f"- Total classes: {metrics.get('total_classes', 'N/A')}\n"
            response += f"- Average complexity: {metrics.get('avg_complexity', 'N/A')}\n\n"
        
        # Quality metrics
        if include_quality and analytics.get('quality_metrics'):
            quality = analytics['quality_metrics']
            response += "**✨ Quality Metrics**:\n"
            response += f"- High complexity functions: {quality.get('high_complexity_count', 'N/A')}\n"
            response += f"- Functions without docstrings: {quality.get('missing_docstrings', 'N/A')}\n"
            response += f"- Code coverage estimate: {quality.get('coverage_estimate', 'N/A')}\n\n"
        
        # Security analysis
        if include_security and analytics.get('security_analysis'):
            security = analytics['security_analysis']
            response += "**🔒 Security Analysis**:\n"
            response += f"- Security-critical functions: {security.get('critical_functions', 'N/A')}\n"
            response += f"- Potential vulnerabilities: {security.get('vulnerabilities', 'N/A')}\n\n"
        
        # Architecture patterns
        if include_architecture and analytics.get('architecture_patterns'):
            patterns = analytics['architecture_patterns']
            response += "**🏗️ Architecture Patterns**:\n"
            for pattern, count in patterns.items():
                response += f"- {pattern}: {count}\n"
            response += "\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating analytics: {str(e)}")]

async def handle_find_similar(args: Dict[str, Any]) -> List[TextContent]:
    """Handle similarity search."""
    entity_name = args["entity_name"]
    threshold = args.get("similarity_threshold", 0.7)
    max_results = args.get("max_results", 10)
    
    try:
        pipeline = ensure_pipeline()
        
        # Find similar entities
        similar_entities = pipeline.find_similar_entities(
            entity_name=entity_name,
            threshold=threshold,
            max_results=max_results
        )
        
        if not similar_entities:
            return [TextContent(type="text", text=f"No similar entities found for '{entity_name}' with threshold {threshold}")]
        
        response = f"🔄 **Similar Code to**: '{entity_name}'\n"
        response += f"🎯 **Threshold**: {threshold} | **Results**: {len(similar_entities)}\n\n"
        
        for i, (entity, similarity) in enumerate(similar_entities, 1):
            response += f"**{i}. {entity.get('name', 'Unknown')}** (Similarity: {similarity:.3f})\n"
            response += f"   📁 File: `{entity.get('file_path', 'N/A')}`\n"
            response += f"   🏷️ Type: {entity.get('type', 'N/A')}\n"
            
            if entity.get('docstring'):
                response += f"   📝 {entity['docstring'][:80]}...\n"
                
            response += "\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error finding similar code: {str(e)}")]

async def handle_get_entity_details(args: Dict[str, Any]) -> List[TextContent]:
    """Handle entity details request."""
    entity_name = args["entity_name"]
    include_source = args.get("include_source", True)
    include_dependencies = args.get("include_dependencies", True)
    
    try:
        pipeline = ensure_pipeline()
        
        # Get entity details
        entity_details = pipeline.get_entity_details(
            entity_name=entity_name,
            include_source=include_source,
            include_dependencies=include_dependencies
        )
        
        if not entity_details:
            return [TextContent(type="text", text=f"Entity '{entity_name}' not found in the index")]
        
        entity = entity_details
        response = f"📋 **Entity Details**: {entity.get('name', 'Unknown')}\n\n"
        
        # Basic info
        response += f"**🏷️ Type**: {entity.get('type', 'N/A')}\n"
        response += f"**📁 File**: `{entity.get('file_path', 'N/A')}`\n"
        response += f"**📍 Lines**: {entity.get('line_start', 'N/A')}-{entity.get('line_end', 'N/A')}\n"
        response += f"**🔢 Complexity**: {entity.get('complexity', 'N/A')}\n"
        
        if entity.get('business_domain'):
            response += f"**🏢 Domain**: {entity['business_domain']}\n"
        
        if entity.get('docstring'):
            response += f"\n**📝 Documentation**:\n{entity['docstring']}\n"
        
        if entity.get('parameters'):
            response += f"\n**⚙️ Parameters**: {', '.join(entity['parameters'])}\n"
        
        if include_dependencies and entity.get('imports'):
            response += f"\n**📦 Imports**: {', '.join(entity['imports'])}\n"
        
        if include_source and entity.get('source_code'):
            response += f"\n**💻 Source Code**:\n```python\n{entity['source_code']}\n```\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting entity details: {str(e)}")]

async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Semantic Code Indexer MCP Server")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
