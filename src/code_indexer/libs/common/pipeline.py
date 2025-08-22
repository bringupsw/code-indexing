"""
Pipeline Orchestrator

This module provides the main pipeline orchestrator that coordinates all components
of the semantic code indexing system.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from .models import CodeEntity, CodeRelation
from .ast_analyzer import SemanticASTAnalyzer
from .embeddings import CodeEmbeddingGenerator
from .knowledge_graph import CodeKnowledgeGraph
from .ai_agent import CodebaseAIAgent


class CodebaseSemanticPipeline:
    """
    Complete pipeline that orchestrates the entire semantic indexing process.

    This is the main orchestrator class that coordinates:
    1. AST Analysis and Semantic Extraction
    2. Embedding Generation
    3. Knowledge Graph Construction
    4. Search and Query Capabilities

    The pipeline is designed to be:
    - Scalable: Uses parallel processing for large codebases
    - Extensible: Easy to add new embedding types or analysis techniques
    - Persistent: Stores results in a durable knowledge graph
    - Queryable: Supports multiple search paradigms

    Typical workflow:
    1. Initialize pipeline with output directory
    2. Process codebase to extract and index entities
    3. Use search methods to query the indexed knowledge

    Attributes:
        output_dir (Path): Directory for storing index files
        analyzer (SemanticASTAnalyzer): AST analysis component
        embedding_generator (CodeEmbeddingGenerator): Embedding generation component
        knowledge_graph (CodeKnowledgeGraph): Knowledge storage and query component
    """

    def __init__(
        self, output_dir: str = "./code_index", context_config_dir: str = "./context_config"
    ):
        """
        Initialize the complete semantic pipeline with contextual knowledge.

        Args:
            output_dir (str): Directory where index files will be stored
            context_config_dir (str): Directory containing contextual configuration files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.analyzer = SemanticASTAnalyzer(context_config_dir)
        self.embedding_generator = CodeEmbeddingGenerator()
        self.knowledge_graph = CodeKnowledgeGraph(str(self.output_dir / "knowledge.db"))

    def process_codebase(self, codebase_path: str, project_name: str = None):
        """
        Execute the complete pipeline processing workflow.

        This method runs the complete semantic indexing pipeline:
        1. Analyzes the codebase structure and extracts entities
        2. Generates embeddings for all discovered entities
        3. Builds and persists the knowledge graph
        4. Generates summary analytics

        Args:
            codebase_path (str): Path to the codebase to analyze
            project_name (str): Identifier/name for the project being indexed

        Returns:
            Dict: Summary statistics about the processing results

        Raises:
            FileNotFoundError: If codebase path doesn't exist
            PermissionError: If output directory cannot be created
        """
        print("🔍 Phase 1: AST Analysis and Semantic Extraction")
        entities, relations = self.analyzer.analyze_codebase(codebase_path)
        print(f"   Extracted {len(entities)} entities, {len(relations)} relations")

        print("🧠 Phase 2: Embedding Generation")
        embeddings = self.embedding_generator.generate_embeddings(entities)
        print(f"   Generated embeddings for {len(embeddings)} entities")

        print("📊 Phase 3: Knowledge Graph Construction")
        self.knowledge_graph.store_entities_and_embeddings(entities, embeddings)
        self.knowledge_graph.store_relations(relations)
        print("   Knowledge graph constructed and stored")

        print("✅ Pipeline completed successfully!")

        # Generate summary report with project information
        self._generate_summary_report(entities, relations, codebase_path, project_name)

        return {
            "entities": len(entities),
            "relations": len(relations),
            "embeddings": len(embeddings),
            "database_path": str(self.output_dir / "knowledge.db"),
        }

    def search(self, query: str, search_type: str = "hybrid") -> List[Dict]:
        """
        Search the indexed codebase using various search strategies.

        This method provides multiple search approaches:
        - semantic: Vector similarity search using embeddings
        - graph: Graph traversal search for relationships
        - hybrid: Combination of semantic and graph search

        Args:
            query (str): Search query (text for semantic, entity_id for graph)
            search_type (str): Type of search ('semantic', 'graph', 'hybrid')

        Returns:
            List[Dict]: Search results with relevance scores and metadata

        Raises:
            ValueError: If search_type is not supported
        """
        if search_type == "semantic":
            # Generate query embedding and search
            query_entity = CodeEntity(
                id="query",
                name=query,
                type="query",
                file_path="",
                line_start=0,
                line_end=0,
                source_code=query,
                docstring=None,
                signature="",
                complexity=0,
            )
            query_embeddings = self.embedding_generator.generate_embeddings([query_entity])
            return self.knowledge_graph.semantic_search(query_embeddings["query"])

        elif search_type == "graph":
            # Graph-based search (requires entity_id)
            return self.knowledge_graph.graph_search(query)

        else:  # hybrid
            # Combine semantic and graph search
            semantic_results = self.search(query, "semantic")
            # Enhance with graph information for top results
            enhanced_results = []
            for result in semantic_results[:5]:
                graph_info = self.knowledge_graph.graph_search(result["id"], depth=1)
                result["connections"] = len(graph_info["edges"])
                enhanced_results.append(result)

            return enhanced_results

    def contextual_search(self, query: str, context_filters: Dict[str, Any] = None) -> List[Dict]:
        """
        Perform contextual search using business domain and team context.

        Args:
            query (str): Search query
            context_filters (Dict[str, Any], optional): Context-based filters

        Returns:
            List[Dict]: Contextually ranked search results
        """
        return self.knowledge_graph.contextual_search(query, context_filters)

    def get_contextual_analytics(self) -> Dict[str, Any]:
        """
        Generate comprehensive contextual analytics about the codebase.

        Returns:
            Dict[str, Any]: Analytics including domain and team insights
        """
        stats = self.knowledge_graph.get_statistics()

        analytics = {"statistics": stats, "generated_at": datetime.now().isoformat()}

        return analytics

    def create_ai_agent(self) -> CodebaseAIAgent:
        """
        Create an AI agent interface for querying the knowledge graph.

        Returns:
            CodebaseAIAgent: Configured AI agent for natural language queries
        """
        return CodebaseAIAgent(self.knowledge_graph)

    def query_agent(self, query: str) -> Dict[str, Any]:
        """
        Convenience method to query the AI agent and get response.

        Args:
            query (str): Natural language query

        Returns:
            Dict[str, Any]: Query result with entities and metadata
        """
        agent = self.create_ai_agent()
        result = agent.ask(query)
        return {
            "entities": result.entities,
            "confidence": result.confidence,
            "query_time": result.query_time,
            "metadata": result.metadata,
        }

    def _generate_summary_report(
        self,
        entities: List[CodeEntity],
        relations: List[CodeRelation],
        codebase_path: str = None,
        project_name: str = None,
    ):
        """
        Generate comprehensive summary report of the indexed codebase.

        This method analyzes the extracted entities and relationships to
        produce insights about the codebase structure, complexity distribution,
        and other key metrics. The report is saved as JSON for later reference.

        Args:
            entities (List[CodeEntity]): All extracted entities
            relations (List[CodeRelation]): All discovered relationships
            codebase_path (str): Path to the source codebase
            project_name (str): Identifier/name for the project

        Report includes:
        - Project identification information
        - Entity type distribution
        - Complexity analysis
        - Relationship type analysis
        - Most complex functions
        - Other structural metrics
        """
        # Determine project name if not provided
        if not project_name and codebase_path:
            project_name = Path(codebase_path).name
        elif not project_name:
            project_name = "Unknown Project"

        summary = {
            "project_name": project_name,
            "source_path": codebase_path,
            "created_at": datetime.now().isoformat(),
            "total_entities": len(entities),
            "entity_types": {},
            "total_relations": len(relations),
            "relation_types": {},
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
            "top_complex_functions": [],
        }

        # Analyze entities
        for entity in entities:
            summary["entity_types"][entity.type] = summary["entity_types"].get(entity.type, 0) + 1

            # Categorize complexity
            if entity.complexity <= 5:
                complexity_level = "low"
            elif entity.complexity <= 10:
                complexity_level = "medium"
            else:
                complexity_level = "high"

            summary["complexity_distribution"][complexity_level] += 1

            if entity.type in ["function", "method"] and entity.complexity > 5:
                summary["top_complex_functions"].append(
                    {
                        "name": entity.name,
                        "complexity": entity.complexity,
                        "file": entity.file_path,
                        "business_domain": entity.business_domain,
                        "team_owner": entity.team_owner,
                        "importance_score": entity.importance_score,
                    }
                )

        # Add contextual analytics
        summary["business_domains"] = {}
        summary["team_ownership"] = {}
        summary["architectural_patterns"] = {}

        for entity in entities:
            if entity.business_domain:
                domain = entity.business_domain
                if domain not in summary["business_domains"]:
                    summary["business_domains"][domain] = {
                        "count": 0,
                        "avg_complexity": 0,
                        "total_complexity": 0,
                    }
                summary["business_domains"][domain]["count"] += 1
                summary["business_domains"][domain]["total_complexity"] += entity.complexity

            if entity.team_owner:
                team = entity.team_owner
                if team not in summary["team_ownership"]:
                    summary["team_ownership"][team] = {
                        "count": 0,
                        "avg_complexity": 0,
                        "total_complexity": 0,
                    }
                summary["team_ownership"][team]["count"] += 1
                summary["team_ownership"][team]["total_complexity"] += entity.complexity

            if entity.architectural_pattern:
                pattern = entity.architectural_pattern
                summary["architectural_patterns"][pattern] = (
                    summary["architectural_patterns"].get(pattern, 0) + 1
                )

        # Calculate averages
        for domain_data in summary["business_domains"].values():
            if domain_data["count"] > 0:
                domain_data["avg_complexity"] = round(
                    domain_data["total_complexity"] / domain_data["count"], 2
                )
                del domain_data["total_complexity"]

        for team_data in summary["team_ownership"].values():
            if team_data["count"] > 0:
                team_data["avg_complexity"] = round(
                    team_data["total_complexity"] / team_data["count"], 2
                )
                del team_data["total_complexity"]

        # Analyze relations
        for relation in relations:
            relation_type = relation.relation_type
            summary["relation_types"][relation_type] = (
                summary["relation_types"].get(relation_type, 0) + 1
            )

        # Sort complex functions
        summary["top_complex_functions"].sort(key=lambda x: x["complexity"], reverse=True)
        summary["top_complex_functions"] = summary["top_complex_functions"][:10]

        # Save summary
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📈 Summary Report:")
        print(f"   Project: {summary['project_name']}")
        print(f"   Source: {summary['source_path']}")
        print(f"   Total Entities: {summary['total_entities']}")
        print(f"   Entity Types: {summary['entity_types']}")
        print(f"   Total Relations: {summary['total_relations']}")
        print(f"   Complexity Distribution: {summary['complexity_distribution']}")
        print(f"   Business Domains: {len(summary['business_domains'])}")
        print(f"   Teams: {len(summary['team_ownership'])}")
        print(f"   Architectural Patterns: {len(summary['architectural_patterns'])}")

        # Show domain insights
        if summary["business_domains"]:
            print(f"   Top Business Domains:")
            sorted_domains = sorted(
                summary["business_domains"].items(), key=lambda x: x[1]["count"], reverse=True
            )
            for domain, data in sorted_domains[:3]:
                print(
                    f"     {domain}: {data['count']} entities (avg complexity: {data['avg_complexity']})"
                )

        # Show team insights
        if summary["team_ownership"]:
            print(f"   Team Distribution:")
            sorted_teams = sorted(
                summary["team_ownership"].items(), key=lambda x: x[1]["count"], reverse=True
            )
            for team, data in sorted_teams[:3]:
                print(
                    f"     {team}: {data['count']} entities (avg complexity: {data['avg_complexity']})"
                )
