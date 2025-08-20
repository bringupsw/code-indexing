"""
Common library for the code indexer.

This package contains the shared components and utilities used across
the code indexer application.
"""

from .models import (
    CodeEntity,
    CodeRelation,
    QueryResult,
    QueryIntent,
    BusinessDomain,
    ArchitecturalPattern,
    TeamOwnership,
    ContextualAnnotation,
)
from .contextual_knowledge import ContextualKnowledgeLoader
from .ast_analyzer import SemanticASTAnalyzer
from .embeddings import CodeEmbeddingGenerator
from .knowledge_graph import CodeKnowledgeGraph
from .ai_agent import CodebaseAIAgent, NaturalLanguageQueryProcessor
from .pipeline import CodebaseSemanticPipeline

__all__ = [
    "CodeEntity",
    "CodeRelation",
    "QueryResult",
    "QueryIntent",
    "BusinessDomain",
    "ArchitecturalPattern",
    "TeamOwnership",
    "ContextualAnnotation",
    "ContextualKnowledgeLoader",
    "SemanticASTAnalyzer",
    "CodeEmbeddingGenerator",
    "CodeKnowledgeGraph",
    "CodebaseAIAgent",
    "NaturalLanguageQueryProcessor",
    "CodebaseSemanticPipeline",
]
