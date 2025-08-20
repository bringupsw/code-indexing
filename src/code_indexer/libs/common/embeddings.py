"""
Embedding Generation Module

This module provides code embedding generation capabilities using multiple approaches
to create rich vector representations of code entities.
"""

import hashlib
import numpy as np
import os
from typing import Dict, List

from .models import CodeEntity

# Try to import ML dependencies, fall back to mock if not available
USE_REAL_EMBEDDINGS = os.getenv("USE_REAL_EMBEDDINGS", "false").lower() == "true"

try:
    if USE_REAL_EMBEDDINGS:
        from sentence_transformers import SentenceTransformer
        import torch

        ML_AVAILABLE = True
    else:
        ML_AVAILABLE = False
except ImportError:
    ML_AVAILABLE = False
    USE_REAL_EMBEDDINGS = False


class CodeEmbeddingGenerator:
    """
    Generates embeddings for code entities using multiple approaches.

    This class creates vector representations of code entities by combining
    three different embedding techniques:
    1. Text-based embeddings: From source code and documentation
    2. Structure-based embeddings: From AST patterns and metrics
    3. Context-based embeddings: From semantic patterns and intent

    The hybrid approach provides richer representations that capture both
    syntactic and semantic aspects of code entities.

    Supports both mock embeddings (for development/testing) and real ML
    embeddings (for production use).

    Attributes:
        model_name (str): Name of the base model for text embeddings
        use_real_embeddings (bool): Whether to use real ML models or mock embeddings
        text_model: Loaded sentence transformer model (if using real embeddings)
    """

    def __init__(self, model_name="microsoft/codebert-base", use_real_embeddings=None):
        """
        Initialize the embedding generator.

        Args:
            model_name (str): Model identifier for text-based embeddings
            use_real_embeddings (bool): Override to use real ML models
        """
        self.model_name = model_name
        self.use_real_embeddings = (
            use_real_embeddings if use_real_embeddings is not None else USE_REAL_EMBEDDINGS
        )
        self.text_model = None

        if self.use_real_embeddings and ML_AVAILABLE:
            try:
                print(f"🚀 Loading real ML embedding model: {model_name}")
                # Use a code-specific model if available, fall back to general models
                if "codebert" in model_name.lower():
                    # For CodeBERT, we'll use sentence-transformers compatible model
                    self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
                else:
                    self.text_model = SentenceTransformer(model_name)
                print("✅ Real ML embeddings loaded successfully!")
            except Exception as e:
                print(f"⚠️  Failed to load ML model, falling back to mock embeddings: {e}")
                self.use_real_embeddings = False
        else:
            print("📚 Using mock embeddings (fast, deterministic, good for development)")

    def get_embedding_info(self):
        """Get information about the current embedding configuration."""
        return {
            "type": "real_ml" if self.use_real_embeddings else "mock",
            "model_name": self.model_name,
            "ml_available": ML_AVAILABLE,
            "dimensions": (
                480
                if not self.use_real_embeddings
                else (
                    self.text_model.get_sentence_embedding_dimension() + 96
                    if self.text_model
                    else 480
                )
            ),
        }

    def generate_embeddings(self, entities: List[CodeEntity]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all code entities.

        This method processes each entity through the complete embedding
        pipeline, generating and combining multiple types of embeddings
        into a single vector representation.

        Args:
            entities (List[CodeEntity]): List of entities to embed

        Returns:
            Dict[str, np.ndarray]: Mapping from entity IDs to embedding vectors
        """
        embeddings = {}

        for entity in entities:
            # Generate multiple types of embeddings
            text_embedding = self._generate_text_embedding(entity)
            structure_embedding = self._generate_structure_embedding(entity)
            context_embedding = self._generate_context_embedding(entity)

            # Combine embeddings
            combined_embedding = np.concatenate(
                [text_embedding, structure_embedding, context_embedding]
            )

            embeddings[entity.id] = combined_embedding

        return embeddings

    def _generate_text_embedding(self, entity: CodeEntity) -> np.ndarray:
        """
        Generate text-based embedding from source code and docstring.

        This method uses either real ML models (sentence transformers) or
        mock embeddings based on configuration.

        Args:
            entity (CodeEntity): Entity to generate embedding for

        Returns:
            np.ndarray: Text-based embedding vector
        """
        text = f"{entity.name} {entity.docstring or ''} {entity.source_code}"

        if self.use_real_embeddings and self.text_model:
            # Use real ML model
            try:
                embedding = self.text_model.encode([text], convert_to_numpy=True)[0]
                return embedding.astype(np.float32)
            except Exception as e:
                print(f"⚠️  ML embedding failed, using mock: {e}")
                # Fall back to mock

        # Mock embedding for demo - creates deterministic embedding based on content
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        return np.random.random(384).astype(np.float32)

    def _generate_structure_embedding(self, entity: CodeEntity) -> np.ndarray:
        """
        Generate structure-based embedding from AST patterns.

        This method creates embeddings based on structural properties
        of the code entity such as complexity, size, and dependencies.
        In practice, this would use code2vec-style path embeddings.

        Args:
            entity (CodeEntity): Entity to generate embedding for

        Returns:
            np.ndarray: Structure-based embedding vector
        """
        # This would use code2vec-style path embeddings in practice
        features = [
            entity.complexity,
            len(entity.imports),
            entity.line_end - entity.line_start,
            len(entity.parameters),
            1 if entity.docstring else 0,
        ]

        # Pad to fixed size for consistent vector dimensions
        features.extend([0] * (64 - len(features)))
        return np.array(features[:64], dtype=np.float32)

    def _generate_context_embedding(self, entity: CodeEntity) -> np.ndarray:
        """
        Generate context-based embedding from semantic patterns.

        This method creates embeddings based on semantic context such as
        entity type, naming patterns, visibility, and inferred intent.
        These features help capture the semantic role of code entities.

        Args:
            entity (CodeEntity): Entity to generate embedding for

        Returns:
            np.ndarray: Context-based embedding vector
        """
        context_features = []

        # Entity type encoding
        type_map = {"function": 0, "method": 1, "class": 2, "variable": 3}
        context_features.append(type_map.get(entity.type, 0))

        # Semantic intent encoding based on annotations
        intent_map = {"data_access": 0, "data_modification": 1, "computation": 2, "control": 3}
        intent = entity.annotations.get("intent", "computation")
        context_features.append(intent_map.get(intent, 2))

        # Complexity level
        complexity_map = {"low": 0, "medium": 1, "high": 2}
        complexity_level = entity.annotations.get("complexity_level", "medium")
        context_features.append(complexity_map.get(complexity_level, 1))

        # Visibility
        visibility_map = {"public": 0, "private": 1, "protected": 2}
        visibility = entity.annotations.get("visibility", "public")
        context_features.append(visibility_map.get(visibility, 0))

        # Business domain encoding
        domain_features = [0] * 8  # Support for 8 domains
        if entity.business_domain:
            domain_names = [
                "Authentication",
                "Database",
                "API",
                "Payment",
                "Core",
                "UI",
                "Configuration",
                "Testing",
            ]
            if entity.business_domain in domain_names:
                domain_features[domain_names.index(entity.business_domain)] = 1

        context_features.extend(domain_features)

        # Pad to fixed size
        context_features.extend([0] * (32 - len(context_features)))
        return np.array(context_features[:32], dtype=np.float32)
