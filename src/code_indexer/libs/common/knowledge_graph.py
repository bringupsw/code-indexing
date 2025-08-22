"""
Knowledge Graph Module

This module provides knowledge graph construction and querying capabilities
for code entities and relationships using SQLite and vector embeddings.
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional

from .models import CodeEntity, CodeRelation


class CodeKnowledgeGraph:
    """
    Builds and manages a knowledge graph of code entities and relationships.

    This class provides a hybrid storage and query system that combines:
    1. Relational database (SQLite) for structured entity and relationship data
    2. Vector storage for embedding-based semantic search
    3. Graph traversal capabilities for relationship discovery

    The knowledge graph enables multiple types of queries:
    - Semantic similarity search using vector embeddings
    - Graph traversal for finding related entities
    - Structured queries for filtering by properties
    - Hybrid search combining multiple approaches

    Database Schema:
    - entities: Stores code entities with metadata and embeddings
    - relations: Stores relationships between entities

    Attributes:
        db_path (str): Path to the SQLite database file
    """

    def __init__(self, db_path: str = "code_knowledge.db"):
        """
        Initialize the knowledge graph with database setup.

        Args:
            db_path (str): Path where the SQLite database will be stored
        """
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """
        Initialize SQLite database schema for knowledge graph.

        Creates tables for entities and relations with appropriate indexes
        for efficient querying. The schema supports:
        - Full-text search on entity names and types
        - Vector similarity search on embeddings
        - Graph traversal queries on relationships
        """
        conn = sqlite3.connect(self.db_path)

        # Entities table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                file_path TEXT,
                line_start INTEGER,
                line_end INTEGER,
                source_code TEXT,
                docstring TEXT,
                signature TEXT,
                complexity INTEGER,
                parameters TEXT,
                return_type TEXT,
                decorators TEXT,
                class_name TEXT,
                imports TEXT,
                calls TEXT,
                variables TEXT,
                has_docstring BOOLEAN,
                is_public BOOLEAN,
                is_tested BOOLEAN,
                handles_exceptions BOOLEAN,
                has_type_hints BOOLEAN,
                base_classes TEXT,
                is_abstract BOOLEAN,
                business_domain TEXT,
                team_owner TEXT,
                architectural_pattern TEXT,
                importance_score REAL,
                quality_issues TEXT,
                annotations TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Relations table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                weight REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES entities (id),
                FOREIGN KEY (target_id) REFERENCES entities (id)
            )
        """
        )

        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_business_domain ON entities(business_domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_team_owner ON entities(team_owner)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_importance_score ON entities(importance_score)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_target ON relations(source_id, target_id)"
        )

        conn.commit()
        conn.close()

    def store_entities_and_embeddings(
        self, entities: List[CodeEntity], embeddings: Dict[str, np.ndarray]
    ):
        """
        Store entities and their embeddings in the knowledge graph.

        This method persists code entities along with their vector embeddings
        to the database. Embeddings are stored as binary data for efficient
        retrieval during similarity search operations.

        Args:
            entities (List[CodeEntity]): List of entities to store
            embeddings (Dict[str, np.ndarray]): Mapping from entity IDs to embeddings

        Note:
            Uses INSERT OR REPLACE to handle updates to existing entities
        """
        conn = sqlite3.connect(self.db_path)

        for entity in entities:
            embedding_blob = embeddings[entity.id].tobytes() if entity.id in embeddings else None

            conn.execute(
                """
                INSERT OR REPLACE INTO entities 
                (id, type, name, file_path, line_start, line_end, source_code, docstring, 
                 signature, complexity, parameters, return_type, decorators, class_name,
                 imports, calls, variables, has_docstring, is_public, is_tested,
                 handles_exceptions, has_type_hints, base_classes, is_abstract, business_domain, team_owner,
                 architectural_pattern, importance_score, quality_issues, annotations, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entity.id,
                    entity.type,
                    entity.name,
                    entity.file_path,
                    entity.line_start,
                    entity.line_end,
                    entity.source_code,
                    entity.docstring,
                    entity.signature,
                    entity.complexity,
                    json.dumps(entity.parameters),
                    entity.return_type,
                    json.dumps(entity.decorators),
                    entity.class_name,
                    json.dumps(entity.imports),
                    json.dumps(entity.calls),
                    json.dumps(entity.variables),
                    entity.has_docstring,
                    entity.is_public,
                    entity.is_tested,
                    entity.handles_exceptions,
                    entity.has_type_hints,
                    json.dumps(entity.base_classes),
                    entity.is_abstract,
                    entity.business_domain,
                    entity.team_owner,
                    entity.architectural_pattern,
                    entity.importance_score,
                    json.dumps(entity.quality_issues),
                    json.dumps(entity.annotations),
                    embedding_blob,
                ),
            )

        conn.commit()
        conn.close()

    def store_relations(self, relations: List[CodeRelation]):
        """
        Store relationships between entities.

        This method persists relationship data that connects entities
        in the knowledge graph. Relationships enable graph traversal
        and discovery of related code elements.

        Args:
            relations (List[CodeRelation]): List of relationships to store
        """
        conn = sqlite3.connect(self.db_path)

        for relation in relations:
            conn.execute(
                """
                INSERT OR REPLACE INTO relations
                (source_id, target_id, relation_type, weight, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    relation.source_id,
                    relation.target_id,
                    relation.relation_type,
                    relation.weight,
                    json.dumps(relation.metadata),
                ),
            )

        conn.commit()
        conn.close()

    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Perform semantic search using vector similarity.

        This method finds entities that are semantically similar to a query
        by comparing vector embeddings using cosine similarity. It's useful
        for finding code that performs similar functions or has similar
        characteristics.

        Args:
            query_embedding (np.ndarray): Query vector to search for
            top_k (int): Maximum number of results to return

        Returns:
            List[Dict]: List of similar entities with similarity scores

        Note:
            Uses cosine similarity metric for comparing embeddings
        """
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute(
            "SELECT id, name, type, embedding FROM entities WHERE embedding IS NOT NULL"
        )

        results = []
        for row in cursor:
            entity_id, name, entity_type, embedding_blob = row

            # Deserialize embedding
            entity_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, entity_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
            )

            results.append(
                {"id": entity_id, "name": name, "type": entity_type, "similarity": similarity}
            )

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        conn.close()

        return results[:top_k]

    def graph_search(
        self, entity_id: str, relation_types: List[str] = None, depth: int = 2
    ) -> Dict:
        """
        Perform graph traversal search starting from an entity.

        This method explores the relationship graph starting from a specific
        entity and returns connected entities within a given depth. It's
        useful for understanding code dependencies and impact analysis.

        Args:
            entity_id (str): Starting entity for traversal
            relation_types (List[str], optional): Filter by specific relation types
            depth (int): Maximum traversal depth

        Returns:
            Dict: Graph structure with nodes and edges

        Note:
            Uses breadth-first traversal with cycle detection
        """
        conn = sqlite3.connect(self.db_path)

        visited = set()
        result = {"nodes": [], "edges": []}

        def traverse(current_id, current_depth):
            if current_depth > depth or current_id in visited:
                return

            visited.add(current_id)

            # Get entity info
            cursor = conn.execute("SELECT * FROM entities WHERE id = ?", (current_id,))
            entity = cursor.fetchone()
            if entity:
                result["nodes"].append(
                    {"id": entity[0], "type": entity[1], "name": entity[2], "complexity": entity[9]}
                )

            # Get related entities
            relation_filter = ""
            params = [current_id]

            if relation_types:
                placeholders = ",".join(["?"] * len(relation_types))
                relation_filter = f" AND relation_type IN ({placeholders})"
                params.extend(relation_types)

            cursor = conn.execute(
                f"""
                SELECT target_id, relation_type, weight 
                FROM relations 
                WHERE source_id = ?{relation_filter}
            """,
                params,
            )

            for target_id, relation_type, weight in cursor:
                result["edges"].append(
                    {
                        "source": current_id,
                        "target": target_id,
                        "type": relation_type,
                        "weight": weight,
                    }
                )

                traverse(target_id, current_depth + 1)

        traverse(entity_id, 0)
        conn.close()

        return result

    def contextual_search(
        self, query: str, context_filters: Dict[str, Any] = None, top_k: int = 10
    ) -> List[Dict]:
        """
        Perform contextual search considering business domains, teams, and importance.

        This method provides enhanced search that takes into account:
        - Business domain relevance
        - Team ownership
        - Importance scores
        - Contextual annotations

        Args:
            query (str): Search query
            context_filters (Dict[str, Any], optional): Context-based filters
            top_k (int): Maximum number of results

        Returns:
            List[Dict]: Contextually ranked search results
        """
        conn = sqlite3.connect(self.db_path)

        # Build the base query
        sql_query = """
            SELECT id, name, type, business_domain, team_owner, 
                   importance_score, annotations, embedding,
                   file_path, complexity
            FROM entities 
            WHERE (name LIKE ? OR docstring LIKE ? OR source_code LIKE ?)
        """

        like_query = f"%{query}%"
        params = [like_query, like_query, like_query]

        # Apply context filters
        if context_filters:
            if "business_domain" in context_filters:
                sql_query += " AND business_domain = ?"
                params.append(context_filters["business_domain"])

            if "team_owner" in context_filters:
                sql_query += " AND team_owner = ?"
                params.append(context_filters["team_owner"])

            if "min_importance" in context_filters:
                sql_query += " AND importance_score >= ?"
                params.append(context_filters["min_importance"])

            if "architectural_pattern" in context_filters:
                sql_query += " AND architectural_pattern = ?"
                params.append(context_filters["architectural_pattern"])

            if "type" in context_filters:
                sql_query += " AND type = ?"
                params.append(context_filters["type"])

        sql_query += " ORDER BY importance_score DESC"

        cursor = conn.execute(sql_query, params)

        results = []
        for row in cursor:
            (
                entity_id,
                name,
                entity_type,
                business_domain,
                team_owner,
                importance_score,
                annotations,
                embedding_blob,
                file_path,
                complexity,
            ) = row

            # Parse annotations
            parsed_annotations = {}
            if annotations:
                try:
                    parsed_annotations = json.loads(annotations)
                except:
                    parsed_annotations = {}

            results.append(
                {
                    "id": entity_id,
                    "name": name,
                    "type": entity_type,
                    "importance_score": importance_score,
                    "business_domain": business_domain,
                    "team_owner": team_owner,
                    "annotations": parsed_annotations,
                    "file_path": file_path,
                    "complexity": complexity,
                }
            )

        conn.close()

        return results[:top_k]

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Retrieve a specific entity by its ID.

        Args:
            entity_id (str): Entity ID to retrieve

        Returns:
            Optional[Dict]: Entity data if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()

        if row:
            # Convert row to dictionary
            columns = [desc[0] for desc in cursor.description]
            entity_dict = dict(zip(columns, row))

            # Parse JSON fields
            for field in [
                "parameters",
                "decorators",
                "imports",
                "calls",
                "variables",
                "quality_issues",
                "annotations",
            ]:
                if entity_dict.get(field):
                    try:
                        entity_dict[field] = json.loads(entity_dict[field])
                    except:
                        entity_dict[field] = []

            conn.close()
            return entity_dict

        conn.close()
        return None

    def get_statistics(self) -> Dict:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dict: Statistics including entity counts, relation counts, etc.
        """
        conn = sqlite3.connect(self.db_path)

        stats = {}

        # Entity statistics
        cursor = conn.execute("SELECT COUNT(*) FROM entities")
        stats["total_entities"] = cursor.fetchone()[0]

        cursor = conn.execute("SELECT type, COUNT(*) FROM entities GROUP BY type")
        stats["entity_types"] = dict(cursor.fetchall())

        # Relation statistics
        cursor = conn.execute("SELECT COUNT(*) FROM relations")
        stats["total_relations"] = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type"
        )
        stats["relation_types"] = dict(cursor.fetchall())

        # Business domain statistics
        cursor = conn.execute(
            "SELECT business_domain, COUNT(*) FROM entities WHERE business_domain IS NOT NULL GROUP BY business_domain"
        )
        stats["business_domains"] = dict(cursor.fetchall())

        # Complexity statistics
        cursor = conn.execute(
            "SELECT AVG(complexity), MIN(complexity), MAX(complexity) FROM entities"
        )
        avg_complexity, min_complexity, max_complexity = cursor.fetchone()
        stats["complexity"] = {
            "average": avg_complexity,
            "minimum": min_complexity,
            "maximum": max_complexity,
        }

        conn.close()
        return stats
