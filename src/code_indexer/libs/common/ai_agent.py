"""
AI Agent Interface System

This module provides natural language querying capabilities for the code knowledge graph,
allowing users to ask questions about their codebase in natural language.
"""

import json
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional

from .models import QueryResult, QueryIntent
from .knowledge_graph import CodeKnowledgeGraph


class AIQueryProcessor(ABC):
    """Abstract base class for AI query processors."""
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Check if this processor can handle the given query."""
        pass
    
    @abstractmethod
    def process(self, query: str, knowledge_graph: CodeKnowledgeGraph) -> QueryResult:
        """Process the query and return results."""
        pass


class NaturalLanguageQueryProcessor(AIQueryProcessor):
    """
    Processes natural language queries and converts them to knowledge graph operations.
    
    This processor analyzes natural language queries to understand user intent
    and translates them into appropriate knowledge graph queries.
    """
    
    def __init__(self):
        """Initialize the natural language processor with intent patterns."""
        self.intent_patterns = {
            QueryIntent.FIND_FUNCTION: [
                r"find.*function.*(?:named|called)\\s+(\\w+)",
                r"show.*function.*(\\w+)",
                r"where.*function.*(\\w+)",
                r"function.*(\\w+).*definition",
                r"(?:find|show|locate).*(\\w+).*function"
            ],
            QueryIntent.FIND_CLASS: [
                r"find.*class.*(?:named|called)\\s+(\\w+)",
                r"show.*class.*(\\w+)",
                r"where.*class.*(\\w+)",
                r"class.*(\\w+).*definition",
                r"(?:find|show|locate).*(\\w+).*class"
            ],
            QueryIntent.SECURITY_ANALYSIS: [
                r"security.*(?:issues|problems|vulnerabilities)",
                r"(?:find|show).*security.*(?:critical|sensitive)",
                r"authentication.*(?:code|functions)",
                r"authorization.*(?:code|functions)",
                r"(?:password|token|key).*handling",
                r"security.*annotations"
            ],
            QueryIntent.PERFORMANCE_ANALYSIS: [
                r"performance.*(?:issues|problems|bottlenecks)",
                r"slow.*(?:functions|code|methods)",
                r"(?:find|show).*performance.*critical",
                r"latency.*(?:issues|problems)",
                r"optimization.*(?:opportunities|candidates)"
            ],
            QueryIntent.DEPENDENCY_ANALYSIS: [
                r"(?:dependencies|depends).*(?:of|for)\\s+(\\w+)",
                r"what.*uses.*(\\w+)",
                r"what.*depends.*(\\w+)",
                r"(?:find|show).*dependencies",
                r"impact.*(?:of|from).*(\\w+)"
            ],
            QueryIntent.TEAM_OWNERSHIP: [
                r"who.*owns.*(\\w+)",
                r"(?:team|owner).*(?:of|for).*(\\w+)",
                r"contact.*(?:for|about).*(\\w+)",
                r"responsible.*(?:for|team)"
            ],
            QueryIntent.DOMAIN_ANALYSIS: [
                r"(?:show|find).*(\\w+).*domain.*code",
                r"business.*domain.*(\\w+)",
                r"domain.*analysis.*(\\w+)",
                r"(\\w+).*domain.*entities"
            ],
            QueryIntent.ARCHITECTURAL_ANALYSIS: [
                r"architectural.*patterns",
                r"(?:mvc|repository|factory).*pattern",
                r"design.*patterns",
                r"architecture.*(?:analysis|overview)"
            ],
            QueryIntent.CODE_QUALITY: [
                r"code.*quality.*(?:issues|problems)",
                r"(?:complex|complicated).*(?:functions|code)",
                r"quality.*(?:metrics|scores)",
                r"(?:style|lint).*(?:issues|violations)"
            ],
            QueryIntent.SIMILAR_CODE: [
                r"similar.*(?:to|like).*(\\w+)",
                r"(?:find|show).*similar.*code",
                r"duplicate.*(?:functions|code)",
                r"related.*(?:functions|code)"
            ],
            QueryIntent.DOCUMENTATION: [
                r"(?:show|find).*documentation.*(\\w+)",
                r"docstring.*(?:for|of).*(\\w+)",
                r"(?:help|docs).*(?:for|about).*(\\w+)",
                r"explain.*(\\w+).*(?:function|class|method)"
            ],
            QueryIntent.CODE_OVERVIEW: [
                r"what.*does.*(?:this|the).*code.*do",
                r"what.*is.*(?:this|the).*code.*for",
                r"explain.*(?:this|the).*(?:code|codebase|system|project)",
                r"describe.*(?:this|the).*(?:code|codebase|system|project)",
                r"overview.*of.*(?:this|the).*(?:code|codebase|system|project)",
                r"summary.*of.*(?:this|the).*(?:code|codebase|system|project)",
                r"what.*does.*(?:this|the).*(?:project|codebase|system).*do",
                r"purpose.*of.*(?:this|the).*(?:code|codebase|system|project)",
                r"functionality.*of.*(?:this|the).*(?:code|codebase|system|project)",
                r"what.*(?:is|does).*(?:this|the).*(?:codebase|system|project)"
            ]
        }
        
        self.domain_keywords = {
            "authentication": ["auth", "login", "user", "password", "token", "session"],
            "payment": ["payment", "billing", "charge", "transaction", "invoice"],
            "database": ["db", "sql", "query", "model", "table", "migration"],
            "api": ["api", "endpoint", "route", "handler", "request", "response"],
            "security": ["security", "encrypt", "hash", "validate", "sanitize"],
            "performance": ["performance", "cache", "optimize", "latency", "speed"]
        }
    
    def can_handle(self, query: str) -> bool:
        """Check if this is a natural language query."""
        # Simple heuristic: if query contains common English words and is longer than a single term
        english_indicators = ["find", "show", "what", "where", "who", "how", "is", "are", "the", "in", "of", "for"]
        words = query.lower().split()
        return len(words) > 1 and any(indicator in words for indicator in english_indicators)
    
    def process(self, query: str, knowledge_graph: CodeKnowledgeGraph) -> QueryResult:
        """Process natural language query and return structured results."""
        start_time = time.time()
        
        # Detect intent
        intent, extracted_terms = self._detect_intent(query)
        
        # Execute appropriate query based on intent
        if intent == QueryIntent.FIND_FUNCTION:
            entities = self._find_entities_by_name(knowledge_graph, extracted_terms, entity_type="function")
        elif intent == QueryIntent.FIND_CLASS:
            entities = self._find_entities_by_name(knowledge_graph, extracted_terms, entity_type="class")
        elif intent == QueryIntent.SECURITY_ANALYSIS:
            entities = self._find_security_critical_code(knowledge_graph)
        elif intent == QueryIntent.PERFORMANCE_ANALYSIS:
            entities = self._find_performance_critical_code(knowledge_graph)
        elif intent == QueryIntent.DEPENDENCY_ANALYSIS:
            entities = self._find_dependencies(knowledge_graph, extracted_terms)
        elif intent == QueryIntent.TEAM_OWNERSHIP:
            entities = self._find_by_team_ownership(knowledge_graph, extracted_terms)
        elif intent == QueryIntent.DOMAIN_ANALYSIS:
            entities = self._find_by_domain(knowledge_graph, extracted_terms)
        elif intent == QueryIntent.ARCHITECTURAL_ANALYSIS:
            entities = self._find_by_architectural_pattern(knowledge_graph, extracted_terms)
        elif intent == QueryIntent.CODE_QUALITY:
            entities = self._find_quality_issues(knowledge_graph)
        elif intent == QueryIntent.SIMILAR_CODE:
            entities = self._find_similar_code(knowledge_graph, query)
        elif intent == QueryIntent.DOCUMENTATION:
            entities = self._find_documented_code(knowledge_graph, extracted_terms)
        elif intent == QueryIntent.CODE_OVERVIEW:
            entities = self._generate_code_overview(knowledge_graph)
        else:
            # Fallback to semantic search
            entities = knowledge_graph.contextual_search(query, top_k=10)
        
        # Generate context and suggestions
        context = self._generate_context(entities, intent, query)
        suggestions = self._generate_suggestions(intent, entities, query)
        confidence = self._calculate_confidence(entities, intent, query)
        
        execution_time = time.time() - start_time
        
        return QueryResult(
            entities=entities,
            confidence=confidence,
            query_time=execution_time,
            metadata={
                'query': query,
                'intent': intent,
                'context': context,
                'suggestions': suggestions
            }
        )
    
    def _detect_intent(self, query: str) -> Tuple[str, List[str]]:
        """Detect query intent and extract relevant terms."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    # Extract captured groups as relevant terms
                    terms = [group for group in match.groups() if group]
                    return intent, terms
        
        # If no specific intent detected, extract domain-related terms
        extracted_terms = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                extracted_terms.append(domain)
        
        return QueryIntent.SIMILAR_CODE, extracted_terms  # Default fallback
    
    def _find_entities_by_name(self, kg: CodeKnowledgeGraph, terms: List[str], entity_type: str = None) -> List[Dict]:
        """Find entities by name or partial name match."""
        if not terms:
            return []
        
        conn = sqlite3.connect(kg.db_path)
        
        conditions = []
        params = []
        
        for term in terms:
            conditions.append("(name LIKE ? OR signature LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%"])
        
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)
        
        where_clause = " AND ".join(conditions)
        
        cursor = conn.execute(f"""
            SELECT id, name, type, file_path, complexity, business_domain, 
                   team_owner, importance_score, docstring
            FROM entities 
            WHERE {where_clause}
            ORDER BY importance_score DESC
            LIMIT 15
        """, params)
        
        results = []
        for row in cursor:
            results.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'file_path': row[3],
                'complexity': row[4],
                'business_domain': row[5],
                'team_owner': row[6],
                'importance_score': row[7],
                'docstring': row[8]
            })
        
        conn.close()
        return results
    
    def _find_security_critical_code(self, kg: CodeKnowledgeGraph) -> List[Dict]:
        """Find security-critical code entities."""
        return kg.contextual_search("security authentication", {
            'min_importance': 1.5
        })
    
    def _find_performance_critical_code(self, kg: CodeKnowledgeGraph) -> List[Dict]:
        """Find performance-critical code entities."""
        conn = sqlite3.connect(kg.db_path)
        
        cursor = conn.execute("""
            SELECT id, name, type, file_path, complexity, business_domain,
                   team_owner, importance_score, annotations
            FROM entities 
            WHERE (annotations LIKE '%performance%' 
                   OR annotations LIKE '%critical_path%'
                   OR complexity > 10)
            ORDER BY importance_score DESC, complexity DESC
            LIMIT 15
        """)
        
        results = []
        for row in cursor:
            results.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'file_path': row[3],
                'complexity': row[4],
                'business_domain': row[5],
                'team_owner': row[6],
                'importance_score': row[7],
                'annotations': row[8]
            })
        
        conn.close()
        return results
    
    def _find_dependencies(self, kg: CodeKnowledgeGraph, terms: List[str]) -> List[Dict]:
        """Find dependencies for specified entities."""
        if not terms:
            return []
        
        # First find the entities mentioned in terms
        entities = self._find_entities_by_name(kg, terms)
        
        results = []
        for entity in entities[:5]:  # Limit to first 5 matches
            # Get dependencies using graph search
            graph_result = kg.graph_search(entity['id'], ['depends_on', 'calls'], depth=2)
            
            entity['dependencies'] = graph_result
            results.append(entity)
        
        return results
    
    def _find_by_team_ownership(self, kg: CodeKnowledgeGraph, terms: List[str]) -> List[Dict]:
        """Find entities by team ownership."""
        if not terms:
            return []
        
        conn = sqlite3.connect(kg.db_path)
        
        conditions = []
        params = []
        
        for term in terms:
            conditions.append("team_owner LIKE ?")
            params.append(f"%{term}%")
        
        where_clause = " OR ".join(conditions)
        
        cursor = conn.execute(f"""
            SELECT id, name, type, file_path, complexity, business_domain,
                   team_owner, importance_score
            FROM entities 
            WHERE {where_clause}
            ORDER BY importance_score DESC
            LIMIT 20
        """, params)
        
        results = []
        for row in cursor:
            results.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'file_path': row[3],
                'complexity': row[4],
                'business_domain': row[5],
                'team_owner': row[6],
                'importance_score': row[7]
            })
        
        conn.close()
        return results
    
    def _find_by_domain(self, kg: CodeKnowledgeGraph, terms: List[str]) -> List[Dict]:
        """Find entities by business domain."""
        domain_filters = {}
        for term in terms:
            if term.lower() in ['authentication', 'auth']:
                domain_filters['business_domain'] = 'Authentication'
            elif term.lower() in ['payment', 'billing']:
                domain_filters['business_domain'] = 'Payment'
            elif term.lower() in ['database', 'db']:
                domain_filters['business_domain'] = 'Database'
            elif term.lower() in ['api']:
                domain_filters['business_domain'] = 'API'
        
        if domain_filters:
            return kg.contextual_search(" ".join(terms), domain_filters)
        else:
            return kg.contextual_search(" ".join(terms))
    
    def _find_by_architectural_pattern(self, kg: CodeKnowledgeGraph, terms: List[str]) -> List[Dict]:
        """Find entities by architectural pattern."""
        conn = sqlite3.connect(kg.db_path)
        
        cursor = conn.execute("""
            SELECT architectural_pattern, COUNT(*) as count,
                   AVG(complexity) as avg_complexity
            FROM entities 
            WHERE architectural_pattern IS NOT NULL
            GROUP BY architectural_pattern
            ORDER BY count DESC
        """)
        
        patterns = []
        for row in cursor:
            patterns.append({
                'pattern': row[0],
                'entity_count': row[1],
                'avg_complexity': round(row[2], 2) if row[2] else 0
            })
        
        conn.close()
        return patterns
    
    def _find_quality_issues(self, kg: CodeKnowledgeGraph) -> List[Dict]:
        """Find code quality issues."""
        conn = sqlite3.connect(kg.db_path)
        
        cursor = conn.execute("""
            SELECT id, name, type, file_path, complexity, business_domain,
                   team_owner, quality_issues
            FROM entities 
            WHERE (quality_issues IS NOT NULL AND quality_issues != '[]'
                   OR complexity > 15)
            ORDER BY complexity DESC
            LIMIT 20
        """)
        
        results = []
        for row in cursor:
            try:
                quality_issues = json.loads(row[7]) if row[7] else []
            except:
                quality_issues = []
            
            results.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'file_path': row[3],
                'complexity': row[4],
                'business_domain': row[5],
                'team_owner': row[6],
                'quality_issues': quality_issues
            })
        
        conn.close()
        return results
    
    def _find_similar_code(self, kg: CodeKnowledgeGraph, query: str) -> List[Dict]:
        """Find similar code using semantic search."""
        return kg.contextual_search(query, top_k=15)
    
    def _find_documented_code(self, kg: CodeKnowledgeGraph, terms: List[str]) -> List[Dict]:
        """Find well-documented code entities."""
        if terms:
            entities = self._find_entities_by_name(kg, terms)
            # Filter for entities with docstrings
            return [e for e in entities if e.get('docstring')]
        
        conn = sqlite3.connect(kg.db_path)
        
        cursor = conn.execute("""
            SELECT id, name, type, file_path, complexity, business_domain,
                   team_owner, docstring
            FROM entities 
            WHERE docstring IS NOT NULL AND docstring != ''
            ORDER BY LENGTH(docstring) DESC
            LIMIT 15
        """)
        
        results = []
        for row in cursor:
            results.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'file_path': row[3],
                'complexity': row[4],
                'business_domain': row[5],
                'team_owner': row[6],
                'docstring': row[7]
            })
        
        conn.close()
        return results
    
    def _generate_code_overview(self, kg: CodeKnowledgeGraph) -> List[Dict]:
        """Generate a comprehensive overview of the codebase."""
        stats = kg.get_statistics()
        
        # Get representative entities from different domains
        conn = sqlite3.connect(kg.db_path)
        
        cursor = conn.execute("""
            SELECT name, type, business_domain, complexity, docstring, importance_score
            FROM entities 
            WHERE importance_score > 1.0
            ORDER BY importance_score DESC, complexity DESC
            LIMIT 10
        """)
        
        key_entities = []
        for row in cursor:
            key_entities.append({
                'name': row[0],
                'type': row[1],
                'business_domain': row[2],
                'complexity': row[3],
                'docstring': row[4][:100] + "..." if row[4] and len(row[4]) > 100 else row[4],
                'importance_score': row[5]
            })
        
        conn.close()
        
        # Combine statistics and key entities
        overview = {
            'statistics': stats,
            'key_entities': key_entities
        }
        
        return [overview]
    
    def _generate_context(self, entities: List[Dict], intent: str, query: str) -> Dict[str, Any]:
        """Generate contextual information about the results."""
        context = {
            'total_results': len(entities),
            'intent': intent,
            'query_type': 'natural_language'
        }
        
        if entities:
            # Analyze domains
            domains = {}
            teams = {}
            complexities = []
            
            for entity in entities:
                if isinstance(entity, dict):
                    domain = entity.get('business_domain')
                    if domain:
                        domains[domain] = domains.get(domain, 0) + 1
                    
                    team = entity.get('team_owner')
                    if team:
                        teams[team] = teams.get(team, 0) + 1
                    
                    complexity = entity.get('complexity', 0)
                    if complexity:
                        complexities.append(complexity)
            
            context['domains_found'] = domains
            context['teams_involved'] = teams
            if complexities:
                context['avg_complexity'] = round(sum(complexities) / len(complexities), 2)
                context['max_complexity'] = max(complexities)
        
        return context
    
    def _generate_suggestions(self, intent: str, entities: List[Dict], query: str) -> List[str]:
        """Generate follow-up suggestions based on results."""
        suggestions = []
        
        if intent == QueryIntent.FIND_FUNCTION and entities:
            suggestions.append("Show dependencies of this function")
            suggestions.append("Find similar functions")
            suggestions.append("Check who owns this code")
        
        elif intent == QueryIntent.SECURITY_ANALYSIS:
            suggestions.append("Show team responsible for security")
            suggestions.append("Find authentication patterns")
            suggestions.append("Check security compliance")
        
        elif intent == QueryIntent.PERFORMANCE_ANALYSIS:
            suggestions.append("Show complex functions")
            suggestions.append("Find optimization opportunities")
            suggestions.append("Check performance annotations")
        
        elif intent == QueryIntent.TEAM_OWNERSHIP:
            suggestions.append("Show team's code quality metrics")
            suggestions.append("Find team's most complex code")
            suggestions.append("Show team's architectural patterns")
        
        elif intent == QueryIntent.CODE_OVERVIEW:
            suggestions.append("Find the most complex functions")
            suggestions.append("Show security-critical code")
            suggestions.append("Analyze architectural patterns")
        
        else:
            suggestions.extend([
                "Try a more specific search",
                "Search by business domain",
                "Find by team ownership"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _calculate_confidence(self, entities: List[Dict], intent: str, query: str) -> float:
        """Calculate confidence score for the results."""
        if not entities:
            return 0.1
        
        base_confidence = 0.7
        
        # Boost confidence for specific intents with good matches
        if intent in [QueryIntent.FIND_FUNCTION, QueryIntent.FIND_CLASS] and entities:
            # Check if we found exact name matches
            query_terms = query.lower().split()
            for entity in entities[:3]:
                if isinstance(entity, dict) and entity.get('name'):
                    if any(term in entity['name'].lower() for term in query_terms):
                        base_confidence += 0.2
                        break
        
        return min(base_confidence, 1.0)


class CodebaseAIAgent:
    """
    Main AI agent that provides natural language interface to the code knowledge graph.
    
    This class coordinates different query processors and provides a unified interface
    for asking questions about the codebase in natural language.
    """
    
    def __init__(self, knowledge_graph: CodeKnowledgeGraph):
        """
        Initialize the AI agent with a knowledge graph.
        
        Args:
            knowledge_graph (CodeKnowledgeGraph): The knowledge graph to query
        """
        self.knowledge_graph = knowledge_graph
        self.processors = [
            NaturalLanguageQueryProcessor()
        ]
    
    def ask(self, query: str) -> QueryResult:
        """
        Process a natural language query about the codebase.
        
        Args:
            query (str): Natural language query
            
        Returns:
            QueryResult: Structured result with entities and context
        """
        # Find the appropriate processor
        for processor in self.processors:
            if processor.can_handle(query):
                return processor.process(query, self.knowledge_graph)
        
        # Fallback to basic search
        entities = self.knowledge_graph.contextual_search(query)
        return QueryResult(
            entities=entities,
            confidence=0.5,
            query_time=0.0,
            metadata={'query': query, 'intent': 'fallback'}
        )
