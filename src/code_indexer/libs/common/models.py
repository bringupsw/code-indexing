"""
Data Models for Semantic Code Indexing

This module contains the core data structures used throughout the semantic indexing pipeline.
It includes models for business domains, architectural patterns, code entities, and relationships.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class BusinessDomain:
    """
    Represents a business domain with associated patterns and importance.

    Attributes:
        name (str): Domain name (e.g., "Authentication", "Payment")
        keywords (List[str]): Keywords associated with this domain
        patterns (List[str]): Regex patterns to match domain-related code
        importance (float): Importance multiplier for this domain
    """

    name: str
    keywords: List[str]
    patterns: List[str]
    importance: float


@dataclass
class ArchitecturalPattern:
    """
    Represents an architectural pattern with quality metrics.

    Attributes:
        name (str): Pattern name (e.g., "MVC", "Repository")
        indicators (List[str]): Keywords that indicate this pattern
        quality_metrics (Dict[str, Any]): Quality requirements for this pattern
        relationships (List[str]): Expected relationships in this pattern
    """

    name: str
    indicators: List[str]
    quality_metrics: Dict[str, Any]
    relationships: List[str]


@dataclass
class TeamOwnership:
    """
    Represents team ownership and preferences.

    Attributes:
        name (str): Team name
        owned_modules (List[str]): Module paths owned by this team
        expertise (List[str]): Areas of expertise
        contact (str): Contact information
        code_style (Dict[str, Any]): Code style preferences
    """

    name: str
    owned_modules: List[str]
    expertise: List[str]
    contact: str
    code_style: Dict[str, Any]


@dataclass
class ContextualAnnotation:
    """
    Represents contextual annotations for specific code entities.

    Attributes:
        file_path (str): Path to the file
        function_name (str): Name of the function
        category (str): Annotation category (e.g., "performance", "security")
        annotations (Dict[str, Any]): Specific annotations
    """

    file_path: str
    function_name: str
    category: str
    annotations: Dict[str, Any]


@dataclass
class CodeEntity:
    """
    Represents a semantic entity extracted from code (function, class, method, etc.).

    This is the core data structure that holds all information about a code entity,
    including its semantic metadata, contextual information, and quality metrics.

    Attributes:
        id (str): Unique identifier for the entity
        name (str): Entity name
        full_name (str): Full qualified name including module path
        type (str): Entity type ('function', 'class', 'method', 'variable')
        file_path (str): Path to the source file
        line_start (int): Starting line number
        line_end (int): Ending line number
        source_code (str): Raw source code
        docstring (Optional[str]): Docstring if available
        signature (str): Function/method signature
        complexity (int): Cyclomatic complexity score
        parameters (List[str]): Parameter names
        return_type (Optional[str]): Return type annotation
        decorators (List[str]): Applied decorators
        class_name (Optional[str]): Parent class name (for methods)
        imports (List[str]): Imported modules/functions used
        calls (List[str]): Functions/methods called
        variables (List[str]): Variables used/defined
        dependencies (List[str]): Dependencies identified in the code
        has_docstring (bool): Whether entity has documentation
        is_public (bool): Whether entity is public API
        is_tested (bool): Whether entity has corresponding tests
        handles_exceptions (bool): Whether entity handles exceptions
        has_type_hints (bool): Whether entity has type annotations
        base_classes (List[str]): Parent classes for inheritance (classes only)
        is_abstract (bool): Whether class is abstract or has abstract methods
        business_domain (Optional[str]): Associated business domain
        team_owner (Optional[str]): Team responsible for this code
        architectural_pattern (Optional[str]): Associated architectural pattern
        importance_score (float): Calculated importance score
        quality_issues (List[str]): Identified quality issues
        annotations (Dict[str, Any]): Additional contextual annotations
    """

    id: str
    name: str
    full_name: str
    type: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: Optional[str] = None
    signature: str = ""
    complexity: int = 1
    parameters: List[str] = None
    return_type: Optional[str] = None
    decorators: List[str] = None
    class_name: Optional[str] = None
    imports: List[str] = None
    calls: List[str] = None
    variables: List[str] = None
    dependencies: List[str] = None
    has_docstring: bool = False
    is_public: bool = True
    is_tested: bool = False
    handles_exceptions: bool = False
    has_type_hints: bool = False
    base_classes: List[str] = None
    is_abstract: bool = False
    business_domain: Optional[str] = None
    team_owner: Optional[str] = None
    architectural_pattern: Optional[str] = None
    importance_score: float = 1.0
    quality_issues: List[str] = None
    annotations: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.parameters is None:
            self.parameters = []
        if self.decorators is None:
            self.decorators = []
        if self.imports is None:
            self.imports = []
        if self.calls is None:
            self.calls = []
        if self.variables is None:
            self.variables = []
        if self.dependencies is None:
            self.dependencies = []
        if self.base_classes is None:
            self.base_classes = []
        if self.quality_issues is None:
            self.quality_issues = []
        if self.annotations is None:
            self.annotations = {}


@dataclass
class CodeRelation:
    """
    Represents a relationship between two code entities.

    Attributes:
        source_id (str): ID of the source entity
        target_id (str): ID of the target entity
        relation_type (str): Type of relationship
        weight (float): Strength of the relationship
        metadata (Dict[str, Any]): Additional relationship metadata
    """

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryResult:
    """
    Represents the result of a query operation.

    Attributes:
        entities (List[CodeEntity]): Matching entities
        confidence (float): Confidence score of the result
        query_time (float): Time taken to process the query
        metadata (Dict[str, Any]): Additional result metadata
    """

    entities: List[CodeEntity]
    confidence: float
    query_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryIntent:
    """
    Represents the detected intent of a natural language query.

    Attributes:
        intent_type (str): The type of intent detected
        confidence (float): Confidence in the intent detection
        entities (List[str]): Extracted entities from the query
        parameters (Dict[str, Any]): Query parameters
    """

    intent_type: str
    confidence: float
    entities: List[str] = None
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.entities is None:
            self.entities = []
        if self.parameters is None:
            self.parameters = {}
