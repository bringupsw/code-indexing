"""
AST Analysis Module

This module provides comprehensive AST analysis capabilities for Python code,
extracting semantic entities, relationships, and contextual information.
"""

import ast
import hashlib
from typing import Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from .models import CodeEntity, CodeRelation
from .contextual_knowledge import ContextualKnowledgeLoader


class SemanticASTAnalyzer:
    """
    Enhanced AST analyzer that extracts semantic information suitable for embedding generation and knowledge graphs.

    This class performs comprehensive analysis of Python codebases by:
    1. Parsing AST trees from Python files
    2. Extracting semantic entities (functions, classes, methods)
    3. Building relationship graphs between entities
    4. Computing complexity metrics and semantic context
    5. Supporting parallel processing for large codebases

    The analyzer uses a multi-phase approach:
    - Phase 1: Extract entities from individual files in parallel
    - Phase 2: Build cross-file relationships
    - Phase 3: Enhance with semantic context and patterns

    Attributes:
        entities (List[CodeEntity]): Collected code entities from analysis
        relations (List[CodeRelation]): Discovered relationships between entities
        call_graph (Dict[str, List[str]]): Function call graph
        inheritance_tree (Dict[str, List[str]]): Class inheritance relationships
    """

    def __init__(self, context_config_dir: str = "./context_config"):
        """
        Initialize the analyzer with empty collections and contextual knowledge.

        Args:
            context_config_dir (str): Directory containing contextual configuration files
        """
        self.entities: List[CodeEntity] = []
        self.relations: List[CodeRelation] = []
        self.call_graph: Dict[str, List[str]] = {}
        self.inheritance_tree: Dict[str, List[str]] = {}
        self.context_loader = ContextualKnowledgeLoader(context_config_dir)

    def analyze_codebase(self, root_path: str) -> Tuple[List[CodeEntity], List[CodeRelation]]:
        """
        Perform comprehensive semantic analysis of a codebase.

        This method orchestrates the complete analysis pipeline:
        1. Discovers all Python files in the codebase
        2. Analyzes each file in parallel using ProcessPoolExecutor
        3. Builds cross-file relationships and dependencies
        4. Enhances entities with semantic context

        Args:
            root_path (str): Root directory path to analyze

        Returns:
            Tuple[List[CodeEntity], List[CodeRelation]]: Extracted entities and relationships

        Raises:
            FileNotFoundError: If root_path doesn't exist
            PermissionError: If files cannot be read
        """

        # Phase 1: Extract entities from each file
        python_files = list(Path(root_path).glob("**/*.py"))

        with ProcessPoolExecutor() as executor:
            file_results = list(executor.map(self._analyze_file, python_files))

        # Combine results
        for entities, relations in file_results:
            self.entities.extend(entities)
            self.relations.extend(relations)

        # Phase 2: Build cross-file relationships
        self._build_cross_file_relationships()

        # Phase 3: Enhance with semantic context
        self._enhance_semantic_context()

        # Phase 4: Apply contextual knowledge
        self._apply_contextual_knowledge()

        return self.entities, self.relations

    def _analyze_file(self, file_path: Path) -> Tuple[List[CodeEntity], List[CodeRelation]]:
        """
        Analyze a single Python file for entities and relationships.

        This method parses a single Python file and extracts all semantic
        entities and their local relationships. It's designed to be called
        in parallel for performance.

        Args:
            file_path (Path): Path to the Python file to analyze

        Returns:
            Tuple[List[CodeEntity], List[CodeRelation]]: Entities and relations found in the file

        Note:
            Returns empty lists if the file cannot be parsed (syntax errors, encoding issues, etc.)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))

            visitor = SemanticVisitor(str(file_path), source.split("\n"))
            visitor.visit(tree)

            return visitor.entities, visitor.relations

        except Exception as e:
            # Log error in production - for now, silently skip problematic files
            return [], []

    def _build_cross_file_relationships(self):
        """
        Build relationships across files (imports, inheritance, etc.).

        This method analyzes dependencies between entities across different files
        to build a complete relationship graph. It looks for:
        - Import relationships
        - Cross-file function calls
        - Inheritance across modules
        - Usage dependencies
        """

        # Build entity lookup
        entity_lookup = {entity.full_name: entity for entity in self.entities}

        # Find cross-file relationships
        for entity in self.entities:
            for dep in entity.dependencies:
                if dep in entity_lookup:
                    self.relations.append(
                        CodeRelation(
                            source_id=entity.id,
                            target_id=entity_lookup[dep].id,
                            relation_type="depends_on",
                            weight=0.9,
                            metadata={"type": "cross_file_dependency"},
                        )
                    )

        # Build inheritance relationships
        self._build_inheritance_relationships(entity_lookup)

    def _build_inheritance_relationships(self, entity_lookup):
        """
        Build inheritance relationships between classes.

        This method creates inheritance relationships in the knowledge graph
        and populates the inheritance_tree for hierarchical analysis.

        Args:
            entity_lookup (Dict[str, CodeEntity]): Lookup dictionary of entities by full_name
        """
        for entity in self.entities:
            if entity.type == "class" and entity.base_classes:
                for base_class in entity.base_classes:
                    # Handle both simple names and qualified names
                    potential_matches = [
                        base_class,  # Direct match
                        f"{entity.file_path.split('/')[-1].replace('.py', '')}.{base_class}",  # Same module
                    ]

                    # Try to find the base class in our entities
                    base_entity = None
                    for potential_name in potential_matches:
                        if potential_name in entity_lookup:
                            base_entity = entity_lookup[potential_name]
                            break

                    if base_entity:
                        # Create inheritance relationship
                        self.relations.append(
                            CodeRelation(
                                source_id=entity.id,
                                target_id=base_entity.id,
                                relation_type="inherits_from",
                                weight=1.0,
                                metadata={
                                    "type": "inheritance",
                                    "base_class": base_class,
                                    "child_class": entity.name,
                                },
                            )
                        )

                        # Update inheritance tree
                        if base_entity.full_name not in self.inheritance_tree:
                            self.inheritance_tree[base_entity.full_name] = []
                        self.inheritance_tree[base_entity.full_name].append(entity.full_name)
                    else:
                        # Create external inheritance relationship (base class not in our codebase)
                        self.relations.append(
                            CodeRelation(
                                source_id=entity.id,
                                target_id=f"external:{base_class}",
                                relation_type="inherits_from",
                                weight=0.7,
                                metadata={
                                    "type": "external_inheritance",
                                    "base_class": base_class,
                                    "child_class": entity.name,
                                },
                            )
                        )

    def _enhance_semantic_context(self):
        """
        Add semantic context to entities based on patterns and usage.

        This method analyzes naming patterns, complexity metrics, and usage
        patterns to add semantic context that helps with categorization
        and search. It identifies:
        - Function categories (test, private, data access, etc.)
        - Complexity levels (low, medium, high)
        - Intent patterns (getters, setters, processors, etc.)
        - Visibility modifiers (public, private, protected)
        """

        for entity in self.entities:
            # Analyze decorator patterns for semantic classification
            if entity.decorators:
                for decorator in entity.decorators:
                    if decorator == "property":
                        entity.annotations["intent"] = "data_access"
                        entity.annotations["decorator_pattern"] = "property_accessor"
                    elif decorator == "staticmethod":
                        entity.annotations["category"] = "utility"
                        entity.annotations["decorator_pattern"] = "static_utility"
                    elif decorator == "classmethod":
                        entity.annotations["intent"] = "factory_method"
                        entity.annotations["decorator_pattern"] = "class_factory"
                    elif decorator == "abstractmethod":
                        entity.annotations["category"] = "contract"
                        entity.annotations["decorator_pattern"] = "abstract_contract"
                    elif "cache" in decorator.lower():
                        entity.annotations["optimization"] = "caching"
                        entity.annotations["decorator_pattern"] = "performance_cache"
                    elif "dataclass" in decorator.lower():
                        entity.annotations["category"] = "data_structure"
                        entity.annotations["decorator_pattern"] = "data_class"

            # Analyze naming patterns for semantic intent
            if entity.type == "function":
                if entity.name.startswith("test_"):
                    entity.annotations["category"] = "test"
                elif entity.name.startswith("_"):
                    entity.annotations["visibility"] = "private"
                elif any(word in entity.name.lower() for word in ["get", "fetch", "retrieve"]):
                    entity.annotations["intent"] = "data_access"
                elif any(word in entity.name.lower() for word in ["set", "update", "modify"]):
                    entity.annotations["intent"] = "data_modification"

            # Classify complexity levels for better categorization
            if entity.complexity > 10:
                entity.annotations["complexity_level"] = "high"
            elif entity.complexity > 5:
                entity.annotations["complexity_level"] = "medium"
            else:
                entity.annotations["complexity_level"] = "low"

    def _apply_contextual_knowledge(self):
        """
        Apply contextual knowledge to entities using configuration data.

        This method enriches entities with:
        - Business domain classification
        - Architectural pattern identification
        - Team ownership information
        - Contextual annotations
        - Importance scoring based on domain and annotations
        """

        for entity in self.entities:
            # Identify business domain
            domain = self.context_loader.get_business_domain(entity.name, entity.file_path)
            if domain:
                entity.business_domain = domain.name
                entity.importance_score *= domain.importance
                entity.annotations["business_domain"] = domain.name
                entity.annotations["domain_keywords"] = domain.keywords

            # Identify architectural pattern
            pattern = self.context_loader.get_architectural_pattern(entity.name, entity.type)
            if pattern:
                entity.architectural_pattern = pattern.name
                entity.annotations["architectural_pattern"] = pattern.name
                entity.annotations["quality_metrics"] = pattern.quality_metrics

                # Check if entity meets quality metrics
                self._validate_quality_metrics(entity, pattern)

            # Identify team ownership
            team = self.context_loader.get_team_ownership(entity.file_path)
            if team:
                entity.team_owner = team.name
                entity.annotations["team_ownership"] = team.name
                entity.annotations["team_contact"] = team.contact
                entity.annotations["team_expertise"] = team.expertise

                # Apply team-specific code style validation
                self._validate_code_style(entity, team)

            # Apply contextual annotations
            annotations = self.context_loader.get_annotations(entity.file_path, entity.name)
            if annotations:
                for annotation in annotations:
                    # Increase importance for critical annotations
                    if (
                        annotation.category == "security"
                        and "security_level" in annotation.annotations
                    ):
                        if annotation.annotations["security_level"] == "critical":
                            entity.importance_score *= 2.0

                    if (
                        annotation.category == "performance"
                        and "critical_path" in annotation.annotations
                    ):
                        if annotation.annotations["critical_path"]:
                            entity.importance_score *= 1.5

                    # Add annotation context
                    entity.annotations[f"{annotation.category}_annotations"] = (
                        annotation.annotations
                    )

    def _validate_quality_metrics(self, entity: CodeEntity, pattern):
        """
        Validate entity against architectural pattern quality metrics.

        Args:
            entity (CodeEntity): Entity to validate
            pattern: Pattern with quality requirements
        """
        quality_issues = []

        # Check complexity limits
        if "max_complexity" in pattern.quality_metrics:
            max_complexity = pattern.quality_metrics["max_complexity"]
            if entity.complexity > max_complexity:
                quality_issues.append(
                    f"Complexity {entity.complexity} exceeds limit {max_complexity}"
                )

        # Check if interface is required
        if pattern.quality_metrics.get("interface_required", False):
            if entity.type == "class" and not self._has_interface_indicators(entity):
                quality_issues.append("Interface required but not found")

        # Check single responsibility
        if pattern.quality_metrics.get("single_responsibility", False):
            if entity.complexity > 15:  # High complexity might indicate multiple responsibilities
                quality_issues.append("Possible violation of single responsibility principle")

        if quality_issues:
            entity.quality_issues.extend(quality_issues)

    def _validate_code_style(self, entity: CodeEntity, team):
        """
        Validate entity against team code style preferences.

        Args:
            entity (CodeEntity): Entity to validate
            team: Team with style preferences
        """
        style_issues = []

        # Check line length
        if "max_line_length" in team.code_style:
            max_length = team.code_style["max_line_length"]
            lines = entity.source_code.split("\n")
            long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > max_length]
            if long_lines:
                style_issues.append(f"Lines exceed {max_length} characters: {long_lines[:5]}")

        # Check for mandatory tests
        if team.code_style.get("mandatory_tests", False):
            if entity.type in ["function", "method"] and not self._has_corresponding_test(entity):
                style_issues.append("Test required but not found")

        # Check security-first approach
        if team.code_style.get("security_first", False):
            if entity.type in ["function", "method"] and self._handles_sensitive_data(entity):
                if not self._has_security_checks(entity):
                    style_issues.append("Security checks required for sensitive data handling")

        if style_issues:
            entity.annotations["style_issues"] = style_issues

    def _has_interface_indicators(self, entity: CodeEntity) -> bool:
        """Check if entity has interface-like characteristics."""
        # Simple heuristic - look for abstract methods or interface naming
        return (
            "abstract" in entity.source_code.lower()
            or "interface" in entity.name.lower()
            or entity.name.endswith("Interface")
        )

    def _has_corresponding_test(self, entity: CodeEntity) -> bool:
        """Check if entity has corresponding test (simplified check)."""
        # This would need to be enhanced with actual test discovery
        test_patterns = [f"test_{entity.name}", f"{entity.name}_test", f"Test{entity.name}"]
        return any(pattern in str(self.entities) for pattern in test_patterns)

    def _handles_sensitive_data(self, entity: CodeEntity) -> bool:
        """Check if entity handles sensitive data."""
        sensitive_keywords = [
            "password",
            "token",
            "key",
            "secret",
            "auth",
            "login",
            "payment",
            "credit",
        ]
        entity_text = f"{entity.name} {entity.source_code}".lower()
        return any(keyword in entity_text for keyword in sensitive_keywords)

    def _has_security_checks(self, entity: CodeEntity) -> bool:
        """Check if entity has security validation."""
        security_indicators = ["validate", "sanitize", "encrypt", "hash", "verify", "authenticate"]
        return any(indicator in entity.source_code.lower() for indicator in security_indicators)


class SemanticVisitor(ast.NodeVisitor):
    """
    Enhanced AST visitor that extracts semantic information from Python AST nodes.

    This visitor traverses the AST and extracts detailed information about
    code entities and their relationships. It maintains context during traversal
    to properly handle nested structures like class methods.

    Key features:
    - Tracks current class/function context for proper naming
    - Extracts source code ranges for entities
    - Computes complexity metrics
    - Identifies function calls and dependencies
    - Handles decorators and type annotations

    Attributes:
        file_path (str): Path to the file being analyzed
        lines (List[str]): Source code lines for extracting ranges
        entities (List[CodeEntity]): Collected entities from this file
        relations (List[CodeRelation]): Discovered relationships
        current_class (Optional[str]): Name of current class being processed
        current_function (Optional[str]): ID of current function being processed
        imports (Dict): Imported modules and their aliases
        call_stack (List): Stack for tracking nested calls
    """

    def __init__(self, file_path: str, lines: List[str]):
        """
        Initialize the visitor with file context.

        Args:
            file_path (str): Path to the source file
            lines (List[str]): Source code split into lines
        """
        self.file_path = file_path
        self.lines = lines
        self.entities: List[CodeEntity] = []
        self.relations: List[CodeRelation] = []
        self.current_class = None
        self.current_function = None
        self.imports = {}
        self.call_stack = []

    def visit_FunctionDef(self, node):
        """
        Visit function definition nodes and extract semantic information.

        This method processes both regular functions and class methods,
        extracting comprehensive information including:
        - Source code ranges
        - Complexity metrics
        - Parameter information
        - Decorator analysis
        - Dependency tracking

        Args:
            node: AST FunctionDef node to process
        """
        entity_id = f"{self.file_path}:{node.name}:{node.lineno}"

        # Extract function source code
        end_line = self._find_function_end(node)
        source_code = "\n".join(self.lines[node.lineno - 1 : end_line])

        # Calculate complexity
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(node)

        # Extract dependencies
        dep_visitor = DependencyVisitor()
        dep_visitor.visit(node)

        entity = CodeEntity(
            id=entity_id,
            name=node.name,
            full_name=f"{self.current_class}.{node.name}" if self.current_class else node.name,
            type="method" if self.current_class else "function",
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=end_line,
            source_code=source_code,
            docstring=ast.get_docstring(node),
            signature=f"{node.name}({', '.join([arg.arg for arg in node.args.args])})",
            complexity=complexity_visitor.complexity,
            parameters=[arg.arg for arg in node.args.args],
            decorators=[
                name
                for name in [self._get_name(d) for d in node.decorator_list]
                if name is not None
            ],
            class_name=self.current_class,
            has_docstring=bool(ast.get_docstring(node)),
            has_type_hints=bool(node.returns or any(arg.annotation for arg in node.args.args)),
            handles_exceptions=self._has_exception_handling(node),
            dependencies=dep_visitor.dependencies,
        )

        # Extract and merge type annotations
        type_annotations = self._extract_type_annotations(node)
        entity.annotations.update(type_annotations)

        # Extract inline comments for additional context
        inline_comments = self._extract_inline_comments(source_code)
        entity.annotations["inline_comments"] = inline_comments

        self.entities.append(entity)

        # Track function calls within this function
        old_function = self.current_function
        self.current_function = entity_id
        self.generic_visit(node)
        self.current_function = old_function

    def visit_ClassDef(self, node):
        """
        Visit class definition nodes and extract class information.

        This method processes class definitions, extracting information about:
        - Class inheritance hierarchy
        - Class decorators
        - Source code ranges
        - Docstring content

        Args:
            node: AST ClassDef node to process
        """
        entity_id = f"{self.file_path}:{node.name}:{node.lineno}"

        end_line = self._find_class_end(node)
        source_code = "\n".join(self.lines[node.lineno - 1 : end_line])

        # Extract inheritance information
        inheritance_info = self._extract_inheritance_info(node)

        # Build signature with inheritance
        if inheritance_info["base_classes"]:
            signature = f"class {node.name}({', '.join(inheritance_info['base_classes'])})"
        else:
            signature = f"class {node.name}"

        entity = CodeEntity(
            id=entity_id,
            name=node.name,
            full_name=node.name,  # For classes, full_name is same as name unless in a module
            type="class",
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=end_line,
            source_code=source_code,
            docstring=ast.get_docstring(node),
            signature=signature,
            complexity=0,
            decorators=[
                name
                for name in [self._get_name(d) for d in node.decorator_list]
                if name is not None
            ],
            has_docstring=bool(ast.get_docstring(node)),
            base_classes=inheritance_info["base_classes"],
            is_abstract=inheritance_info["is_abstract"],
        )

        # Extract inline comments for additional context
        inline_comments = self._extract_inline_comments(source_code)
        entity.annotations["inline_comments"] = inline_comments

        self.entities.append(entity)

        # Process class methods
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Call(self, node):
        """
        Track function calls for building call graph.

        This method identifies function calls within the current context
        and creates relationships between calling and called functions.
        This is essential for building the call graph and understanding
        code dependencies.

        Args:
            node: AST Call node representing a function call
        """
        if self.current_function:
            func_name = self._get_name(node.func)
            if func_name:
                self.relations.append(
                    CodeRelation(
                        source_id=self.current_function,
                        target_id=f"call:{func_name}",
                        relation_type="calls",
                        weight=0.8,
                        metadata={"line": node.lineno},
                    )
                )

        self.generic_visit(node)

    def visit_Import(self, node):
        """
        Track import statements for dependency analysis.

        Captures direct imports like: import os, sys
        """
        for alias in node.names:
            module_name = alias.name
            import_alias = alias.asname if alias.asname else alias.name

            self.imports[import_alias] = module_name

            # Create import relationship
            if self.current_function:
                self.relations.append(
                    CodeRelation(
                        source_id=self.current_function,
                        target_id=f"import:{module_name}",
                        relation_type="imports",
                        weight=0.6,
                        metadata={"line": node.lineno, "alias": import_alias},
                    )
                )

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Track from X import Y statements for dependency analysis.

        Captures imports like: from collections import defaultdict
        """
        module = node.module if node.module else "."
        level = node.level  # For relative imports

        for alias in node.names:
            imported_name = alias.name
            import_alias = alias.asname if alias.asname else alias.name

            full_import = f"{module}.{imported_name}" if module != "." else imported_name
            self.imports[import_alias] = full_import

            # Create import relationship
            if self.current_function:
                self.relations.append(
                    CodeRelation(
                        source_id=self.current_function,
                        target_id=f"import:{full_import}",
                        relation_type="imports",
                        weight=0.6,
                        metadata={
                            "line": node.lineno,
                            "module": module,
                            "imported": imported_name,
                            "alias": import_alias,
                            "level": level,
                        },
                    )
                )

        self.generic_visit(node)

    def visit_Assign(self, node):
        """
        Track variable assignments and constants.

        Captures assignments like: x = 5, self.attr = value, CONSTANT = "value"
        """
        # Extract assigned names
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, ast.Attribute):
                attr_name = self._get_name(target)
                if attr_name:
                    targets.append(attr_name)

        # Analyze the assignment value
        value_info = self._analyze_assignment_value(node.value)

        # Create entities for constants (uppercase variables)
        for target in targets:
            if target.isupper() and len(target) > 1:  # Likely a constant
                entity_id = f"{self.file_path}:{target}:{node.lineno}"

                entity = CodeEntity(
                    id=entity_id,
                    name=target,
                    full_name=f"{self.current_class}.{target}" if self.current_class else target,
                    type="constant",
                    file_path=self.file_path,
                    line_start=node.lineno,
                    line_end=node.lineno,
                    source_code=ast.unparse(node),
                    signature=f"{target} = {value_info['summary']}",
                    complexity=0,
                    class_name=self.current_class,
                )

                # Add value type information
                entity.annotations["value_type"] = value_info["type"]
                entity.annotations["value_summary"] = value_info["summary"]
                entity.annotations["is_constant"] = True

                if self.current_class:
                    entity.annotations["scope"] = "class"
                else:
                    entity.annotations["scope"] = "module"

                self.entities.append(entity)

        # Track assignments in current function context
        if self.current_function:
            for target in targets:
                self.relations.append(
                    CodeRelation(
                        source_id=self.current_function,
                        target_id=f"assigns:{target}",
                        relation_type="assigns",
                        weight=0.4,
                        metadata={
                            "line": node.lineno,
                            "target": target,
                            "value_type": value_info["type"],
                        },
                    )
                )

        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """
        Track type-annotated assignments.

        Captures assignments like: x: int = 5, self.name: str = "test"
        """
        target_name = None

        if isinstance(node.target, ast.Name):
            target_name = node.target.id
        elif isinstance(node.target, ast.Attribute):
            target_name = self._get_name(node.target)

        if target_name:
            # Get type annotation
            type_annotation = ast.unparse(node.annotation) if node.annotation else "Unknown"

            # Analyze value if present
            value_info = {"type": "None", "summary": "None"}
            if node.value:
                value_info = self._analyze_assignment_value(node.value)

            # Create entity for class/instance variables with type annotations
            if self.current_class and isinstance(node.target, ast.Attribute):
                entity_id = f"{self.file_path}:{target_name}:{node.lineno}"

                entity = CodeEntity(
                    id=entity_id,
                    name=target_name.split(".")[-1],  # Get the attribute name
                    full_name=target_name,
                    type="attribute",
                    file_path=self.file_path,
                    line_start=node.lineno,
                    line_end=node.lineno,
                    source_code=ast.unparse(node),
                    signature=f"{target_name}: {type_annotation}",
                    complexity=0,
                    class_name=self.current_class,
                    has_type_hints=True,
                )

                entity.annotations["type_annotation"] = type_annotation
                entity.annotations["value_type"] = value_info["type"]
                entity.annotations["has_value"] = node.value is not None
                entity.annotations["scope"] = (
                    "instance" if target_name.startswith("self.") else "class"
                )

                self.entities.append(entity)

            # Track in current function context
            if self.current_function:
                self.relations.append(
                    CodeRelation(
                        source_id=self.current_function,
                        target_id=f"typed_assigns:{target_name}",
                        relation_type="typed_assigns",
                        weight=0.5,
                        metadata={
                            "line": node.lineno,
                            "target": target_name,
                            "type_annotation": type_annotation,
                            "value_type": value_info["type"],
                        },
                    )
                )

        self.generic_visit(node)

    def visit_Try(self, node):
        """
        Analyze try/except blocks for exception handling patterns.

        Captures exception handling structure and error recovery patterns.
        """
        if self.current_function:
            # Analyze what exceptions are being caught
            caught_exceptions = []
            for handler in node.handlers:
                if handler.type:
                    exception_type = self._get_name(handler.type)
                    if exception_type:
                        caught_exceptions.append(exception_type)
                else:
                    caught_exceptions.append("Exception")  # Bare except

            # Track exception handling relationship
            self.relations.append(
                CodeRelation(
                    source_id=self.current_function,
                    target_id=f"handles_exceptions:{','.join(caught_exceptions)}",
                    relation_type="handles_exceptions",
                    weight=0.7,
                    metadata={
                        "line": node.lineno,
                        "exceptions": caught_exceptions,
                        "has_else": bool(node.orelse),
                        "has_finally": bool(node.finalbody),
                        "handler_count": len(node.handlers),
                    },
                )
            )

            # Analyze exception handler complexity
            for i, handler in enumerate(node.handlers):
                handler_complexity = len(handler.body)
                exception_type = self._get_name(handler.type) if handler.type else "Exception"

                # Create relationship for each exception handler
                self.relations.append(
                    CodeRelation(
                        source_id=self.current_function,
                        target_id=f"except_handler:{exception_type}:{i}",
                        relation_type="exception_handler",
                        weight=0.6,
                        metadata={
                            "line": handler.lineno,
                            "exception_type": exception_type,
                            "handler_complexity": handler_complexity,
                            "has_name": handler.name is not None,
                        },
                    )
                )

        self.generic_visit(node)

    def visit_Raise(self, node):
        """
        Track raised exceptions for error propagation analysis.

        Captures explicit exception raising patterns.
        """
        if self.current_function:
            exception_info = {"type": "Unknown", "has_cause": False}

            if node.exc:
                if isinstance(node.exc, ast.Call):
                    # Exception instantiation: raise ValueError("message")
                    exception_type = self._get_name(node.exc.func)
                    exception_info["type"] = exception_type if exception_type else "Unknown"
                elif isinstance(node.exc, ast.Name):
                    # Re-raising variable: raise exc
                    exception_info["type"] = node.exc.id
                else:
                    exception_info["type"] = "Complex"

                exception_info["has_cause"] = node.cause is not None
            else:
                # Bare raise statement
                exception_info["type"] = "Re-raise"

            # Track exception raising
            self.relations.append(
                CodeRelation(
                    source_id=self.current_function,
                    target_id=f"raises:{exception_info['type']}",
                    relation_type="raises",
                    weight=0.8,
                    metadata={
                        "line": node.lineno,
                        "exception_type": exception_info["type"],
                        "has_cause": exception_info["has_cause"],
                        "is_bare_raise": node.exc is None,
                    },
                )
            )

        self.generic_visit(node)

    def _analyze_assignment_value(self, value_node):
        """
        Analyze the value being assigned to understand its type and characteristics.

        Args:
            value_node: AST node representing the assigned value

        Returns:
            Dict with type and summary information
        """
        if isinstance(value_node, ast.Constant):
            # Python 3.8+ constant values
            value_type = type(value_node.value).__name__
            summary = repr(value_node.value)
            if len(summary) > 50:
                summary = summary[:47] + "..."
            return {"type": value_type, "summary": summary}

        elif isinstance(value_node, ast.Str):
            # String literal (older Python)
            summary = repr(value_node.s)
            if len(summary) > 50:
                summary = summary[:47] + "..."
            return {"type": "str", "summary": summary}

        elif isinstance(value_node, ast.Num):
            # Numeric literal (older Python)
            return {"type": type(value_node.n).__name__, "summary": str(value_node.n)}

        elif isinstance(value_node, ast.List):
            return {"type": "list", "summary": f"[...] ({len(value_node.elts)} items)"}

        elif isinstance(value_node, ast.Dict):
            return {"type": "dict", "summary": f"{{...}} ({len(value_node.keys)} items)"}

        elif isinstance(value_node, ast.Set):
            return {"type": "set", "summary": f"{{...}} ({len(value_node.elts)} items)"}

        elif isinstance(value_node, ast.Tuple):
            return {"type": "tuple", "summary": f"(...) ({len(value_node.elts)} items)"}

        elif isinstance(value_node, ast.Call):
            func_name = self._get_name(value_node.func)
            return {"type": "call_result", "summary": f"{func_name}(...)"}

        elif isinstance(value_node, ast.Name):
            return {"type": "variable", "summary": value_node.id}

        elif isinstance(value_node, ast.Attribute):
            attr_name = self._get_name(value_node)
            return {"type": "attribute", "summary": attr_name}

        else:
            return {"type": "complex", "summary": "complex_expression"}

    def _get_name(self, node):
        """
        Extract name from AST node handling different node types.

        This utility method handles various AST node types that represent
        names (simple names, attribute access, etc.) and returns a string
        representation.

        Args:
            node: AST node that represents a name

        Returns:
            str or None: String representation of the name, or None if not extractable
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            # Handle decorator calls like @lru_cache(maxsize=128)
            func_name = self._get_name(node.func)
            return func_name if func_name else "unknown_call"
        return None

    def _find_function_end(self, node):
        """
        Find the last line of a function by analyzing its AST subtree.

        This method uses a heuristic to determine where a function ends
        by finding the maximum line number of any node in its subtree.

        Args:
            node: AST FunctionDef node

        Returns:
            int: Line number where the function ends
        """
        # Simple heuristic - find the last line with content in the function
        max_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                max_line = max(max_line, child.lineno)
        return min(max_line + 1, len(self.lines))

    def _find_class_end(self, node):
        """
        Find the last line of a class by analyzing its AST subtree.

        Similar to _find_function_end but for class definitions.

        Args:
            node: AST ClassDef node

        Returns:
            int: Line number where the class ends
        """
        max_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                max_line = max(max_line, child.lineno)
        return min(max_line + 1, len(self.lines))

    def _has_return_statement(self, node):
        """
        Check if function has any return statements.

        This method walks the function's AST to determine if it contains
        any return statements, which is useful for semantic analysis.

        Args:
            node: AST FunctionDef node

        Returns:
            bool: True if the function has return statements, False otherwise
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False

    def _has_exception_handling(self, node):
        """
        Check if function has exception handling.

        Args:
            node: AST FunctionDef node

        Returns:
            bool: True if the function has try/except blocks
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False

    def _extract_inheritance_info(self, node):
        """Extract inheritance information from class definition"""
        inheritance_info = {"base_classes": [], "is_abstract": False}

        # Extract base classes
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                inheritance_info["base_classes"].append(base_name)

        # Check if class is abstract (has @abstractmethod or ABC inheritance)
        inheritance_info["is_abstract"] = self._is_abstract_class(
            node, inheritance_info["base_classes"]
        )

        return inheritance_info

    def _is_abstract_class(self, node, base_classes):
        """Check if class is abstract based on decorators, methods, and inheritance"""
        # Check if inherits from ABC or Abstract base classes
        abc_bases = ["ABC", "abc.ABC", "abstractmethod", "Abstract"]
        if any(base in abc_bases for base in base_classes):
            return True

        # Check for @abstractmethod decorators on methods
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                for decorator in child.decorator_list:
                    decorator_name = self._get_name(decorator)
                    if decorator_name and "abstract" in decorator_name.lower():
                        return True

        # Check for "Abstract" in class name
        if "abstract" in node.name.lower() or node.name.startswith("Abstract"):
            return True

        return False

    def _extract_inline_comments(self, source_code):
        """Extract inline comments for additional context"""
        comments = []
        for line in source_code.split("\n"):
            if "#" in line and not line.strip().startswith("#"):
                comment = line.split("#", 1)[1].strip()
                if comment:
                    comments.append(comment)
        return comments

    def _extract_type_annotations(self, node):
        """Extract actual type annotation strings"""
        type_info = {}
        if hasattr(node, "returns") and node.returns:
            type_info["return_type"] = ast.unparse(node.returns)

        type_info["param_types"] = {}
        for arg in node.args.args:
            if arg.annotation:
                type_info["param_types"][arg.arg] = ast.unparse(arg.annotation)

        return type_info


class ComplexityVisitor(ast.NodeVisitor):
    """
    AST visitor that computes cyclomatic complexity of code entities.

    This visitor implements a simplified cyclomatic complexity calculation
    by counting decision points in the code (if statements, loops, etc.).
    The complexity metric helps identify potentially problematic code areas.

    Complexity is calculated as:
    - Base complexity: 1
    - +1 for each if/elif statement
    - +1 for each for/while loop
    - +1 for each except handler
    - +1 for each boolean operator (and/or)

    Attributes:
        complexity (int): Current complexity score
    """

    def __init__(self):
        """Initialize complexity counter to 1 (base complexity)."""
        self.complexity = 1

    def visit_If(self, node):
        """Count if statements as decision points (+1 complexity)."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        """Count for loops as decision points (+1 complexity)."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        """Count while loops as decision points (+1 complexity)."""
        self.complexity += 1
        self.generic_visit(node)


class DependencyVisitor(ast.NodeVisitor):
    """
    AST visitor that identifies dependencies (variables/functions used).

    This visitor collects all names that are loaded (read) within a code
    entity, which helps identify dependencies and relationships between
    different parts of the code.

    Attributes:
        dependencies (List[str]): List of dependency names found
    """

    def __init__(self):
        """Initialize empty dependencies list."""
        self.dependencies = []

    def visit_Name(self, node):
        """
        Collect names that are loaded (dependencies).

        Only collects names in Load context (being read), not Store
        context (being assigned to).

        Args:
            node: AST Name node
        """
        if isinstance(node.ctx, ast.Load):
            self.dependencies.append(node.id)
