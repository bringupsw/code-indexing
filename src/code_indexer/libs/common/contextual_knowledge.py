"""
Contextual Knowledge Loader

This module handles loading and managing contextual knowledge from configuration files,
including business domains, architectural patterns, team ownership, and annotations.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .models import BusinessDomain, ArchitecturalPattern, TeamOwnership, ContextualAnnotation


class ContextualKnowledgeLoader:
    """
    Loads and manages contextual knowledge from configuration files.

    This class reads YAML configuration files that define:
    - Business domains and their importance
    - Architectural patterns and quality metrics
    - Team ownership and expertise
    - Specific annotations for code entities
    """

    def __init__(self, config_dir: str = "./context_config"):
        """
        Initialize the contextual knowledge loader.

        Args:
            config_dir (str): Directory containing configuration YAML files
        """
        self.config_dir = Path(config_dir)
        self.business_domains: List[BusinessDomain] = []
        self.architectural_patterns: List[ArchitecturalPattern] = []
        self.teams: List[TeamOwnership] = []
        self.annotations: Dict[str, List[ContextualAnnotation]] = {}

        if self.config_dir.exists():
            self._load_configurations()
        else:
            print(
                f"⚠️  Context config directory '{config_dir}' not found. Using default configurations."
            )
            self._load_default_configurations()

    def _load_configurations(self):
        """Load all configuration files from the config directory."""
        # Load business domains
        domains_file = self.config_dir / "business_domains.yaml"
        if domains_file.exists():
            self._load_business_domains(domains_file)

        # Load architectural patterns
        patterns_file = self.config_dir / "architectural_patterns.yaml"
        if patterns_file.exists():
            self._load_architectural_patterns(patterns_file)

        # Load team information
        teams_file = self.config_dir / "teams.yaml"
        if teams_file.exists():
            self._load_teams(teams_file)

        # Load annotations
        annotations_file = self.config_dir / "annotations.yaml"
        if annotations_file.exists():
            self._load_annotations(annotations_file)

    def _load_business_domains(self, file_path: Path):
        """Load business domain configurations."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        for domain_data in data.get("domains", []):
            domain = BusinessDomain(
                name=domain_data["name"],
                keywords=domain_data.get("keywords", []),
                patterns=domain_data.get("patterns", []),
                importance=domain_data.get("importance", 1.0),
            )
            self.business_domains.append(domain)

    def _load_architectural_patterns(self, file_path: Path):
        """Load architectural pattern configurations."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        for pattern_data in data.get("patterns", []):
            pattern = ArchitecturalPattern(
                name=pattern_data["name"],
                indicators=pattern_data.get("indicators", []),
                quality_metrics=pattern_data.get("quality_metrics", {}),
                relationships=pattern_data.get("relationships", []),
            )
            self.architectural_patterns.append(pattern)

    def _load_teams(self, file_path: Path):
        """Load team ownership configurations."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        for team_data in data.get("teams", []):
            team = TeamOwnership(
                name=team_data["name"],
                owned_modules=team_data.get("owned_modules", []),
                expertise=team_data.get("expertise", []),
                contact=team_data.get("contact", ""),
                code_style=team_data.get("code_style", {}),
            )
            self.teams.append(team)

    def _load_annotations(self, file_path: Path):
        """Load contextual annotations."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        for category, category_data in data.items():
            for file_func, annotations in category_data.items():
                parts = file_func.split(":")
                if len(parts) >= 2:
                    file_path = parts[0]
                    function_name = parts[1]

                    annotation = ContextualAnnotation(
                        file_path=file_path,
                        function_name=function_name,
                        category=category,
                        annotations=annotations,
                    )

                    key = f"{file_path}:{function_name}"
                    if key not in self.annotations:
                        self.annotations[key] = []
                    self.annotations[key].append(annotation)

    def _load_default_configurations(self):
        """
        Load default contextual configurations when config files are not available.

        This provides sensible defaults for common software development patterns,
        business domains, and architectural patterns.
        """

        # Default business domains
        self.business_domains = [
            BusinessDomain(
                name="Authentication",
                keywords=[
                    "auth",
                    "login",
                    "password",
                    "token",
                    "jwt",
                    "oauth",
                    "user",
                    "session",
                    "security",
                ],
                patterns=[r".*auth.*", r".*login.*", r".*user.*", r".*session.*"],
                importance=1.8,
            ),
            BusinessDomain(
                name="Database",
                keywords=[
                    "sql",
                    "query",
                    "database",
                    "db",
                    "table",
                    "migration",
                    "orm",
                    "model",
                    "repository",
                ],
                patterns=[r".*db.*", r".*sql.*", r".*model.*", r".*repo.*"],
                importance=1.4,
            ),
            BusinessDomain(
                name="API",
                keywords=[
                    "api",
                    "endpoint",
                    "rest",
                    "http",
                    "request",
                    "response",
                    "route",
                    "handler",
                ],
                patterns=[r".*api.*", r".*endpoint.*", r".*route.*", r".*handler.*"],
                importance=1.5,
            ),
            BusinessDomain(
                name="Payment",
                keywords=[
                    "payment",
                    "billing",
                    "invoice",
                    "subscription",
                    "charge",
                    "refund",
                    "transaction",
                ],
                patterns=[r".*payment.*", r".*billing.*", r".*invoice.*", r".*transaction.*"],
                importance=2.0,
            ),
            BusinessDomain(
                name="Core",
                keywords=["core", "util", "common", "shared", "base", "foundation"],
                patterns=[r".*core.*", r".*util.*", r".*common.*", r".*shared.*"],
                importance=1.6,
            ),
            BusinessDomain(
                name="UI",
                keywords=["ui", "view", "template", "render", "display", "frontend", "component"],
                patterns=[r".*ui.*", r".*view.*", r".*template.*", r".*component.*"],
                importance=1.2,
            ),
            BusinessDomain(
                name="Configuration",
                keywords=["config", "settings", "env", "environment", "setup", "init"],
                patterns=[r".*config.*", r".*settings.*", r".*env.*", r".*setup.*"],
                importance=1.3,
            ),
            BusinessDomain(
                name="Testing",
                keywords=["test", "mock", "fixture", "assert", "spec", "unit", "integration"],
                patterns=[r".*test.*", r".*mock.*", r".*spec.*"],
                importance=1.1,
            ),
        ]

        # Default architectural patterns
        self.architectural_patterns = [
            ArchitecturalPattern(
                name="MVC",
                indicators=["controller", "view", "model"],
                quality_metrics={
                    "max_complexity": 10,
                    "min_test_coverage": 0.8,
                    "single_responsibility": True,
                },
                relationships=["controller_uses_model", "controller_renders_view"],
            ),
            ArchitecturalPattern(
                name="Repository",
                indicators=["repository", "repo"],
                quality_metrics={
                    "interface_required": True,
                    "max_complexity": 8,
                    "single_responsibility": True,
                },
                relationships=["service_uses_repository"],
            ),
            ArchitecturalPattern(
                name="Service",
                indicators=["service", "manager"],
                quality_metrics={"max_complexity": 12, "interface_required": False},
                relationships=["service_uses_repository", "controller_uses_service"],
            ),
            ArchitecturalPattern(
                name="Factory",
                indicators=["factory", "builder", "create"],
                quality_metrics={"single_responsibility": True, "max_complexity": 6},
                relationships=["creates_objects"],
            ),
            ArchitecturalPattern(
                name="Handler",
                indicators=["handler", "processor", "executor"],
                quality_metrics={"max_complexity": 15, "error_handling_required": True},
                relationships=["handles_requests"],
            ),
            ArchitecturalPattern(
                name="Validator",
                indicators=["validator", "validate", "check"],
                quality_metrics={"max_complexity": 8, "comprehensive_coverage": True},
                relationships=["validates_input"],
            ),
        ]

        # Default teams (generic development team structure)
        self.teams = [
            TeamOwnership(
                name="Backend Team",
                owned_modules=["api/", "core/", "services/", "models/"],
                expertise=["api_development", "database_design", "performance", "scalability"],
                contact="backend-team@company.com",
                code_style={
                    "max_line_length": 100,
                    "mandatory_tests": True,
                    "prefer_composition": True,
                    "error_handling_required": True,
                },
            ),
            TeamOwnership(
                name="Frontend Team",
                owned_modules=["ui/", "views/", "templates/", "static/"],
                expertise=["user_interface", "user_experience", "frontend_frameworks"],
                contact="frontend-team@company.com",
                code_style={
                    "max_line_length": 120,
                    "component_based": True,
                    "accessibility_required": True,
                },
            ),
            TeamOwnership(
                name="Security Team",
                owned_modules=["auth/", "security/", "crypto/"],
                expertise=["security", "authentication", "authorization", "cryptography"],
                contact="security-team@company.com",
                code_style={
                    "mandatory_tests": True,
                    "security_first": True,
                    "audit_logging": True,
                    "input_validation": True,
                },
            ),
            TeamOwnership(
                name="Data Team",
                owned_modules=["data/", "analytics/", "ml/", "etl/"],
                expertise=["data_analysis", "machine_learning", "data_engineering"],
                contact="data-team@company.com",
                code_style={
                    "documentation_required": True,
                    "reproducible_results": True,
                    "data_validation": True,
                },
            ),
            TeamOwnership(
                name="Infrastructure Team",
                owned_modules=["deploy/", "config/", "scripts/", "ops/"],
                expertise=["devops", "infrastructure", "deployment", "monitoring"],
                contact="infra-team@company.com",
                code_style={
                    "idempotent_scripts": True,
                    "monitoring_required": True,
                    "rollback_capability": True,
                },
            ),
        ]

        # Default annotations (empty - these are typically project-specific)
        self.annotations = {}

        print(f"✅ Loaded default configurations:")
        print(f"   📊 {len(self.business_domains)} business domains")
        print(f"   🏗️  {len(self.architectural_patterns)} architectural patterns")
        print(f"   👥 {len(self.teams)} team structures")

    def create_default_config_files(self, output_dir: str = "./context_config"):
        """
        Create default configuration files that users can customize.

        Args:
            output_dir (str): Directory where config files will be created
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"📁 Creating default configuration files in {output_dir}")

        # Create default business domains config
        domains_config = {
            "domains": [
                {
                    "name": domain.name,
                    "keywords": domain.keywords,
                    "patterns": domain.patterns,
                    "importance": domain.importance,
                }
                for domain in self.business_domains
            ]
        }

        with open(output_path / "business_domains.yaml", "w") as f:
            yaml.safe_dump(domains_config, f, default_flow_style=False, indent=2)

        # Create default architectural patterns config
        patterns_config = {
            "patterns": [
                {
                    "name": pattern.name,
                    "indicators": pattern.indicators,
                    "quality_metrics": pattern.quality_metrics,
                    "relationships": pattern.relationships,
                }
                for pattern in self.architectural_patterns
            ]
        }

        with open(output_path / "architectural_patterns.yaml", "w") as f:
            yaml.safe_dump(patterns_config, f, default_flow_style=False, indent=2)

        # Create default teams config
        teams_config = {
            "teams": [
                {
                    "name": team.name,
                    "owned_modules": team.owned_modules,
                    "expertise": team.expertise,
                    "contact": team.contact,
                    "code_style": team.code_style,
                }
                for team in self.teams
            ]
        }

        with open(output_path / "teams.yaml", "w") as f:
            yaml.safe_dump(teams_config, f, default_flow_style=False, indent=2)

        # Create empty annotations config with examples
        annotations_config = {
            "performance": {
                "example/api.py:process_request": {
                    "critical_path": True,
                    "expected_latency_ms": 200,
                    "rate_limit": "100/minute",
                }
            },
            "security": {
                "example/auth.py:validate_token": {
                    "audit_required": True,
                    "security_level": "critical",
                    "encryption_required": True,
                }
            },
        }

        with open(output_path / "annotations.yaml", "w") as f:
            yaml.safe_dump(annotations_config, f, default_flow_style=False, indent=2)

        print(f"✅ Created configuration files:")
        print(f"   📊 business_domains.yaml - {len(self.business_domains)} domains")
        print(f"   🏗️  architectural_patterns.yaml - {len(self.architectural_patterns)} patterns")
        print(f"   👥 teams.yaml - {len(self.teams)} teams")
        print(f"   📝 annotations.yaml - example annotations")
        print(f"\\n💡 Customize these files for your specific project needs!")

    def get_business_domain(self, entity_name: str, file_path: str) -> Optional[BusinessDomain]:
        """Identify the business domain for a code entity."""
        entity_text = f"{entity_name} {file_path}".lower()

        # Find the best matching domain
        best_match = None
        max_matches = 0

        for domain in self.business_domains:
            matches = 0

            # Count keyword matches
            for keyword in domain.keywords:
                if keyword.lower() in entity_text:
                    matches += 1

            # Check pattern matches
            import re

            for pattern in domain.patterns:
                if re.search(pattern, entity_text, re.IGNORECASE):
                    matches += 2  # Pattern matches are weighted higher

            if matches > max_matches:
                max_matches = matches
                best_match = domain

        return best_match if max_matches > 0 else None

    def get_architectural_pattern(
        self, entity_name: str, entity_type: str
    ) -> Optional[ArchitecturalPattern]:
        """Identify the architectural pattern for a code entity."""
        entity_text = entity_name.lower()

        for pattern in self.architectural_patterns:
            for indicator in pattern.indicators:
                if indicator.lower() in entity_text:
                    return pattern

        return None

    def get_team_ownership(self, file_path: str) -> Optional[TeamOwnership]:
        """Identify team ownership for a file path."""
        for team in self.teams:
            for module_path in team.owned_modules:
                if file_path.startswith(module_path):
                    return team
        return None

    def get_annotations(self, file_path: str, function_name: str) -> List[ContextualAnnotation]:
        """Get contextual annotations for a specific function."""
        key = f"{file_path}:{function_name}"
        return self.annotations.get(key, [])
