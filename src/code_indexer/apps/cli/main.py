"""
Command Line Interface for the Code Indexer

This module provides a comprehensive CLI for the semantic code indexing pipeline.
It supports multiple operation modes and provides extensive configuration options.
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ...libs.common import CodebaseSemanticPipeline, ContextualKnowledgeLoader


def parse_arguments():
    """
    Parse and validate command line arguments for the semantic pipeline.

    This function sets up a comprehensive argument parser that supports
    multiple operation modes and configuration options:

    Operation Modes:
    - --codebase: Index a new codebase
    - --search: Search an existing index
    - --report: Generate report from existing index

    Configuration Options:
    - Processing: workers, model, file patterns
    - Search: search type, result limits
    - Output: format, verbosity, file saving

    Returns:
        argparse.Namespace: Parsed and validated arguments

    Raises:
        SystemExit: If arguments are invalid or conflicting
    """
    parser = argparse.ArgumentParser(
        description="Semantic Code Indexing Pipeline - Extract, embed, and index Python codebases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default configuration files for your project
  python -m code_indexer.apps.cli --create-config --context-config ./my_context

  # Basic usage - index a codebase with contextual knowledge and project identifier
  python -m code_indexer.apps.cli --codebase /path/to/codebase --project-name "my-project" --output ./index --context-config ./context_config

  # Index different projects with clear identification
  python -m code_indexer.apps.cli --codebase /path/to/paperless-ngx --project-name "paperless-ngx" --output ./paperless-index
  python -m code_indexer.apps.cli --codebase /path/to/django-app --project-name "my-django-app" --output ./django-index

  # AI Agent natural language queries
  python -m code_indexer.apps.cli --ask "Find all authentication functions" --index ./index
  python -m code_indexer.apps.cli --ask "Show me security-critical code" --index ./index --agent-format structured
  python -m code_indexer.apps.cli --ask "Who owns the payment processing code?" --index ./index
  python -m code_indexer.apps.cli --ask "What are the most complex functions?" --index ./index
  python -m code_indexer.apps.cli --ask "Find functions similar to user_login" --index ./index

  # Traditional search methods
  python -m code_indexer.apps.cli --search "login authentication" --index ./index --search-type contextual --domain Authentication
  python -m code_indexer.apps.cli --search "database" --index ./index --team "Platform Team" --min-importance 1.5

  # Reports and analytics
  python -m code_indexer.apps.cli --report --index ./index --analytics --save-results ./report.json

  # Update existing index (incremental)
  python -m code_indexer.apps.cli --codebase /path/to/codebase --project-name "my-project" --output ./index --incremental
        """,
    )

    # Main operation modes
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        "--codebase", "-c", type=str, help="Path to the codebase to analyze and index"
    )
    operation_group.add_argument(
        "--search", "-s", type=str, help="Search query to run against existing index"
    )
    operation_group.add_argument(
        "--report", "-r", action="store_true", help="Generate report from existing index"
    )
    operation_group.add_argument(
        "--create-config", action="store_true", help="Create default configuration files"
    )
    operation_group.add_argument(
        "--ask", type=str, help="Ask the AI agent a natural language question about the codebase"
    )

    # Project identification (required when indexing)
    parser.add_argument(
        "--project-name",
        "-p",
        type=str,
        help="Project identifier/name (required when using --codebase)",
    )

    # Output/Index paths
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./code_index",
        help="Output directory for index files (default: ./code_index)",
    )
    parser.add_argument(
        "--index", "-i", type=str, help="Path to existing index (for search/report operations)"
    )

    # Processing options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="microsoft/codebert-base",
        help="Model name for embeddings (default: microsoft/codebert-base)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="**/*.py",
        help="File pattern to match (default: **/*.py)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Enable incremental processing (only process changed files)",
    )

    # Search options
    parser.add_argument(
        "--search-type",
        choices=["semantic", "graph", "hybrid", "contextual"],
        default="hybrid",
        help="Type of search to perform (default: hybrid)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of search results to return (default: 10)"
    )

    # Contextual search filters
    parser.add_argument("--domain", type=str, help="Filter by business domain")
    parser.add_argument("--team", type=str, help="Filter by team ownership")
    parser.add_argument("--pattern", type=str, help="Filter by architectural pattern")
    parser.add_argument("--min-importance", type=float, help="Minimum importance score filter")
    parser.add_argument(
        "--security-critical", action="store_true", help="Show only security-critical entities"
    )

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    parser.add_argument(
        "--output-format",
        choices=["json", "table", "summary"],
        default="summary",
        help="Output format for results (default: summary)",
    )
    parser.add_argument("--save-results", type=str, help="Save results to file (JSON format)")

    # Advanced options
    parser.add_argument(
        "--exclude-tests", action="store_true", help="Exclude test files from analysis"
    )
    parser.add_argument(
        "--min-complexity",
        type=int,
        default=0,
        help="Minimum complexity threshold for functions (default: 0)",
    )
    parser.add_argument(
        "--max-file-size", type=int, default=None, help="Maximum file size in KB to process"
    )
    parser.add_argument(
        "--analytics", action="store_true", help="Generate contextual analytics report"
    )
    parser.add_argument(
        "--context-config",
        type=str,
        default="./context_config",
        help="Path to contextual configuration directory (default: ./context_config)",
    )
    parser.add_argument(
        "--agent-format",
        choices=["natural", "structured", "json"],
        default="natural",
        help="Format for AI agent responses (default: natural)",
    )

    args = parser.parse_args()

    # Validation
    if args.codebase and not args.project_name:
        parser.error("--project-name is required when using --codebase")

    if args.search or args.report or args.ask:
        if not args.index:
            args.index = args.output
        if not Path(args.index).exists():
            parser.error(f"Index directory '{args.index}' does not exist")

    if args.codebase and not Path(args.codebase).exists():
        parser.error(f"Codebase path '{args.codebase}' does not exist")

    if args.quiet and args.verbose:
        parser.error("Cannot use --quiet and --verbose together")

    return args


def setup_logging(verbose: bool, quiet: bool):
    """
    Configure logging based on verbosity preferences.

    Sets up Python logging with appropriate levels and formats:
    - quiet: Only error messages
    - verbose: Debug level with timestamps
    - normal: Info level with simple format

    Args:
        verbose (bool): Enable verbose debug output
        quiet (bool): Suppress non-error output

    Returns:
        logging.Logger: Configured logger instance
    """
    if quiet:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    elif verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    return logging.getLogger(__name__)


def print_progress(message: str, quiet: bool = False):
    """
    Print timestamped progress message unless in quiet mode.

    Args:
        message (str): Progress message to display
        quiet (bool): Whether to suppress output
    """
    if not quiet:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")


def format_results(results: List[Dict], format_type: str) -> str:
    """
    Format search results in the specified output format.

    Supports multiple output formats for different use cases:
    - json: Machine-readable JSON format
    - table: Human-readable tabular format
    - summary: Detailed summary with descriptions

    Args:
        results (List[Dict]): Search results to format
        format_type (str): Output format ('json', 'table', 'summary')

    Returns:
        str: Formatted results string

    Raises:
        ValueError: If format_type is not supported
    """
    if format_type == "json":
        return json.dumps(results, indent=2)

    elif format_type == "table":
        if not results:
            return "No results found."

        # Simple table formatting with contextual information
        headers = ["Name", "Type", "Score", "Domain", "Team", "File"]
        rows = []
        for result in results:
            score = result.get("contextual_score", result.get("similarity", 0))
            rows.append(
                [
                    result.get("name", "N/A"),
                    result.get("type", "N/A"),
                    f"{score:.3f}",
                    result.get("business_domain", "N/A")[:15],
                    result.get("team_ownership", "N/A")[:15],
                    result.get("file_path", "N/A")[-30:],
                ]
            )

        # Calculate column widths
        widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

        # Format table
        table_lines = []
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        table_lines.append(header_line)
        table_lines.append("-" * len(header_line))

        for row in rows:
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            table_lines.append(row_line)

        return "\n".join(table_lines)

    else:  # summary format
        if not results:
            return "No results found."

        summary_lines = [f"Found {len(results)} results:\n"]
        for i, result in enumerate(results, 1):
            summary_lines.append(f"{i}. {result.get('name', 'Unknown')}")
            summary_lines.append(f"   Type: {result.get('type', 'N/A')}")

            # Show appropriate score
            if "contextual_score" in result:
                summary_lines.append(f"   Contextual Score: {result['contextual_score']:.3f}")
                summary_lines.append(f"   Importance: {result.get('importance_score', 1.0):.2f}")
            elif "similarity" in result:
                summary_lines.append(f"   Similarity: {result['similarity']:.3f}")

            # Show contextual information
            if "business_domain" in result and result["business_domain"]:
                summary_lines.append(f"   Domain: {result['business_domain']}")
            if "team_ownership" in result and result["team_ownership"]:
                summary_lines.append(f"   Team: {result['team_ownership']}")
            if "annotations" in result and result["annotations"]:
                summary_lines.append(f"   Annotations: {len(result['annotations'])} categories")

            if "connections" in result:
                summary_lines.append(f"   Connections: {result['connections']}")

            summary_lines.append("")

        return "\n".join(summary_lines)


def main():
    """
    Main entry point for the semantic code indexing pipeline.

    This function orchestrates the complete workflow based on command-line
    arguments. It handles multiple operation modes:

    1. Codebase Processing: Analyzes and indexes a codebase
    2. Search Operations: Queries an existing index
    3. Report Generation: Creates analytics from an index
    4. AI Agent Queries: Natural language querying

    The function includes comprehensive error handling, progress reporting,
    and performance timing. It ensures graceful handling of interrupts
    and provides detailed error messages for debugging.

    Exit Codes:
    - 0: Success
    - 1: General error
    - 130: Keyboard interrupt (Ctrl+C)

    Raises:
        SystemExit: On various error conditions or user interrupt
    """
    args = parse_arguments()
    logger = setup_logging(args.verbose, args.quiet)

    start_time = time.time()

    try:
        if args.codebase:
            # Index/process codebase
            print_progress("🚀 Starting semantic code indexing pipeline", args.quiet)
            print_progress(f"📁 Codebase: {args.codebase}", args.quiet)
            print_progress(f"📊 Output: {args.output}", args.quiet)
            print_progress(f"🔧 Context config: {args.context_config}", args.quiet)

            # Initialize pipeline
            pipeline = CodebaseSemanticPipeline(args.output, args.context_config)

            if args.workers:
                print_progress(f"⚡ Using {args.workers} worker processes", args.quiet)

            # Process the codebase with project identifier
            results = pipeline.process_codebase(args.codebase, project_name=args.project_name)

            elapsed = time.time() - start_time
            print_progress(f"✅ Processing completed in {elapsed:.2f} seconds", args.quiet)
            print_progress(
                f"📈 Results: {results['entities']} entities, {results['relations']} relations",
                args.quiet,
            )

            if args.save_results:
                with open(args.save_results, "w") as f:
                    json.dump(results, f, indent=2)
                print_progress(f"💾 Results saved to {args.save_results}", args.quiet)

        elif args.create_config:
            # Create default configuration files
            print_progress("📁 Creating default configuration files", args.quiet)

            # Use the specified context config directory or default
            config_dir = (
                args.context_config if hasattr(args, "context_config") else "./context_config"
            )

            # Create a temporary loader to access the default configs
            temp_loader = ContextualKnowledgeLoader(
                "/nonexistent"
            )  # This will trigger default loading
            temp_loader.create_default_config_files(config_dir)

            print_progress("✅ Configuration files created successfully!", args.quiet)
            print_progress(
                f"💡 Edit the files in '{config_dir}' to customize for your project", args.quiet
            )

        elif args.ask:
            # AI Agent Query
            print_progress(f"🤖 Asking AI agent: '{args.ask}'", args.quiet)

            # Initialize pipeline with existing index
            index_path = args.index if args.index else args.output
            if not Path(index_path).exists():
                print(
                    "❌ Index not found. Please run indexing first or specify correct --index path",
                    file=sys.stderr,
                )
                sys.exit(1)

            pipeline = CodebaseSemanticPipeline(index_path, args.context_config)

            # Query the AI agent
            response = pipeline.query_agent(args.ask)

            if not args.quiet:
                print(f"\n🤖 AI Agent Response:")
                print("=" * 50)

            # Format response based on format type
            if args.agent_format == "json":
                print(json.dumps(response, indent=2))
            elif args.agent_format == "structured":
                print(f"Query: {args.ask}")
                print(f"Found {len(response['entities'])} relevant entities")
                print(f"Confidence: {response['confidence']:.2f}")
                print(f"Query time: {response['query_time']:.3f}s")
                if response["entities"]:
                    print("\nEntities:")
                    for entity in response["entities"]:
                        print(f"  - {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
                        if entity.get("business_domain"):
                            print(f"    Domain: {entity['business_domain']}")
                        if entity.get("team_ownership"):
                            print(f"    Team: {entity['team_ownership']}")
            else:  # natural format
                if response["entities"]:
                    print(f"Found {len(response['entities'])} relevant entities:")
                    for entity in response["entities"]:
                        print(f"• {entity.get('name', 'N/A')} - {entity.get('type', 'N/A')}")
                        if entity.get("business_domain"):
                            print(f"  Domain: {entity['business_domain']}")
                        if entity.get("file_path"):
                            print(f"  File: {entity['file_path']}")
                else:
                    print("No relevant entities found for your query.")

            if args.save_results:
                with open(args.save_results, "w") as f:
                    json.dump(response, f, indent=2, default=str)
                print_progress(f"💾 Agent result saved to {args.save_results}", args.quiet)

        elif args.search:
            # Search existing index
            print_progress(f"🔍 Searching index: {args.index}", args.quiet)
            print_progress(f"🎯 Query: '{args.search}'", args.quiet)
            print_progress(f"📋 Search type: {args.search_type}", args.quiet)

            # Initialize pipeline with existing index
            pipeline = CodebaseSemanticPipeline(args.index, args.context_config)

            # Prepare context filters for contextual search
            context_filters = {}
            if args.domain:
                context_filters["business_domain"] = args.domain
            if args.team:
                context_filters["team_ownership"] = args.team
            if args.pattern:
                context_filters["architectural_pattern"] = args.pattern
            if args.min_importance:
                context_filters["min_importance"] = args.min_importance
            if args.security_critical:
                context_filters["has_security_annotations"] = True

            # Perform search
            if args.search_type == "contextual":
                search_results = pipeline.contextual_search(args.search, context_filters)
            else:
                search_results = pipeline.search(args.search, search_type=args.search_type)

            # Limit results
            search_results = search_results[: args.top_k]

            # Format and display results
            formatted_results = format_results(search_results, args.output_format)

            if not args.quiet:
                print(f"\n🔍 Search Results ({len(search_results)} found):")
                print("=" * 50)
            print(formatted_results)

            if args.save_results:
                with open(args.save_results, "w") as f:
                    json.dump(search_results, f, indent=2)
                print_progress(f"💾 Search results saved to {args.save_results}", args.quiet)

        elif args.report:
            # Generate report from existing index
            print_progress(f"📊 Generating report from index: {args.index}", args.quiet)

            # Initialize pipeline
            pipeline = CodebaseSemanticPipeline(args.index, args.context_config)

            # Load summary file
            summary_path = Path(args.index) / "summary.json"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    summary = json.load(f)

                if not args.quiet:
                    print("\n📈 Codebase Analysis Report")
                    print("=" * 50)

                print(f"Total Entities: {summary['total_entities']:,}")
                print(f"Total Relations: {summary['total_relations']:,}")
                print(f"\nEntity Distribution:")
                for entity_type, count in summary["entity_types"].items():
                    print(f"  {entity_type}: {count:,}")

                print(f"\nComplexity Distribution:")
                for level, count in summary["complexity_distribution"].items():
                    print(f"  {level}: {count:,}")

                # Show contextual information
                if "business_domains" in summary and summary["business_domains"]:
                    print(f"\nBusiness Domains:")
                    for domain, data in summary["business_domains"].items():
                        print(
                            f"  {domain}: {data['count']} entities (avg complexity: {data['avg_complexity']})"
                        )

                if "team_ownership" in summary and summary["team_ownership"]:
                    print(f"\nTeam Ownership:")
                    for team, data in summary["team_ownership"].items():
                        print(
                            f"  {team}: {data['count']} entities (avg complexity: {data['avg_complexity']})"
                        )

                if "architectural_patterns" in summary and summary["architectural_patterns"]:
                    print(f"\nArchitectural Patterns:")
                    for pattern, count in summary["architectural_patterns"].items():
                        print(f"  {pattern}: {count} entities")

                if summary["top_complex_functions"]:
                    print(f"\nMost Complex Functions:")
                    for func in summary["top_complex_functions"][:5]:
                        domain_info = (
                            f" ({func.get('business_domain', 'N/A')})"
                            if func.get("business_domain")
                            else ""
                        )
                        print(f"  {func['name']}: {func['complexity']}{domain_info}")

                # Generate contextual analytics if requested
                if args.analytics:
                    print_progress(f"🧠 Generating contextual analytics...", args.quiet)
                    analytics = pipeline.get_contextual_analytics()

                    print(f"\n🧠 Contextual Analytics:")
                    print("=" * 30)
                    print(f"Generated at: {analytics['generated_at']}")

                if args.save_results:
                    report_data = summary
                    if args.analytics:
                        analytics = pipeline.get_contextual_analytics()
                        report_data["contextual_analytics"] = analytics

                    with open(args.save_results, "w") as f:
                        json.dump(report_data, f, indent=2)
                    print_progress(f"💾 Report saved to {args.save_results}", args.quiet)
            else:
                print("❌ No summary report found in index directory", file=sys.stderr)
                sys.exit(1)

    except KeyboardInterrupt:
        print_progress("⏹️ Process interrupted by user", args.quiet)
        sys.exit(130)

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    finally:
        elapsed = time.time() - start_time
        print_progress(f"⏱️ Total runtime: {elapsed:.2f} seconds", args.quiet)


if __name__ == "__main__":
    main()
