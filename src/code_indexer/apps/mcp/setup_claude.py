#!/usr/bin/env python3
"""
Setup script for configuring Claude Desktop with the Semantic Code Indexer MCP Server
"""

import json
import os
import shutil
from pathlib import Path


def get_claude_config_path():
    """Get the Claude Desktop configuration path based on OS."""
    home = Path.home()

    # macOS
    if os.name == "posix" and "darwin" in os.uname().sysname.lower():
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"

    # Windows
    elif os.name == "nt":
        return home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"

    # Linux
    else:
        return home / ".config" / "Claude" / "claude_desktop_config.json"


def setup_claude_desktop():
    """Set up Claude Desktop configuration for the MCP server."""
    print("🔧 Setting up Claude Desktop MCP Server Configuration")
    print("=" * 60)

    # Get paths
    config_path = get_claude_config_path()

    # Use the new CLI command instead of direct script path
    print(f"📁 Claude config path: {config_path}")
    print(f"🚀 MCP command: code-indexer-mcp")

    # Create configuration using the CLI command
    mcp_config = {
        "mcpServers": {"semantic-code-indexer": {"command": "code-indexer-mcp", "args": []}}
    }

    # Check if config file exists
    if config_path.exists():
        print(f"\n📄 Existing Claude config found")
        try:
            with open(config_path, "r") as f:
                existing_config = json.load(f)

            # Merge configurations
            if "mcpServers" not in existing_config:
                existing_config["mcpServers"] = {}

            existing_config["mcpServers"]["semantic-code-indexer"] = mcp_config["mcpServers"][
                "semantic-code-indexer"
            ]
            mcp_config = existing_config

            print("✅ Merged with existing configuration")

        except json.JSONDecodeError:
            print("⚠️  Existing config is invalid JSON, creating backup...")
            backup_path = config_path.with_suffix(".json.backup")
            shutil.copy2(config_path, backup_path)
            print(f"📄 Backup created: {backup_path}")

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write configuration
    with open(config_path, "w") as f:
        json.dump(mcp_config, f, indent=2)

    print(f"\n✅ Configuration written to: {config_path}")

    # Check if the package is installed
    print(f"\n🔍 Checking installation...")
    try:
        import subprocess

        result = subprocess.run(
            ["code-indexer-mcp", "--test"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("✅ MCP server is properly installed and working")
        else:
            print("❌ MCP server test failed")
            print(f"   Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ code-indexer-mcp command not found")
        print("   Make sure to install the package with: pip install -e .")
        return False
    except subprocess.TimeoutExpired:
        print("❌ MCP server test timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False

    print(f"\n🎉 Setup complete!")
    print(f"\n📋 Next steps:")
    print(f"   1. Restart Claude Desktop")
    print(f"   2. Open a new conversation")
    print(f"   3. Try asking: 'Can you analyze my Python project at /path/to/project?'")

    print(f"\n💡 Example prompts to try:")
    print(f"   • 'Index my codebase at ~/my-project'")
    print(f"   • 'Find all authentication functions'")
    print(f"   • 'What are the most complex functions?'")
    print(f"   • 'Generate a code quality report'")
    print(f"   • 'Find functions similar to login_user'")

    return True


def verify_setup():
    """Verify the MCP server setup."""
    print("\n🔍 Verifying setup...")

    config_path = get_claude_config_path()

    if not config_path.exists():
        print("❌ Claude config file not found")
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "mcpServers" not in config:
            print("❌ No MCP servers configured")
            return False

        if "semantic-code-indexer" not in config["mcpServers"]:
            print("❌ Semantic code indexer not configured")
            return False

        server_config = config["mcpServers"]["semantic-code-indexer"]

        # Test if the command works
        try:
            import subprocess

            result = subprocess.run(
                [server_config["command"], "--test"], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print("✅ Configuration is valid")
                print("✅ MCP server command works")
            else:
                print("❌ MCP server command failed")
                print(f"   Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error testing command: {e}")
            return False

        return True

    except json.JSONDecodeError:
        print("❌ Claude config file is invalid JSON")
        return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup Claude Desktop MCP Server")
    parser.add_argument("--verify", action="store_true", help="Verify existing setup")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration")

    args = parser.parse_args()

    if args.verify:
        verify_setup()
    elif args.show_config:
        config_path = get_claude_config_path()
        if config_path.exists():
            with open(config_path, "r") as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("No Claude config file found")
    else:
        setup_claude_desktop()
