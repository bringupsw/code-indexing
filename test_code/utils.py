"""Configuration and utility functions for the application."""

import os
import json
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration settings."""
    
    database_url: str
    secret_key: str
    debug_mode: bool
    max_session_duration: int
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        return cls(
            database_url=os.getenv('DATABASE_URL', 'sqlite:///app.db'),
            secret_key=os.getenv('SECRET_KEY', 'dev-secret-key'),
            debug_mode=os.getenv('DEBUG', 'False').lower() == 'true',
            max_session_duration=int(os.getenv('SESSION_DURATION', '3600'))
        )


class Logger:
    """Simple logging utility."""
    
    def __init__(self, name: str):
        self.name = name
        
    def info(self, message: str):
        """Log info message."""
        print(f"[INFO] {self.name}: {message}")
        
    def error(self, message: str):
        """Log error message."""
        print(f"[ERROR] {self.name}: {message}")
        
    def debug(self, message: str):
        """Log debug message."""
        print(f"[DEBUG] {self.name}: {message}")


def load_json_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        file_path: Path to JSON configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        return json.load(f)


def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if email format is valid
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def generate_session_token() -> str:
    """Generate a secure session token.
    
    Returns:
        str: Random session token
    """
    import secrets
    return secrets.token_urlsafe(32)


class CacheManager:
    """Simple in-memory cache manager."""
    
    def __init__(self):
        self._cache = {}
        
    def get(self, key: str) -> Any:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        return self._cache.get(key)
        
    def set(self, key: str, value: Any):
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
        
    def delete(self, key: str):
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
        """
        self._cache.pop(key, None)
        
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()
