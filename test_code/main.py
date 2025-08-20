"""Main application module for user authentication and data processing."""

import hashlib
from typing import List, Dict, Optional


class UserManager:
    """Handles user authentication and management operations."""
    
    def __init__(self):
        self.users = {}
        self.active_sessions = {}
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate a user with username and password.
        
        Args:
            username: The user's username
            password: The user's password
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        if username not in self.users:
            return False
        
        hashed_password = self._hash_password(password)
        return self.users[username]['password'] == hashed_password
    
    def create_user(self, username: str, password: str, email: str) -> bool:
        """Create a new user account.
        
        Args:
            username: Unique username
            password: User password
            email: User email address
            
        Returns:
            bool: True if user created successfully
        """
        if username in self.users:
            return False
        
        self.users[username] = {
            'password': self._hash_password(password),
            'email': email,
            'created_at': '2025-01-01'
        }
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()


class DataProcessor:
    """Processes and analyzes user data."""
    
    def __init__(self):
        self.cache = {}
    
    def process_user_data(self, user_id: str, data: Dict) -> Dict:
        """Process user data and return analytics.
        
        Args:
            user_id: Unique user identifier
            data: Raw user data dictionary
            
        Returns:
            Dict: Processed analytics data
        """
        if user_id in self.cache:
            return self.cache[user_id]
        
        processed = {
            'user_id': user_id,
            'total_actions': len(data.get('actions', [])),
            'last_login': data.get('last_login'),
            'preferences': self._extract_preferences(data)
        }
        
        self.cache[user_id] = processed
        return processed
    
    def _extract_preferences(self, data: Dict) -> Dict:
        """Extract user preferences from raw data."""
        return data.get('preferences', {})


def main():
    """Main application entry point."""
    user_manager = UserManager()
    data_processor = DataProcessor()
    
    # Create sample user
    user_manager.create_user("john_doe", "secure123", "john@example.com")
    
    # Authenticate user
    if user_manager.authenticate_user("john_doe", "secure123"):
        print("Authentication successful")
        
        # Process some data
        sample_data = {
            'actions': ['login', 'view_profile', 'update_settings'],
            'last_login': '2025-01-15',
            'preferences': {'theme': 'dark', 'notifications': True}
        }
        
        result = data_processor.process_user_data("john_doe", sample_data)
        print(f"Processed data: {result}")
    else:
        print("Authentication failed")


if __name__ == "__main__":
    main()
