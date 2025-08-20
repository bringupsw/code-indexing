"""Database utilities for user and session management."""

import sqlite3
from typing import List, Dict, Optional
from datetime import datetime


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            
    def create_tables(self):
        """Create necessary database tables."""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        self.connection.commit()


class UserRepository:
    """Repository for user-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def create_user(self, username: str, email: str, password_hash: str) -> int:
        """Create a new user in the database.
        
        Args:
            username: Unique username
            email: User email
            password_hash: Hashed password
            
        Returns:
            int: User ID of created user
        """
        cursor = self.db_manager.connection.cursor()
        cursor.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        self.db_manager.connection.commit()
        return cursor.lastrowid
        
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Retrieve user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            Optional[Dict]: User data if found, None otherwise
        """
        cursor = self.db_manager.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
        
    def get_all_users(self) -> List[Dict]:
        """Retrieve all users from database.
        
        Returns:
            List[Dict]: List of all user records
        """
        cursor = self.db_manager.connection.cursor()
        cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]


class SessionManager:
    """Manages user sessions in the database."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def create_session(self, user_id: int, session_token: str, expires_at: str) -> int:
        """Create a new session.
        
        Args:
            user_id: ID of the user
            session_token: Unique session token
            expires_at: Session expiration timestamp
            
        Returns:
            int: Session ID
        """
        cursor = self.db_manager.connection.cursor()
        cursor.execute(
            'INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?, ?, ?)',
            (user_id, session_token, expires_at)
        )
        self.db_manager.connection.commit()
        return cursor.lastrowid
        
    def get_active_sessions(self, user_id: int) -> List[Dict]:
        """Get all active sessions for a user.
        
        Args:
            user_id: User ID to check
            
        Returns:
            List[Dict]: List of active sessions
        """
        cursor = self.db_manager.connection.cursor()
        cursor.execute(
            'SELECT * FROM sessions WHERE user_id = ? AND expires_at > ?',
            (user_id, datetime.now().isoformat())
        )
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
