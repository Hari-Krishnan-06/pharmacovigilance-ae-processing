#==================================================================================================
# DESCRIPTION : It implements a basic authentication and identity management layer for the application
#==================================================================================================

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import jwt

# ============ Configuration ============

SECRET_KEY = "pharmacovigilance-secret-key-change-in-production"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

# Database path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "audit.db"

# Initialize Argon2 password hasher
ph = PasswordHasher()


# ============ Database Initialization ============

#=========================================================
# creates the authentication storage layer using SQLite
#=========================================================

def init_users_table():
    """Initialize the users table in the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"
        )
        
        conn.commit()


# ============ Password Hashing ============

#========================================================================================
# takes a plain-text password and converts it into a secure,
# irreversible cryptographic hash using Argon2id
#========================================================================================

def hash_password(password: str) -> str:
    """
    Hash a password using Argon2id.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    return ph.hash(password)

#================================================================================================
# It checks whether the password a user typed matches the hashed password stored in the database
#================================================================================================

def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash using Argon2.
    
    Args:
        password: Plain text password to verify
        password_hash: Stored password hash
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        ph.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False


# ============ JWT Token Management ============

#=====================================================================
# It creates a signed login token that proves a user is authenticated
#=====================================================================

def create_token(user_id: int, username: str) -> str:
    """
    Create a JWT token for authenticated user.
    
    Args:
        user_id: User's database ID
        username: User's username
        
    Returns:
        JWT token string
    """
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

#================================================================
# checks whether a JWT token is valid and not expired,
# and if valid, extracts the user identity from it.
#================================================================

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ============ User Management ============

#==================================================================================
# It queries the database for a user record whose username matches the given value
#==================================================================================

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a user by username.
    
    Args:
        username: Username to look up
        
    Returns:
        User dict if found, None otherwise
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, email, password_hash, created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None

#===========================================================================
# queries the database to find a user whose email matches the given value
#===========================================================================

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a user by email.
    
    Args:
        email: Email to look up
        
    Returns:
        User dict if found, None otherwise
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, email, password_hash, created_at FROM users WHERE email = ?",
            (email,)
        )
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None

#===========================================================================
# It retrieves a user record from the database using the user’s internal ID
#===========================================================================

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a user by ID.
    
    Args:
        user_id: User ID to look up
        
    Returns:
        User dict if found, None otherwise
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None

#===============================================================================
# It creates a new user with a securely hashed password and stores it in the database
#===============================================================================

def create_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """
    Create a new user with Argon2 hashed password.
    
    Args:
        username: Unique username
        email: User email
        password: Plain text password (will be hashed)
        
    Returns:
        Dict with success status and user data or error message
    """
    # Check if username already exists
    if get_user_by_username(username):
        return {
            "success": False,
            "error": "Username already exists"
        }
    
    # Check if email already exists
    if get_user_by_email(email):
        return {
            "success": False,
            "error": "Email already registered"
        }
    
    # Hash password with Argon2
    password_hash = hash_password(password)
    created_at = datetime.utcnow().isoformat()
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, email, password_hash, created_at)
            )
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                "success": True,
                "user": {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "created_at": created_at
                }
            }
            
    except sqlite3.IntegrityError as e:
        return {
            "success": False,
            "error": f"Database error: {str(e)}"
        }

#===========================================================
# It implement authentication workflow for the application
#===========================================================

def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Plain text password
        
    Returns:
        Dict with success status, user data, and token if authenticated
    """
    user = get_user_by_username(username)
    
    if not user:
        return {
            "success": False,
            "error": "Invalid username or password"
        }
    
    if not verify_password(password, user["password_hash"]):
        return {
            "success": False,
            "error": "Invalid username or password"
        }
    
    # Create JWT token
    token = create_token(user["id"], user["username"])
    
    return {
        "success": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "created_at": user["created_at"]
        },
        "token": token
    }


# Initialize users table on module import
init_users_table()
