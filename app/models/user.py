"""
User Authentication Schemas
Pydantic models for user signup, login, and responses.
"""
from typing import Optional
from pydantic import BaseModel, Field, EmailStr


class UserCreate(BaseModel):
    """Request schema for user signup."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password (min 6 characters)")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "password": "securepassword123"
            }
        }


class UserLogin(BaseModel):
    """Request schema for user login."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="User password")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "securepassword123"
            }
        }


class UserResponse(BaseModel):
    """Response schema for user data (excludes password)."""
    id: int
    username: str
    email: str
    created_at: str


class AuthResponse(BaseModel):
    """Response schema for successful authentication."""
    success: bool
    message: str
    user: Optional[UserResponse] = None
    token: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Login successful",
                "user": {
                    "id": 1,
                    "username": "johndoe",
                    "email": "john@example.com",
                    "created_at": "2024-01-21T12:00:00"
                },
                "token": "eyJhbGciOiJIUzI1NiIs..."
            }
        }
