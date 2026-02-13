#======================================================================================================
# DESCRIPTION : It defines the data structures for authentication-related API requests and responses
#======================================================================================================


from typing import Optional
from pydantic import BaseModel, Field, EmailStr

#======================================================
# defines the exact structure of a user signup request
#======================================================

class UserCreate(BaseModel):
    
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

#=========================================================
# defines the exact structure of a user login request
#=========================================================

class UserLogin(BaseModel):
    
    username: str = Field(..., description="Username")
    password: str = Field(..., description="User password")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "securepassword123"
            }
        }

#=================================================================================
#defines the safe user data that the backend is allowed to send back to the client
#=================================================================================
class UserResponse(BaseModel):
   
    id: int
    username: str
    email: str
    created_at: str

#=======================================================================================
# defines the structure of the response sent back to the client after a successful login
#=======================================================================================

class AuthResponse(BaseModel):
   
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
