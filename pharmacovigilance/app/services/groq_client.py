#=======================================================================
# DESCRIPTION : It creates and returns an authenticated Groq API client
#=======================================================================

import os
from groq import Groq

#=========================================================================================
# creates and returns a Groq API client using the GROQ_API_KEY from environment variables
#=========================================================================================

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    return Groq(api_key=api_key)
