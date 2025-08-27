from pydantic import BaseModel

class Query(BaseModel):
    prompt: str

class Response(BaseModel):
    response: str