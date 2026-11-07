from pydantic import BaseModel, field_validator


class SearchDocsSchema(BaseModel):
    question: str
    top_k: int = 5

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("top_k must be between 1 and 20")
        return v
