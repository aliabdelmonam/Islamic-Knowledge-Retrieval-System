from pydantic import BaseModel, Field

from app.agents.retrieval_agent import RetrievedDocument
from app.agents.triage_agent import ChitchatType, IslamicCategory


class AskResponse(BaseModel):
    answer: str
    sources: list[RetrievedDocument] = Field(default_factory=list)
    categories: list[IslamicCategory] = Field(default_factory=list)
    chitchat_type: ChitchatType = ChitchatType.NONE
    needs_clarification: bool = False
    resolved_query: str = ""
    tool_calls_made: list[IslamicCategory] = Field(default_factory=list)
    is_fallback: bool = False
    session_id: str


class RetrieveResponse(BaseModel):
    query: str
    categories: list[IslamicCategory] = Field(default_factory=list)
    needs_clarification: bool = False
    results: list[RetrievedDocument] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedDocument] = Field(default_factory=list)
    categories: list[IslamicCategory] = Field(default_factory=list)
    chitchat_type: ChitchatType = ChitchatType.NONE
    needs_clarification: bool = False
    resolved_query: str = ""
    tool_calls_made: list[IslamicCategory] = Field(default_factory=list)
    is_fallback: bool = False
    session_id: str


class HealthResponse(BaseModel):
    status: str
    components: dict[str, bool] = Field(default_factory=dict)
    version: str