# models.py

from pydantic import BaseModel, Field
from typing import Optional

class Task(BaseModel):
    """
    Supervisor가 생성하고 관리하는 작업 단위를 정의하는 모델입니다.
    각각의 Task는 어떤 에이전트가 무슨 일을 해야 하는지를 나타냅니다.
    """
    agent: str = Field(description="이 작업을 수행할 에이전트의 이름 (예: 'communicator')")
    description: str = Field(description="수행할 작업에 대한 명확하고 간단한 설명")
    done: bool = Field(default=False, description="작업의 완료 여부")
    done_at: Optional[str] = Field(default="", description="작업이 완료된 시간 (YYYY-MM-DD HH:MM:SS 형식)")