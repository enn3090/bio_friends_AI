"""
[OpenCode 변환본]

목표: LangChain + OpenAI(Closed LLM, ChatOpenAI) 콘솔 앱을
→ **로컬 오픈소스 LLM (Ollama의 DeepSeek-R1)** + **Gradio UI**로 변환

핵심 변경점
1) API Key, OpenAI 의존 제거 (로컬 Ollama 서버 사용: http://localhost:11434)
2) 모델: deepseek-r1 (예: "deepseek-r1:latest" 또는 "deepseek-r1:7b")
3) 콘솔 while-loop → **Gradio ChatInterface** 채팅 UI
4) LangChain 호환: ChatOllama 사용, System/Human/AI 메시지 유지
5) DeepSeek-R1이 생성하는 <think> 내부 사고 텍스트는 사용자에게 숨김

사전 준비
- Ollama 설치: https://ollama.com
- 모델 다운로드: 터미널에서 `ollama pull deepseek-r1:latest`
- 파이썬 패키지:
  pip install gradio langchain langchain-community ollama

실행
- `python app.py` 실행 후 브라우저에서 http://127.0.0.1:7860
"""

from typing import List, Tuple
import re

import gradio as gr

# LangChain 메시지 타입 (원 코드와 동일 인터페이스)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangChain용 Ollama 채팅 모델
# 최신 버전에서는 `langchain_ollama` 패키지도 있습니다.
# 여기서는 호환성 좋은 커뮤니티 드라이버를 사용합니다.
from langchain_community.chat_models import ChatOllama

# ===== 설정 =====
MODEL_NAME = "deepseek-r1:latest"  # 필요 시 "deepseek-r1:7b" 등으로 교체
SYSTEM_PROMPT = "너는 사용자를 도와주는 상담사야."
TEMPERATURE = 0.9  # OpenAI 예시 코드의 temperature에 대응

# DeepSeek-R1의 내부 사고를 감싼 <think>...</think> 제거용 패턴
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """DeepSeek-R1이 출력하는 내부 사고 내용을 제거해 사용자에게는 보이지 않도록 함."""
    return THINK_TAG_PATTERN.sub("", text).strip()


# LangChain ChatOllama 인스턴스 생성
# - Ollama 로컬 서버가 실행 중이어야 합니다: `ollama serve`
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    # base_url="http://localhost:11434",  # 기본값 사용 시 주석 가능
)


def history_to_messages(history: List[Tuple[str, str]]) -> List:
    """
    Gradio ChatInterface의 history([[user, assistant], ...])를
    LangChain 메시지 리스트로 변환.
    """
    msgs: List = [SystemMessage(SYSTEM_PROMPT)]
    for u, a in history:
        if u:
            msgs.append(HumanMessage(u))
        if a:
            msgs.append(AIMessage(a))
    return msgs


def respond(user_input: str, history: List[Tuple[str, str]]) -> str:
    """
    ChatInterface 콜백:
    - 기존 히스토리 + 사용자 입력을 LangChain 메시지로 구성해 모델 호출
    - DeepSeek의 <think> 섹션은 제거 후 반환
    """
    messages = history_to_messages(history)
    messages.append(HumanMessage(user_input))

    # LangChain ChatOllama는 .invoke로 동기 호출
    ai_msg = llm.invoke(messages)
    content = getattr(ai_msg, "content", str(ai_msg))
    return strip_think(content)


# ===== Gradio UI 구성 =====
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 💬 Chatbot (DeepSeek-R1 on Ollama + Gradio + LangChain)")

    chat_ui = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            label="상담사",
            height=520,
        ),
        title="",
        description=(
            "로컬 **Ollama**의 **DeepSeek-R1** 모델을 LangChain으로 구동합니다. "
            "좌측 입력창에 질문을 적고 메시지를 보내보세요."
        ),
        theme="soft",
        retry_btn="다시 생성",
        undo_btn="이전 메시지 삭제",
        clear_btn="대화 초기화",
        examples=[
            "안녕하세요! 오늘 기분이 좀 우울해요.",
            "면접이 걱정돼요. 준비 팁이 있을까요?",
            "학습/업무 루틴을 잡고 싶은데 도와줘.",
        ],
    )

    with gr.Accordion("시스템/모델 정보", open=False):
        gr.Markdown(
            f"""
- 사용 모델: **{MODEL_NAME}**  
- 로컬 서버: **http://localhost:11434** (Ollama 기본)  
- 프롬프트 역할: `{SYSTEM_PROMPT}`  
- Temperature: `{TEMPERATURE}`
"""
        )

if __name__ == "__main__":
    # 내부망만 사용할 경우 share=False 유지
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

