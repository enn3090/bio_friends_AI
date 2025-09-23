"""
[OpenCode 변환본]

목표: Streamlit + OpenAI(ChatOpenAI) 기반 RAG 챗봇을
→ **로컬 오픈소스 LLM (Ollama의 DeepSeek-R1)** + **Gradio UI**로 변환

핵심 변경점
1) OpenAI/키 제거, **Ollama 로컬 서버(http://localhost:11434)** 사용
2) 모델: deepseek-r1 (예: "deepseek-r1:latest" 또는 "deepseek-r1:7b/14b")
3) Streamlit → **Gradio ChatInterface** (스트리밍 대응)
4) 기존 `retriever` 모듈의 체인(`query_augmentation_chain`, `retriever`, `document_chain`)은 **그대로** 사용
5) DeepSeek-R1의 내부 사고(<think>...</think>)는 사용자 출력에서 **자동으로 숨김**

사전 준비
- Ollama 설치: https://ollama.com  후 `ollama pull deepseek-r1:latest`
- 패키지: `pip install gradio langchain langchain-community`
- (중요) `retriever.py`가 동일 경로에 존재하고, 아래 속성을 제공한다고 가정:
  - retriever.query_augmentation_chain
  - retriever.retriever
  - retriever.document_chain (stream 메서드 지원)

실행
- `python app.py` 실행 → 브라우저에서 http://127.0.0.1:7860
"""

from __future__ import annotations
from typing import List, Tuple, Generator
import re

import gradio as gr

# LangChain 메시지 타입 (원 코드와 동일 인터페이스)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LangChain용 Ollama 채팅 모델 (로컬 LLM)
from langchain_community.chat_models import ChatOllama

# 기존 RAG 구성요소를 그대로 사용
import retriever


# =========================
# 설정
# =========================
MODEL_NAME = "deepseek-r1:latest"   # 필요 시 "deepseek-r1:7b"/"deepseek-r1:14b"
TEMPERATURE = 0.7
SYSTEM_PROMPT = "너는 문서에 기반해 답변하는 도시 정책 전문가야"

# DeepSeek-R1의 내부 사고(<think>...</think>) 제거용 정규식
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """
    최종 텍스트에서 <think>...</think> 내부 내용을 제거.
    """
    return THINK_TAG_RE.sub("", text).strip()


def stream_hide_think(chunks: Generator[str, None, None]) -> Generator[str, None, None]:
    """
    스트리밍 중에도 <think> 섹션을 화면에 노출하지 않도록 제거.
    간단한 상태머신으로 '<think>' ~ '</think>' 사이 텍스트를 필터링.
    """
    in_think = False
    buf = ""

    for piece in chunks:
        s = str(piece)
        buf += s
        out_parts = []
        i = 0

        while i < len(buf):
            if not in_think:
                start = buf.find("<think>", i)
                if start == -1:
                    out_parts.append(buf[i:])
                    buf = ""
                    break
                else:
                    out_parts.append(buf[i:start])
                    i = start + len("<think>")
                    in_think = True
            else:
                end = buf.find("</think>", i)
                if end == -1:
                    # 종료 태그가 아직 안 왔으면 다음 piece 기다림
                    buf = buf[i:]
                    i = len(buf)
                    break
                else:
                    # think 블록을 건너뛰고 계속 진행
                    i = end + len("</think>")
                    in_think = False

        visible = "".join(out_parts)
        if visible:
            yield visible


# =========================
# 로컬 LLM (Ollama) 인스턴스
# =========================
# 주의: Ollama 서버(기본: http://localhost:11434)가 떠 있어야 합니다.
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    # base_url="http://localhost:11434",  # 기본값이면 생략 가능
)


# =========================
# Gradio <-> LangChain 히스토리 변환
# =========================
def history_to_messages(history: List[Tuple[str, str]]) -> List:
    """
    Gradio ChatInterface의 history([[user, assistant], ...])를
    LangChain 메시지 리스트로 변환. 항상 SYSTEM_PROMPT를 첫 메시지로 포함.
    """
    msgs: List = [SystemMessage(SYSTEM_PROMPT)]
    for user_text, ai_text in history:
        if user_text:
            msgs.append(HumanMessage(user_text))
        if ai_text:
            msgs.append(AIMessage(ai_text))
    return msgs


# =========================
# 메인 응답 로직
# =========================
def respond(user_input: str, history: List[Tuple[str, str]]):
    """
    ChatInterface 콜백:
      1) 기존 히스토리 + 사용자 입력 구성
      2) 질의 확장(query_augmentation_chain)
      3) 관련 문서 검색(retriever.invoke)
      4) 문서 컨텍스트로 document_chain.stream 스트리밍 생성
      5) 스트리밍 동안 <think> 섹션 제거

    반환 형식:
      - yield (partial_text, side_info_markdown)
    """
    # 1) 메시지 구성
    messages = history_to_messages(history) + [HumanMessage(user_input)]

    # 2) 질의 확장
    try:
        augmented_query = retriever.query_augmentation_chain.invoke({
            "messages": messages,
            "query": user_input,
        })
    except Exception as e:
        augmented_query = ""
        # 실패해도 계속 진행

    # 3) 관련 문서 검색
    try:
        docs = retriever.retriever.invoke(f"{user_input}\n{augmented_query}")
    except Exception:
        docs = []

    # 우측 패널에 표시할 간단 정보(문서 수/보강 질의)
    side_info = [
        "### 🔎 RAG 컨텍스트",
        f"- 관련 문서 수: **{len(docs)}**",
        f"- 보강 질의(Augmented Query):",
        f"```\n{augmented_query}\n```" if augmented_query else "_생성 실패_",
    ]
    side_md = "\n".join(side_info)

    # 4) 문서 컨텍스트로 스트리밍 생성
    try:
        raw_stream = retriever.document_chain.stream({
            "messages": messages,
            "context": docs,
        })
    except Exception as e:
        yield f"문서 기반 응답을 생성하지 못했어요.\n사유: {e}", side_md
        return

    # 5) 내부 사고(<think>) 숨김
    safe_stream = stream_hide_think(raw_stream)

    for chunk in safe_stream:
        yield chunk, side_md


# =========================
# Gradio UI 구성
# =========================
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 🏙️ 문서기반 도시정책 챗봇 (DeepSeek-R1 on Ollama + Gradio)")

    chat = gr.ChatInterface(
        fn=respond,
        additional_outputs=[gr.Markdown(label="세부 정보")],
        chatbot=gr.Chatbot(
            label="도시정책 전문가",
            height=520,
        ),
        title="",
        description=(
            "로컬 **Ollama**의 **DeepSeek-R1** 모델과 기존 `retriever` 체인을 사용합니다.\n"
            "※ 모델의 내부 사고(<think>...</think>)는 자동으로 숨겨집니다."
        ),
        theme="soft",
        retry_btn="다시 생성",
        undo_btn="이전 메시지 삭제",
        clear_btn="대화 초기화",
        examples=[
            "도시 열섬 완화를 위한 단기 정책은?",
            "보행친화구역 지정 시 고려해야 할 지표는?",
            "우리 구의 주차 수요 관리 방안 사례가 있을까?",
        ],
    )

    with gr.Accordion("시스템/모델 정보", open=False):
        gr.Markdown(
            f"""
- 사용 모델: **{MODEL_NAME}**  
- 로컬 서버: **http://localhost:11434** (Ollama 기본)  
- 시스템 프롬프트: `{SYSTEM_PROMPT}`  
- Temperature: `{TEMPERATURE}`  
- 사용 체인: `retriever.query_augmentation_chain`, `retriever.retriever`, `retriever.document_chain`
"""
        )

if __name__ == "__main__":
    # 내부망만 사용할 경우 share=False 권장
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
