"""
[OpenCode 변환본]

목표: Streamlit + OpenAI(Closed LLM) 기반 RAG 챗봇 코드를
→ **로컬 오픈소스 LLM (Ollama의 DeepSeek-R1)** + **Gradio UI**로 변환

핵심 변경점
1) OpenAI/키 삭제, **Ollama 로컬** 사용 (기본: http://localhost:11434)
2) 모델: deepseek-r1 (예: "deepseek-r1:latest" 또는 "deepseek-r1:7b/14b")
3) Streamlit → **Gradio ChatInterface** (스트리밍 지원)
4) 기존 retriever 모듈(query_augmentation_chain, retriever, document_chain) **그대로 사용**
5) DeepSeek-R1이 토출하는 내부 사고(<think>...</think>)는 사용자 출력에서 자동 제거

사전 준비
- Ollama 설치: https://ollama.com
- 모델 다운로드: `ollama pull deepseek-r1:latest`
- 파이썬 패키지: `pip install gradio langchain langchain-community`
- (중요) `retriever.py` 는 기존과 동일하게 모듈 경로에 존재해야 합니다.
"""

from __future__ import annotations
from typing import List, Tuple, Generator
import re
import textwrap

import gradio as gr

# LangChain 메시지 타입 (원 코드와 동일)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LangChain용 Ollama 챗 모델
from langchain_community.chat_models import ChatOllama

import retriever  # 기존 모듈 그대로 사용


# ====== 설정 ======
MODEL_NAME = "deepseek-r1:latest"  # 필요 시 "deepseek-r1:7b" 등으로 변경
TEMPERATURE = 0.7
SYSTEM_PROMPT = "너는 문서에 기반해 답변하는 도시 정책 전문가야"

# DeepSeek-R1의 내부 사고(<think>...</think>) 제거용
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """완성된 텍스트에서 <think> 블록을 제거."""
    return THINK_TAG_RE.sub("", text).strip()


def streaming_hide_think(chunks: Generator[str, None, None]) -> Generator[str, None, None]:
    """
    스트리밍 중간에도 <think> 안의 내용이 화면에 보이지 않도록 필터링.
    - 상태 머신 방식으로 '<think>' ~ '</think>' 사이를 숨김.
    """
    in_think = False
    buffer = ""

    for raw in chunks:
        # LangChain 일부 드라이버가 dict/CallbackEvent를 내보내는 경우도 있어 방어코딩
        s = str(raw)

        buffer += s
        out = []

        i = 0
        while i < len(buffer):
            if not in_think:
                # '<think>' 시작 토큰 탐색
                start = buffer.find("<think>", i)
                if start == -1:
                    # 남은 전부 출력 후보
                    out.append(buffer[i:])
                    buffer = ""  # 버퍼 비움
                    break
                else:
                    # 시작 전까지 출력 → 이후부터는 think 모드
                    out.append(buffer[i:start])
                    i = start + len("<think>")
                    in_think = True
            else:
                # think 블록 종료 토큰 탐색
                end = buffer.find("</think>", i)
                if end == -1:
                    # 종료 없으면 나머지는 버퍼에 남겨두고 더 받기
                    buffer = buffer[i:]
                    i = len(buffer)
                    break
                else:
                    # 종료 지점까지는 숨기고, 종료 뒤로 진행
                    i = end + len("</think>")
                    in_think = False
                    # think 블록 이후 텍스트가 뒤에 더 있을 수 있으니 계속 루프

        visible = "".join(out)
        if visible:
            yield visible


# ====== LLM 인스턴스 (로컬 Ollama) ======
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    # base_url="http://localhost:11434",  # 기본값이면 주석 가능
)


# ====== Gradio <-> LangChain 히스토리 변환 ======
def history_to_messages(history: List[Tuple[str, str]]) -> List:
    """
    Gradio ChatInterface의 history([[user, assistant], ...])를 LangChain 메시지 리스트로 변환.
    항상 SYSTEM_PROMPT를 맨 앞에 추가.
    """
    msgs: List = [SystemMessage(SYSTEM_PROMPT)]
    for user_text, ai_text in history:
        if user_text:
            msgs.append(HumanMessage(user_text))
        if ai_text:
            msgs.append(AIMessage(ai_text))
    return msgs


# ====== 문서 패널용 마크다운 생성 ======
def format_docs_markdown(docs) -> str:
    """
    retriever 결과(docs)를 접이식 섹션으로 표시할 마크다운 생성.
    - ChatInterface 의 additional_outputs 로 오른쪽 패널 갱신에 사용.
    """
    parts = ["### 🔎 검색된 문서 (RAG 컨텍스트)\n"]
    for idx, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page", "")
        header = f"**{idx}. 소스:** `{src}`  —  **page:** {page}"
        body = textwrap.indent(doc.page_content.strip(), prefix="> ")
        parts.append(f"{header}\n\n{body}\n")
        parts.append("---")
    return "\n".join(parts) if len(docs) > 0 else "_관련 문서를 찾지 못했습니다._"


# ====== 메인 응답 로직 ======
def respond(user_input: str, history: List[Tuple[str, str]]):
    """
    ChatInterface 콜백.
    1) 히스토리 + 사용자 질문으로 질의 확장(augmentation)
    2) 관련 문서 검색 및 우측 패널 갱신
    3) document_chain.stream 으로 생성 결과를 실시간 스트리밍(+ <think> 숨김)
    반환 형식:
      - 스트리밍 시: yield (partial_text, docs_markdown)
      - 종료 시: return 최종 전체 텍스트
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
        # 실패해도 RAG는 진행 (문제 원인 출력은 최소화)
    
    # 3) 관련 문서 검색
    try:
        docs = retriever.retriever.invoke(f"{user_input}\n{augmented_query}")
    except Exception:
        docs = []

    docs_md = format_docs_markdown(docs)

    # 4) 스트리밍 생성 (retriever.document_chain.stream)
    try:
        raw_stream = retriever.document_chain.stream({
            "messages": messages,
            "context": docs,
        })
    except Exception as e:
        # document_chain 동작 실패 시, 최소한의 오류 메시지 반환
        err_msg = f"문서 기반 응답을 생성하지 못했어요. 사유: {e}"
        yield err_msg, docs_md
        return

    # 내부 사고 숨김 스트리밍
    filtered_stream = streaming_hide_think(raw_stream)

    # Gradio는 (부분문자열, 추가출력) 형태의 튜플을 yield 해도 된다.
    for piece in filtered_stream:
        yield piece, docs_md  # docs 패널은 전체 과정 동안 동일하게 유지

    # 스트리밍이 끝난 뒤 추가 반환은 생략 가능 (위의 yield 들로 이미 완성됨)


# ====== Gradio UI 구성 ======
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 🏙️ 문서기반 도시정책 챗봇 (DeepSeek-R1 on Ollama + Gradio)")

    with gr.Row():
        chat = gr.ChatInterface(
            fn=respond,
            additional_outputs=[gr.Markdown(label="RAG 컨텍스트")],
            chatbot=gr.Chatbot(
                label="도시정책 전문가",
                height=520,
            ),
            title="",
            description=(
                "로컬 **Ollama**의 **DeepSeek-R1** 모델과 기존 `retriever` 체인을 이용해 "
                "문서 기반 질의응답을 수행합니다.\n"
                "※ 내부 사고(<think>...</think>)는 자동으로 숨겨집니다."
            ),
            theme="soft",
            retry_btn="다시 생성",
            undo_btn="이전 메시지 삭제",
            clear_btn="대화 초기화",
            examples=[
                "도시 열섬 현상을 완화하기 위한 단기 정책은?",
                "보행친화구역 지정 시 고려해야 할 지표를 알려줘.",
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
- 사용 모듈: `retriever.query_augmentation_chain`, `retriever.retriever`, `retriever.document_chain`
"""
        )

if __name__ == "__main__":
    # 내부망만 사용할 경우 share=False 권장
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

