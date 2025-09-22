"""
[OpenCode 변환본]

목표: LangGraph + ChatOpenAI(Closed LLM) 콘솔 코드를
→ **로컬 오픈소스 LLM(Ollama · DeepSeek-R1)** + **Gradio UI**로 변환

핵심 변경점
1) ChatOpenAI → ChatOllama (API 키 불필요, 로컬 http://localhost:11434)
2) 콘솔 while-loop → Gradio ChatInterface
3) LangGraph(StateGraph, MemorySaver) 유지: 대화 상태/체크포인트 그대로 활용
4) DeepSeek-R1이 생성하는 내부 사고(<think>...</think>)는 사용자 출력에서 자동 제거
5) 타입 안전: LangChain 메시지 타입(list[BaseMessage])로 상태 정의

사전 준비
- Ollama 설치: https://ollama.com  후 모델 받기
  `ollama pull deepseek-r1:latest`
- 파이썬 패키지:
  `pip install gradio langchain langchain-community langgraph`

실행
- `python app.py` 실행 후 브라우저에서 확인
"""

from __future__ import annotations
import re
import uuid
from typing import Annotated, List, TypedDict

import gradio as gr

# LangChain 메시지 타입
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
# LangChain Ollama 챗 모델 (로컬 LLM)
from langchain_community.chat_models import ChatOllama

# LangGraph (상태 그래프, 체크포인트)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ===== 설정 =====
MODEL_NAME = "deepseek-r1:latest"      # 필요 시 "deepseek-r1:7b", "deepseek-r1:14b" 등
TEMPERATURE = 0.7
SYSTEM_PROMPT = "너는 사용자를 도와주는 상담사야."

# DeepSeek-R1 내부 사고(<think>...</think>) 제거용
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """DeepSeek-R1이 출력하는 내부 사고 블록을 제거."""
    return THINK_TAG_RE.sub("", text).strip()


# ===== LangGraph 상태 정의 =====
class State(TypedDict):
    """
    LangGraph에서 사용할 상태 스키마.

    - messages: 모든 대화 메시지 목록(BaseMessage). add_messages를 사용하면
      새 메시지를 덮어쓰지 않고 리스트에 자동으로 누적된다.
    """
    messages: Annotated[List[BaseMessage], add_messages]


# ===== 로컬 LLM (Ollama) =====
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    # base_url="http://localhost:11434",  # 기본값이면 생략 가능
)


# ===== 그래프 노드: 모델 응답 생성 =====
def generate(state: State) -> dict:
    """
    현재까지의 state["messages"]를 바탕으로 모델에 질의하고,
    모델이 생성한 AIMessage를 messages에 추가하여 반환한다.
    """
    # llm.invoke는 AIMessage를 반환
    ai_msg = llm.invoke(state["messages"])
    # DeepSeek의 <think> 제거
    cleaned = strip_think(ai_msg.content)
    ai_msg = AIMessage(cleaned)
    return {"messages": [ai_msg]}


# ===== 그래프 구성 =====
graph_builder = StateGraph(State)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

# 체크포인트(메모리) 설정
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# ===== Gradio ↔ LangGraph 브릿지 유틸 =====
def ensure_thread_id(thread_id_state: str | None) -> str:
    """세션별 thread_id를 생성/유지."""
    return thread_id_state or str(uuid.uuid4())


def bootstrap_if_needed(initialized: bool, thread_id: str):
    """
    최초 1회에 한해 시스템 메시지를 스레드에 주입.
    LangGraph는 checkpointer를 쓰므로 같은 thread_id로 호출하면 이전 상태를 기억한다.
    """
    if initialized:
        return True  # 이미 초기화됨

    # START 단계에서 시스템 메시지를 먼저 넣고 generate를 실행하도록
    # graph.stream 호출로 부트스트랩한다.
    init_state = {
        "messages": [SystemMessage(SYSTEM_PROMPT)]
    }
    # 한 번 돌려 상태를 저장(메모리에 checkpointer)
    for _ in graph.stream(init_state, {"configurable": {"thread_id": thread_id}}, stream_mode="values"):
        # 생성된 메시지는 UI에 바로 보여줄 필요 없음(시스템 프롬프트만 넣었기 때문)
        pass
    return True


# ====== Gradio 콜백 ======
def respond(user_input: str, history: List[tuple[str, str]], thread_id_state: str, inited: bool):
    """
    Gradio ChatInterface 콜백.
    - 세션 thread_id 확보 → LangGraph 체크포인트와 연결
    - 최초 요청이면 시스템 메시지로 부트스트랩
    - 사용자 입력을 그래프에 흘려보내고, 노드 실행 결과를 순차 스트리밍
    """
    thread_id = ensure_thread_id(thread_id_state)
    inited = bootstrap_if_needed(inited, thread_id)

    # LangGraph 실행: HumanMessage를 입력으로 전달
    # stream_mode="values"로 각 노드 실행 후의 전체 상태를 받는다.
    full_text = ""
    for event in graph.stream(
        {"messages": [HumanMessage(user_input)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        # event["messages"]는 지금까지 누적된 모든 메시지
        last = event["messages"][-1]
        if isinstance(last, AIMessage):
            # 이번 턴에 생성된 최신 AI 응답만 스트리밍
            chunk = last.content
            full_text = chunk
            yield chunk, thread_id, True  # (응답 텍스트, thread_id 유지, 초기화 여부 True)

    # 마지막 한 번 더 반환할 필요는 없지만, 안전하게 최종값을 재전송하지 않고 종료
    return


# ====== Gradio UI ======
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 💬 LangGraph Chat (DeepSeek-R1 on Ollama + Gradio)")

    # 세션 상태: thread_id 및 초기화 여부
    thread_id_state = gr.State("")
    initialized = gr.State(False)

    chat = gr.ChatInterface(
        fn=respond,
        additional_inputs=[thread_id_state, initialized],
        additional_outputs=[thread_id_state, initialized],
        chatbot=gr.Chatbot(
            label="상담사",
            height=520,
        ),
        title="",
        description=(
            "LangGraph + 로컬 **Ollama(DeepSeek-R1)** 기반 챗봇입니다.\n"
            "세션별로 LangGraph의 메모리(MemorySaver)를 사용해 대화 상태를 유지합니다.\n"
            "※ 모델의 내부 사고(<think>...</think>)는 자동으로 숨깁니다."
        ),
        theme="soft",
        retry_btn="다시 생성",
        undo_btn="이전 메시지 삭제",
        clear_btn="대화 초기화",
        examples=[
            "요즘 스트레스가 많아요. 어떻게 관리하면 좋을까요?",
            "업무 몰입도를 높이는 데 도움 되는 루틴을 추천해줘.",
            "회의에서 의견을 설득력 있게 전달하는 팁이 있을까?",
        ],
    )

    with gr.Accordion("시스템/모델 정보", open=False):
        gr.Markdown(
            f"""
- 사용 모델: **{MODEL_NAME}**  
- Temperature: **{TEMPERATURE}**  
- 로컬 서버: **http://localhost:11434** (Ollama 기본)  
- 시스템 프롬프트: `{SYSTEM_PROMPT}`
- LangGraph: `StateGraph`, `MemorySaver` 사용 (세션별 thread_id로 상태 유지)
"""
        )

if __name__ == "__main__":
    # 내부망만 사용할 경우 share=False 권장
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


