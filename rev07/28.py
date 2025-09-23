"""
[OpenCode 변환본]
- 목적: Closed LLM(OpenAI ChatGPT) 의존 코드를 오픈소스 LLM(로컬 Ollama + DeepSeek-R1)로 변환
- 추가: Gradio 기반의 간단한 그래픽 인터페이스(GUI) 제공
- 가정: 로컬 PC에 Ollama가 설치되어 있고 deepseek-r1 모델이 pull 되어 있음
        (예: `ollama pull deepseek-r1:latest`)
- 주의: OpenAI API Key 등 외부 키 사용 없음
"""

from __future__ import annotations

# 표준 라이브러리
from typing import List, Dict, Any
from typing_extensions import TypedDict
from datetime import datetime
import json
import os
import re
import traceback

# LangChain / LangGraph
from langgraph.graph import StateGraph, START, END
# ✅ OpenAI → Ollama 전환: ChatOpenAI 대신 ChatOllama 사용
#   최신 LangChain에서는 `langchain_community`에 ChatOllama가 있습니다.
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, AIMessage, BaseMessage
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 사용자 모듈(기존 코드 유지)
from utils import save_state, get_outline, save_outline
from models import Task  # Pydantic 모델이라고 가정

# GUI
import gradio as gr


# ──────────────────────────────────────────
# 경로/파일 설정
# ──────────────────────────────────────────

filename = os.path.basename(__file__)
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)


# ──────────────────────────────────────────
# LLM 초기화 (Ollama + DeepSeek-R1)
# ──────────────────────────────────────────
"""
- OpenAI용 ChatOpenAI → 오픈소스용 ChatOllama
- 모델명은 로컬 Ollama에서 사용 가능한 deepseek-r1 태그를 사용합니다.
  (예: deepseek-r1:latest, deepseek-r1:7b 등)
- DeepSeek-R1은 사고흐름(<think>...</think>)을 내보낼 수 있으므로,
  시스템/템플릿에서 '최종 출력은 JSON' 등으로 엄격히 제약하여 파싱 안정성을 높입니다.
"""
llm = ChatOllama(
    model="deepseek-r1:latest",  # 필요 시 설치된 태그로 변경
    temperature=0.7,
    # streaming=True  # LangChain stream 사용 시 자동 처리되므로 명시 불필요
)


# ──────────────────────────────────────────
# 상태 정의
# ──────────────────────────────────────────

class State(TypedDict):
    messages: List[AnyMessage | str]
    task_history: List[Task]


# ──────────────────────────────────────────# JSON 유틸: Supervisor 출력 → Task 파싱
# ──────────────────────────────────────────

def extract_json_block(text: str) -> Dict[str, Any]:
    """
    DeepSeek-R1이 <think> 블록 등을 포함하여 출력할 수 있으므로,
    가장 그럴듯한 JSON 객체를 안전하게 추출합니다.
    - 코드블록 ```json ... ``` 우선
    - 중괄호 균형 맞는 첫 객체 추출(단순 휴리스틱)
    """
    # 1) ```json ... ``` 블록 우선 탐색
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2) 첫 번째로 균형 맞는 {} 객체 추정 추출
    stack = 0
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start_idx = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start_idx != -1:
                candidate = text[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # 계속 탐색
                    start_idx = -1
    # 실패 시 빈 딕셔너리
    return {}


def to_task(obj: Dict[str, Any]) -> Task:
    """
    JSON dict → Task(Pydantic)로 엄격 변환.
    필수 필드: agent(str), description(str)
    선택 필드: done(bool), done_at(str)
    """
    # 합리적 기본값
    agent = str(obj.get("agent", "")).strip()
    description = str(obj.get("description", "")).strip()
    done = bool(obj.get("done", False))
    done_at = str(obj.get("done_at", ""))

    if not agent:
        # 기본 라우팅 실패 방지용: communicator로 폴백
        agent = "communicator"
    if not description:
        description = "사용자와 대화를 통해 다음 단계를 파악한다"

    return Task(agent=agent, description=description, done=done, done_at=done_at)


# ──────────────────────────────────────────
# Supervisor 에이전트
# ──────────────────────────────────────────

def supervisor(state: State):
    print("\n\n============ SUPERVISOR ============")

    # DeepSeek-R1이 사고흐름을 내보내더라도,
    # 최종 출력은 JSON 한 덩어리로 하도록 엄격히 지시
    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 쓰는 최종 목표를 염두에 두고,
        지금 당장 수행해야 할 작업(agent)과 그 이유/설명을 결정한다.

        사용 가능한 agent:
        - content_strategist: 요구사항이 명확해졌을 때. 콘텐츠 전략을 확정하고 전체 책의 목차(outline)를 작성/수정한다.
        - communicator: 아직 판단이 서지 않거나 사용자 피드백이 필요할 때. 진행상황을 보고하고 다음 지시를 묻는다.

        다음 정보를 참고하라.
        ------------------------------------------
        previous_outline: {outline}
        ------------------------------------------
        messages:
        {messages}
        ------------------------------------------

        출력 형식(반드시 이 JSON만 출력하라):
        ```json
        {{
          "agent": "content_strategist" | "communicator",
          "description": "간단하고 명확한 할 일 설명",
          "done": false,
          "done_at": ""
        }}
        ```
        """
    )

    # 기존 with_structured_output(Task) → 직접 JSON 파싱 방식으로 변경
    chain = supervisor_system_prompt | llm | StrOutputParser()

    messages = state.get("messages", [])
    inputs = {
        "messages": messages,
        "outline": get_outline(current_path),
    }

    raw = chain.invoke(inputs)
    data = extract_json_block(raw)
    task = to_task(data)

    task_history = state.get("task_history", [])
    task_history.append(task)

    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    return {"messages": messages, "task_history": task_history}


# ──────────────────────────────────────────
# 라우터
# ──────────────────────────────────────────

def supervisor_router(state: State):
    task = state["task_history"][-1]
    return task.agent


# ──────────────────────────────────────────
# 콘텐츠 전략가
# ──────────────────────────────────────────

def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)다.
        이전 대화와 기존 목차를 바탕으로 사용자의 요구사항을 분석하여,
        책의 세부 목차를 작성하거나(없으면 생성) 기존 목차를 개선한다.

        출력은 '최종 목차 텍스트'만 산출하라. 불필요한 설명은 제외한다.
        --------------------------------
        지난 목차:
        {outline}
        --------------------------------
        이전 대화 내용:
        {messages}
        """
    )

    chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {"messages": messages, "outline": outline}

    # 스트림 출력(콘솔)
    gathered = ""
    for chunk in chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")
    print()

    # 목차 저장
    save_outline(current_path, gathered)

    # 상태 메시지 추가
    content_strategist_message = "[Content Strategist] 목차 작성 완료"
    print(content_strategist_message)
    messages.append(AIMessage(content=content_strategist_message))

    task_history = state.get("task_history", [])
    if task_history[-1].agent != "content_strategist":
        raise ValueError(
            "Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n"
            f"{task_history[-1]}"
        )
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 다음 작업: 사용자와 커뮤니케이션
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at="",
    )
    task_history.append(new_task)
    print(new_task)

    return {"messages": messages, "task_history": task_history}


# ──────────────────────────────────────────
# 커뮤니케이터
# ──────────────────────────────────────────

def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터다.
        AI팀의 진행상황을 사용자에게 간결히 보고하고, 다음 지시를 받는다.
        사용자는 이미 outline(목차)을 보고 있다고 가정하므로 목차 원문을 다시 출력할 필요는 없다.

        참고(표시 금지):
        outline: {outline}
        --------------------------------
        messages: {messages}
        """
    )

    system_chain = communicator_system_prompt | llm

    messages = state["messages"]
    inputs = {
        "messages": messages,
        "outline": get_outline(current_path),
    }

    # 스트리밍 수집
    print("\nAI\t: ", end="")
    gathered: BaseMessage | None = None
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end="")
        if gathered is None:
            gathered = chunk
        else:
            # BaseMessage 덧셈은 지원되므로 누적
            gathered += chunk
    print()

    if gathered is None:
        gathered = AIMessage("죄송합니다. 응답을 생성하지 못했습니다. 다시 시도해 주세요.")

    messages.append(gathered)

    task_history = state.get("task_history", [])
    if task_history[-1].agent != "communicator":
        raise ValueError(
            "Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n"
            f"{task_history[-1]}"
        )
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {"messages": messages, "task_history": task_history}


# ──────────────────────────────────────────
# LangGraph 빌드
# ──────────────────────────────────────────

graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
    },
)
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# Mermaid PNG 저장(선택)
try:
    graph.get_graph().draw_mermaid_png(
        output_file_path=absolute_path.replace(".py", ".png")
    )
except Exception:
    # graphviz/mermaid 설치가 안 되어 있을 수 있으므로 실패해도 무시
    pass




# ──────────────────────────────────────────
# 상태 초기화 함수
# ──────────────────────────────────────────

def init_state() -> State:
    return State(
        messages=[
            SystemMessage(
                f"""
            너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가팀이다.
            사용자가 사용하는 언어로 대화하라.

            현재시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.
            """
            )
        ],
        task_history=[],
    )


# ──────────────────────────────────────────
# CLI 루프(옵션) - 기존 기능 유지
# ──────────────────────────────────────────

def run_cli():
    state = init_state()
    while True:
        user_input = input("\nUser\t: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(user_input))
        state = graph.invoke(state)

        print("\n------------------------------------ MESSAGE COUNT\t", len(state["messages"]))
        save_state(current_path, state)


# ──────────────────────────────────────────
# Gradio GUI
# ──────────────────────────────────────────

"""
간단한 1-페이지 UI:
- 좌측: 채팅(히스토리)
- 우측 상단: 현재 저장된 outline 미리보기
- 하단: 사용자 입력 박스 + 전송 버튼 + 상태 초기화 버튼
"""
def ui_on_send(user_text: str, state: State):
    try:
        if state is None:
            state = init_state()

        if user_text.strip():
            state["messages"].append(HumanMessage(user_text.strip()))
            # LangGraph 실행
            new_state = graph.invoke(state)
            # 마지막 AI 메시지 추출
            ai_text = ""
            for m in reversed(new_state["messages"]):
                if isinstance(m, AIMessage):
                    ai_text = m.content
                    break

            # 상태 저장
            save_state(current_path, new_state)

            # 채팅 로그 문자열로 합성(간단 표현)
            chat_log = render_chat(new_state["messages"])
            # 최신 outline 불러오기
            outline_text = get_outline(current_path) or "(아직 목차가 없습니다.)"

            return chat_log, outline_text, new_state
        else:
            chat_log = render_chat(state["messages"])
            outline_text = get_outline(current_path) or "(아직 목차가 없습니다.)"
            return chat_log, outline_text, state
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"[에러] {e}\n\n{tb}"
        chat_log = render_chat(state["messages"] if state else [])
        return chat_log + "\n\n" + err_msg, get_outline(current_path), state


def ui_on_reset():
    state = init_state()
    return render_chat(state["messages"]), get_outline(current_path) or "(아직 목차가 없습니다.)", state


def render_chat(messages: List[BaseMessage | str]) -> str:
    """
    간단한 텍스트 렌더링(Gradio Markdown에 표시).
    필요하면 더 예쁘게 커스터마이징 가능.
    """
    lines = []
    for msg in messages:
        role = "SYS"
        content = ""
        if isinstance(msg, SystemMessage):
            role = "SYSTEM"
            content = msg.content
        elif isinstance(msg, HumanMessage):
            role = "USER"
            content = msg.content
        elif isinstance(msg, AIMessage):
            role = "AI"
            content = msg.content
        elif isinstance(msg, str):
            role = "TEXT"
            content = msg
        else:
            try:
                content = getattr(msg, "content", str(msg))
            except Exception:
                content = str(msg)
            role = msg.__class__.__name__.upper()
        lines.append(f"**{role}:** {content}")
    return "\n\n".join(lines)


def launch_gradio():
    with gr.Blocks(title="AI Book Team (Ollama + DeepSeek-R1)") as demo:
        gr.Markdown("## ✍️ AI Book Team (Local Ollama · DeepSeek-R1)\n오른쪽에 최신 목차가 표시됩니다.")
        with gr.Row():
            with gr.Column(scale=2):
                chat_md = gr.Markdown(value="")
                user_in = gr.Textbox(label="메시지 입력", placeholder="예) 'AI로 책을 쓰고 싶어요. 주제는 ...'")
                with gr.Row():
                    send_btn = gr.Button("보내기", variant="primary")
                    reset_btn = gr.Button("대화 초기화", variant="secondary")
            with gr.Column(scale=1):
                outline_md = gr.Markdown(label="현재 목차", value="(아직 목차가 없습니다.)")

        state_box = gr.State(init_state())

        send_btn.click(
            fn=ui_on_send,
            inputs=[user_in, state_box],
            outputs=[chat_md, outline_md, state_box],
        )
        user_in.submit(
            fn=ui_on_send,
            inputs=[user_in, state_box],
            outputs=[chat_md, outline_md, state_box],
        )
        reset_btn.click(
            fn=ui_on_reset,
            inputs=[],
            outputs=[chat_md, outline_md, state_box],
        )

    demo.launch()  # 필요 시 share=True 등 옵션 추가 가능




# ──────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────

if __name__ == "__main__":
    """
    - 터미널에서 실행 시 기본은 Gradio GUI 실행
    - CLI 사용을 원하면 아래 run_cli() 호출로 바꾸거나,
      `python this.py --cli` 등으로 간단 파서 추가 가능
    """
    launch_gradio()
    # run_cli()

