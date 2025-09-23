# -*- coding: utf-8 -*-
"""
[OpenCode 변환본: Closed LLM(OpenAI) → 오픈소스 LLM(Ollama • DeepSeek‑R1) + Gradio GUI]

본 스크립트는 원본 코드에서 `ChatOpenAI` 의존성을 제거하고,
로컬에서 구동되는 **Ollama + DeepSeek‑R1** 모델을 사용하도록 변환했습니다.
또한 터미널 입력 루프 대신 **Gradio** 기반의 간단한 그래픽 인터페이스를 제공합니다.

전제:
- 로컬 PC에 Ollama가 설치되어 있고, deepseek-r1 계열 모델을 pull 완료
  예) `ollama pull deepseek-r1:7b` 또는 `ollama pull deepseek-r1:32b`
- `utils.py`에 `save_state`, `get_outline`, `save_outline` 함수가 존재(없어도 안전 폴백 제공)
- API Key 불필요 (sk-... 일체 사용하지 않음)

설치:
    pip install langchain langgraph gradio
    pip install langchain-ollama  # (권장)
    # 또는 환경에 따라
    pip install langchain-community

실행:
    python opencode_supervisor_langgraph_ollama_gradio.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List

# LangGraph / LangChain
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 메시지 타입/프롬프트/파서
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

# Ollama(로컬 LLM) — 패키지 버전에 따라 import 경로가 다를 수 있어 try/except 처리
try:
    from langchain_ollama import ChatOllama  # 최신 권장
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # 대체 경로
    except Exception as e:
        raise ImportError(
            "ChatOllama 를 불러올 수 없습니다. `pip install langchain-ollama`\n"
            "또는 `pip install langchain-community`를 설치하세요."
        )

# GUI
import gradio as gr

# utils.py 에서 필요한 함수들을 가져옵니다.
from utils import save_state, get_outline, save_outline


# =========================
# 경로/파일 정보
# =========================
try:
    FILENAME = os.path.basename(__file__)
    ABS_PATH = os.path.abspath(__file__)
    CUR_PATH = os.path.dirname(ABS_PATH)
except NameError:
    # 노트북/REPL 등에서 __file__ 이 없을 경우
    FILENAME = "opencode_supervisor_langgraph_ollama_gradio.py"
    ABS_PATH = os.path.abspath("./" + FILENAME)
    CUR_PATH = os.path.dirname(ABS_PATH)


# =========================
# 로컬 LLM 초기화 (DeepSeek‑R1 on Ollama)
# =========================
# 필요 시 model 태그를 환경에 맞게 변경하세요. (예: "deepseek-r1:32b")
llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.7,
    # 아래 파라미터는 Ollama 및 LangChain 버전에 따라 지원 여부가 다를 수 있습니다.
    # num_ctx=4096,
    # top_p=0.9,
)


# =========================
# 상태 정의
# =========================
class State(TypedDict):
    messages: List[AnyMessage | str]
    task: str


# =========================
# Supervisor 에이전트 — 어떤 에이전트를 호출할지 결정
# =========================

def supervisor(state: State):
    print("\n\n============ SUPERVISOR ============")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고, 
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.     
        - content_strategist: 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 보고하고, 다음 지시를 물어본다. 

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.

        ------------------------------------------
        previous_outline: {outline}
        ------------------------------------------
        messages:
        {messages}
        """
    )

    chain = supervisor_system_prompt | llm | StrOutputParser()

    messages = state.get("messages", [])
    inputs = {"messages": messages, "outline": get_outline(CUR_PATH)}

    # 단발 호출(스트리밍 불필요)로 task 결정
    task = chain.invoke(inputs).strip()

    # Supervisor 로그 메시지
    sup_msg = AIMessage(content=f"[Supervisor] {task}")
    messages.append(sup_msg)
    print(sup_msg.content)

    return {"messages": messages, "task": task}


# =========================
# Router — supervisor 출력에 따라 다음 노드로 이동
# =========================

def supervisor_router(state: State):
    task = (state.get("task") or "").strip().lower()
    # supervisor가 정확히 노드 이름을 말한다고 가정하되, 약간의 관용 처리
    if "content" in task:
        return "content_strategist"
    if "communicator" in task or task == "ask" or "사용자" in task:
        return "communicator"
    # 기본값: communicator로 유도하여 사용자에게 질의
    return "communicator"


# =========================
# 콘텐츠 전략가 — 목차 작성/갱신 후 저장
# =========================

def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

        --------------------------------
        - 지난 목차: {outline}
        --------------------------------
        - 이전 대화 내용: {messages}
        """
    )

    chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]
    outline = get_outline(CUR_PATH)
    inputs = {"messages": messages, "outline": outline}

    # 스트리밍으로 목차 생성 (콘솔 출력)
    gathered = ""
    for chunk in chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")
    print()

    # 목차 저장
    save_outline(CUR_PATH, gathered)

    # 진행 상황 메시지
    done_msg = "[Content Strategist] 목차 작성 완료"
    print(done_msg)
    messages.append(AIMessage(content=done_msg))

    return {"messages": messages}


# =========================
# 커뮤니케이터 — 사용자와 대화(목차 전문은 재출력하지 않음)
# =========================

def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.

        messages: {messages}
        """
    )

    chain = communicator_system_prompt | llm

    messages = state["messages"]
    inputs = {"messages": messages}

    print("\nAI\t: ", end="")
    gathered = None
    for chunk in chain.stream(inputs):
        text = getattr(chunk, "content", "") if chunk else ""
        print(text, end="")
        if gathered is None:
            gathered = AIMessage(content=text)
        else:
            gathered.content += text

    if gathered is None:
        gathered = AIMessage(content="(응답 없음)")

    messages.append(gathered)
    return {"messages": messages}


# =========================
# 그래프 구성
# =========================

graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)

# 흐름: START → supervisor → (조건부) content_strategist 또는 communicator → communicator → END
graph_builder.add_edge(START, "supervisor")

graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
    },
)

# 목차 작성 후에는 사용자 커뮤니케이션으로 연결
graph_builder.add_edge("content_strategist", "communicator")

# 최종적으로 communicator 에서 종료
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# 시각화(옵션)
try:
    graph.get_graph().draw_mermaid_png(output_file_path=ABS_PATH.replace(".py", ".png"))
except Exception:
    pass


# =========================
# 초기 State
# =========================

def initial_state() -> State:
    return State(
        messages=[
            SystemMessage(
                content=f"""
            너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가팀이다.
            사용자가 사용하는 언어로 대화하라.

            현재시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.
            """
            )
        ],
        task="",
    )


# =========================
# Gradio UI
# =========================
"""
구성:
- 좌측: Chatbot (대화 기록)
- 하단: 사용자 입력창 + 전송 버튼
- 우측: 현재 Supervisor 판단(task), 메시지 수, 안내/초기화

동작:
1) 사용자가 입력 → HumanMessage 로 추가
2) LangGraph(graph.invoke) 실행 → supervisor 가 라우팅 결정
3) content_strategist 가 필요 시 목차 저장 후 communicator 가 사용자 응답 생성
4) 최신 AI 응답을 Chatbot 에 반영, state 저장
"""

with gr.Blocks(title="OpenCode • DeepSeek‑R1 (Ollama) • LangGraph + Gradio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 OpenCode: LangGraph + DeepSeek‑R1 (Ollama) + Gradio  
        - **Closed LLM → 오픈소스 LLM 변환본** - 로컬 **Ollama** + **DeepSeek‑R1** 모델 사용 (API Key 불필요)  
        - Supervisor → (Content Strategist | Communicator) → Communicator 파이프라인
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=520, label="대화창")
            user_in = gr.Textbox(placeholder="메시지를 입력하세요...", label="User 입력")
            send_btn = gr.Button("전송", variant="primary")
        with gr.Column(scale=2):
            task_box = gr.Textbox(label="Supervisor 판단(라우팅)", interactive=False)
            gr.Markdown(
                f"**작업 폴더**: `{CUR_PATH}`\n\n"
                " - outline은 `_runs/outline.md`에 저장됩니다.\n"
                " - state는 `_runs/state.txt`에 저장됩니다.\n"
            )
            msg_count = gr.Number(value=1, precision=0, label="메시지 개수", interactive=False)
            reset_btn = gr.Button("세션 초기화", variant="secondary")

    # 세션 상태
    st = gr.State(initial_state())

    def on_reset():
        s = initial_state()
        return s, [], "", 1

    reset_btn.click(on_reset, inputs=None, outputs=[st, chat, task_box, msg_count])

    def on_send(user_text: str, s: State, history: list[list[str, str]]):
        """Gradio 전송 핸들러"""
        user_text = (user_text or "").strip()
        if not user_text:
            return gr.update(), s, history, (s.get("task") or ""), len(s["messages"])

        # 사용자 메시지 추가
        s["messages"].append(HumanMessage(user_text))
        history = history + [[user_text, None]]

        # 그래프 실행 (START→END 전체 플로우)
        new_state = graph.invoke(s)

        # 최신 AI 응답 찾기 (communicator의 응답)
        ai_text = ""
        for m in reversed(new_state["messages"]):
            if isinstance(m, AIMessage):
                ai_text = m.content
                break

        # Chatbot 최신 턴 채우기
        if history and history[-1][1] is None:
            history[-1][1] = ai_text or "(응답 없음)"

        # 상태 저장
        try:
            save_state(CUR_PATH, new_state)
        except Exception:
            pass

        # Supervisor의 최종 판단값 표시
        task_display = (new_state.get("task") or "").strip()

        return "", new_state, history, task_display, len(new_state["messages"])

    send_btn.click(on_send, inputs=[user_in, st, chat], outputs=[user_in, st, chat, task_box, msg_count])
    user_in.submit(on_send, inputs=[user_in, st, chat], outputs=[user_in, st, chat, task_box, msg_count])


# =========================
# 메인 진입 — Gradio 서버 실행
# =========================
if __name__ == "__main__":
    # 외부 접근이 필요하면 share=True 또는 server_name 수정
    try:
        # 그래프 구조 이미지 저장 시도 (옵션)
        graph.get_graph().draw_mermaid_png(output_file_path=ABS_PATH.replace(".py", ".png"))
    except Exception:
        pass

    # Gradio 런치
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)