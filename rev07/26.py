# -*- coding: utf-8 -*-
"""
[OpenCode 변환본: Closed LLM(OpenAI) → 오픈소스 LLM(Ollama • DeepSeek-R1) + Gradio GUI]

본 스크립트는 원본 코드에서 `ChatOpenAI`에 의존하던 부분을
로컬에서 구동되는 **Ollama + DeepSeek-R1** 모델로 대체하고,
터미널 루프 대신 **Gradio** 기반의 간단한 대화형 UI를 제공합니다.

전제:
- 로컬 PC에 Ollama가 설치되어 있고, deepseek-r1 계열 모델이 pull 되어 있음
  예) `ollama pull deepseek-r1:7b` 혹은 `ollama pull deepseek-r1:32b`
- `utils.py`에 `save_state` 함수가 존재한다고 가정
- API Key 불필요 (sk-... 전혀 사용하지 않음)

설치:
    pip install langchain langgraph gradio
    # 환경에 따라
    pip install langchain-ollama  # 권장
    # 또는
    pip install langchain-community

실행:
    python app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List

import gradio as gr
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

# LangChain 메시지/프롬프트/파서
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# Ollama(로컬 LLM) — 패키지 버전에 따라 import 경로가 다를 수 있어 try/except 처리
try:
    from langchain_ollama import ChatOllama  # 최신 권장
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # 대체 경로
    except Exception as e:
        raise ImportError(
            "ChatOllama 를 불러올 수 없습니다. `pip install langchain-ollama` "
            "또는 `pip install langchain-community`를 설치하세요."
        )

# utils.py 에서 필요한 함수를 가져옵니다.
from utils import save_state


# =========================
# 경로/파일 정보
# =========================
try:
    filename = os.path.basename(__file__)  # 현재 파일명
    absolute_path = os.path.abspath(__file__)
    current_path = os.path.dirname(absolute_path)
except NameError:
    # 노트북/REPL 등에서 __file__ 이 없을 경우 대비
    filename = "app.py"
    absolute_path = os.path.abspath("./app.py")
    current_path = os.path.dirname(absolute_path)


# =========================
# 로컬 LLM 초기화 (DeepSeek-R1 on Ollama)
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


# =========================
# 노드(에이전트): communicator
# =========================
def communicator(state: State):
    """
    사용자의 메시지 이력(State.messages)을 바탕으로
    '커뮤니케이터' 역할의 시스템 프롬프트를 구성하여 응답을 생성합니다.
    """
    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        messages: {messages}
        """
    )

    # LangChain Expression Language: 프롬프트 -> LLM
    system_chain = communicator_system_prompt | llm

    messages = state["messages"]
    inputs = {"messages": messages}

    # 스트리밍 수신 (콘솔에도 실시간 출력)
    print('\nAI\t: ', end='')
    gathered_msg = None
    for chunk in system_chain.stream(inputs):
        # chunk 는 AIMessage 혹은 해당 Chunks 일 수 있음
        text = getattr(chunk, "content", "") if chunk else ""
        print(text, end='')
        if gathered_msg is None:
            gathered_msg = AIMessage(content=text)
        else:
            gathered_msg.content += text

    # 대화 이력에 AI 응답 추가
    if gathered_msg is None:
        gathered_msg = AIMessage(content="(응답 생성 실패)")
    messages.append(gathered_msg)

    return {"messages": messages}


# =========================
# 그래프 구성
# =========================
graph_builder = StateGraph(State)
graph_builder.add_node("communicator", communicator)
graph_builder.add_edge(START, "communicator")
graph_builder.add_edge("communicator", END)
graph = graph_builder.compile()

# 그래프 시각화 (선택)
try:
    graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))
except Exception:
    # graphviz 미설치 등으로 인한 실패는 무시
    pass


# =========================
# 초기 State 생성
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
        ]
    )


# =========================
# Gradio UI
# =========================
"""
구성:
- 좌측: Chatbot (대화 기록)
- 하단: 사용자 입력창 + 전송 버튼
- 우측: 상태/컨트롤 (메시지 수, 작업 폴더 표시, 초기화 버튼)

동작:
1) 사용자가 입력 -> HumanMessage 로 추가
2) LangGraph 파이프라인(graph.invoke) 실행
3) communicator 노드의 최신 응답을 Chatbot에 표시
4) state 저장 (utils.save_state)
"""

with gr.Blocks(title="OpenCode • DeepSeek-R1 (Ollama) • LangGraph + Gradio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 OpenCode: LangGraph + DeepSeek-R1 (Ollama) + Gradio
        - **Closed LLM → 오픈소스 LLM 변환본**
        - 로컬 **Ollama** + **DeepSeek-R1** 모델 사용 (API Key 불필요)
        - 단일 노드(**communicator**)로 간단 대화 파이프라인
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=520, label="대화창")
            user_in = gr.Textbox(placeholder="메시지를 입력하세요...", label="User 입력")
            send_btn = gr.Button("전송", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown(
                f"**작업 폴더**: `{current_path}`\n\n"
                " - state는 `_runs/state.txt`에 저장됩니다.\n"
            )
            msg_count = gr.Number(value=1, precision=0, label="메시지 개수", interactive=False)
            reset_btn = gr.Button("세션 초기화", variant="secondary")

    st = gr.State(initial_state())

    def on_reset():
        s = initial_state()
        return s, [], 1

    reset_btn.click(on_reset, inputs=None, outputs=[st, chat, msg_count])

    def on_send(user_text: str, s: State, history: list[list[str, str]]):
        """
        Gradio 핸들러:
        - user_text 를 HumanMessage 로 추가
        - LangGraph 실행
        - 최신 AI 응답을 Chatbot 히스토리에 반영
        - state 저장
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return gr.update(), s, history, len(s["messages"])

        # 사용자 메시지 추가
        s["messages"].append(HumanMessage(user_text))
        history = history + [[user_text, None]]

        # LangGraph 파이프라인 실행
        new_state = graph.invoke(s)

        # 최신 AI 응답 추출
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
            save_state(current_path, new_state)
        except Exception:
            pass

        return "", new_state, history, len(new_state["messages"])

    send_btn.click(
        on_send,
        inputs=[user_in, st, chat],
        outputs=[user_in, st, chat, msg_count],
    )
    user_in.submit(
        on_send,
        inputs=[user_in, st, chat],
        outputs=[user_in, st, chat, msg_count],
    )


# =========================
# CLI 모드(옵션) — 필요 시 주석 해제하여 사용
# =========================
# def run_cli():
#     state = initial_state()
#     while True:
#         user_input = input("\nUser\t: ").strip()
#         if user_input.lower() in ['exit', 'quit', 'q']:
#             print("Goodbye!")
#             break
#         state["messages"].append(HumanMessage(user_input))
#         state = graph.invoke(state)
#         # 마지막 AI 응답 출력
#         for m in reversed(state["messages"]):
#             if isinstance(m, AIMessage):
#                 print("\nAI\t:", m.content)
#                 break
#         print('\n------------------------------------ MESSAGE COUNT\t', len(state["messages"]))
#         save_state(current_path, state)

if __name__ == "__main__":
    # Gradio 서버 실행 (기본 로컬만; 외부 접근 필요 시 share=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)