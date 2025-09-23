# -*- coding: utf-8 -*-
"""
[OpenCode 변환본]

- 목적: Closed LLM(OpenAI ChatGPT) 의존 코드를 *로컬 오픈소스 LLM* (Ollama + DeepSeek-R1) 기반으로
        동작하도록 변환하고, **Gradio** 기반의 간단한 그래픽 인터페이스까지 제공합니다.

- 전제:
  1) 로컬 PC에 Ollama가 설치되어 있고, deepseek-r1 계열 모델이 로컬에 존재한다고 가정합니다.
     (예: `ollama pull deepseek-r1:7b` 또는 `ollama pull deepseek-r1:32b`)
  2) 기존 유틸 함수(utils.py)로 `save_state`, `get_outline`, `save_outline` 가 제공된다고 가정합니다.
  3) 본 코드는 API Key가 전혀 필요하지 않습니다. (sk-... 제거)

- 실행 전 준비:
  pip install langchain langgraph langchain-ollama gradio
  또는 (환경에 따라) pip install langchain langgraph langchain-community gradio

- 실행:
  python app.py  (파일명을 app.py라고 가정)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List

# LangGraph / LangChain
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 메시지 타입 (LangChain Core)
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

# Ollama(로컬) 기반 챗 모델
# 최신 LangChain에서는 `langchain_ollama` 패키지의 ChatOllama 사용을 권장합니다.
# 환경에 따라 `langchain_community.chat_models` 의 ChatOllama 를 사용해야 할 수 있습니다.
try:
    from langchain_ollama import ChatOllama  # 최신
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # 구버전/대체
    except Exception as e:
        raise ImportError(
            "ChatOllama 를 불러올 수 없습니다. `pip install langchain-ollama` "
            "또는 `pip install langchain-community`를 확인하세요."
        )

# Gradio GUI
import gradio as gr

# 유틸 (저장/불러오기) — 프로젝트에 포함되어 있다고 가정
try:
    from utils import save_state, get_outline, save_outline
except Exception:
    # utils 부재 시에도 데모 동작 가능하도록 간단한 대체 구현 제공
    def save_state(path: str, state_obj: dict):
        os.makedirs(os.path.join(path, "_runs"), exist_ok=True)
        with open(os.path.join(path, "_runs", "state.txt"), "w", encoding="utf-8") as f:
            f.write(str(state_obj))

    def get_outline(path: str) -> str:
        fpath = os.path.join(path, "_runs", "outline.md")
        if os.path.exists(fpath):
            return open(fpath, "r", encoding="utf-8").read()
        return ""

    def save_outline(path: str, content: str):
        os.makedirs(os.path.join(path, "_runs"), exist_ok=True)
        with open(os.path.join(path, "_runs", "outline.md"), "w", encoding="utf-8") as f:
            f.write(content)


# =========================
# 경로/파일 정보
# =========================
try:
    filename = os.path.basename(__file__)
    absolute_path = os.path.abspath(__file__)
    current_path = os.path.dirname(absolute_path)
except NameError:
    # 노트북/REPL 등에서 __file__ 이 없을 경우 대비
    filename = "app.py"
    absolute_path = os.path.abspath("./app.py")
    current_path = os.path.dirname(absolute_path)


# =========================
# 로컬 LLM(DeepSeek-R1 on Ollama) 초기화
# =========================
# - 모델 이름은 로컬 설치된 태그에 맞게 변경 가능:
#   예: "deepseek-r1:7b", "deepseek-r1:32b"
# - stream 지원 및 합리적인 파라미터 기본값 지정
llm = ChatOllama(
    model="deepseek-r1:7b",  # 필요 시 "deepseek-r1:32b" 등으로 교체
    temperature=0.7,
    # Ollama 생성 파라미터 예시 (환경/버전에 따라 지원 키가 다를 수 있습니다)
    # num_ctx=4096,
    # top_p=0.9,
)


# =========================
# 상태 정의 (LangGraph)
# =========================
class State(TypedDict):
    messages: List[AnyMessage | str]


# =========================
# 노드(에이전트) 정의
# =========================
def content_strategist(state: State):
    """
    [콘텐츠 전략가]
    - 이전 대화와 저장된 목차(outline)를 토대로 새로운/수정된 세부 목차를 제안.
    - 제안된 목차는 파일로 저장(save_outline).
    - 사용자에게는 "목차 작성 완료"라는 메타 메시지만 전달(목차 자체는 커뮤니케이터가 반복 출력하지 않도록).
    """
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

    # LangChain Expression Language(LCEL): 프롬프트 -> LLM -> 문자열 파서
    chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {"messages": messages, "outline": outline}

    # 스트리밍 수집 (콘솔에도 실시간 출력)
    gathered = ""
    for chunk in chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")
    print()

    # 목차 저장
    save_outline(current_path, gathered)

    # 상태 메시지에 "완료" 메타 정보만 추가
    done_msg = "[Content Strategist] 목차 작성 완료"
    print(done_msg)
    messages.append(AIMessage(content=done_msg))

    return {"messages": messages}


def communicator(state: State):
    """
    [커뮤니케이터]
    - 팀의 진행상황을 사용자에게 보고하고, 피드백을 수집한다.
    - 사용자는 이미 outline 파일을 보고 있다고 가정하므로, 여기서 목차 전문을 재출력하지 않는다.
    """
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

    # 스트림 수신 (LLM이 토큰 단위로 생성할 경우)
    gathered_msg = None
    print("\nAI\t: ", end="")
    for chunk in chain.stream(inputs):
        # chunk 는 ChatMessage 타입에 준하는 토막
        print(chunk.content, end="")
        if gathered_msg is None:
            gathered_msg = chunk
        else:
            # LangChain의 메시지 합성 연산자는 버전에 따라 다를 수 있어 안전하게 이어붙임
            gathered_msg = type(chunk)(
                content=(gathered_msg.content + chunk.content),
                additional_kwargs=getattr(chunk, "additional_kwargs", {}),
                response_metadata=getattr(chunk, "response_metadata", {}),
                id=getattr(chunk, "id", None),
                name=getattr(chunk, "name", None),
                example=getattr(chunk, "example", False),
                tool_calls=getattr(chunk, "tool_calls", None),
                invalid_tool_calls=getattr(chunk, "invalid_tool_calls", None),
            )

    messages.append(gathered_msg)
    return {"messages": messages}


# =========================
# 상태 그래프 구성
# =========================
graph_builder = StateGraph(State)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("communicator", communicator)

graph_builder.add_edge(START, "content_strategist")
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# 그래프 시각화 (옵션)
try:
    graph.get_graph().draw_mermaid_png(
        output_file_path=absolute_path.replace(".py", ".png")
    )
except Exception as e:
    # graphviz 미설치 등으로 실패할 수 있으므로 조용히 패스
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
        ]
    )


# =========================
# Gradio UI
# =========================
"""
구성:
- 좌측: Chatbot (대화 기록)
- 하단: 사용자 입력창 + 전송 버튼
- 우측: 상태 정보(메시지 카운트, 저장 경로 안내) 및 컨트롤(초기화 버튼)

동작:
1) 사용자가 텍스트를 입력하고 전송하면,
2) 해당 입력을 State.messages 에 HumanMessage 로 추가,
3) LangGraph 파이프라인(graph.invoke) 실행,
4) Communicator 가 생성한 최신 AI 응답을 Chatbot 에 반영.
"""

with gr.Blocks(title="OpenCode • DeepSeek-R1 (Ollama) • LangGraph + Gradio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 OpenCode: LangGraph + DeepSeek-R1 (Ollama) + Gradio  
        - **Closed LLM → 오픈소스 LLM 변환본**  
        - 로컬 **Ollama** + **DeepSeek-R1** 모델을 사용하며, **API Key 불필요**  
        - 콘텐츠 전략가 ➜ 커뮤니케이터 에이전트 파이프라인으로 목차/대화 진행
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=520, label="대화창")
            user_in = gr.Textbox(
                placeholder="여기에 메시지를 입력하세요. (예: 챕터 2는 예제로 채워주세요)",
                label="User 입력",
            )
            send_btn = gr.Button("전송", variant="primary")

        with gr.Column(scale=2):
            info = gr.Markdown(
                f"**작업 폴더**: `{current_path}`\n\n"
                " - outline은 `_runs/outline.md`에 저장됩니다.\n"
                " - state는 `_runs/state.txt`에 저장됩니다.\n"
            )
            msg_count = gr.Number(value=1, precision=0, label="메시지 개수", interactive=False)
            reset_btn = gr.Button("세션 초기화", variant="secondary")

    # 세션 보관용 (State 객체)
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
        - 마지막 AI 응답을 history 에 반영
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
            elif hasattr(m, "content") and m.__class__.__name__ == "AIMessageChunk":
                # 혹시 chunk 로 남아 있는 경우 대비
                ai_text = getattr(m, "content", "")
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

    # Enter 로도 전송
    user_in.submit(
        on_send,
        inputs=[user_in, st, chat],
        outputs=[user_in, st, chat, msg_count],
    )


# =========================
# CLI 모드 (옵션)
# =========================
if __name__ == "__main__":
    # Gradio 서버 실행
    # - share=True: 외부에서 접속 필요 시 사용 (기본은 로컬만)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
