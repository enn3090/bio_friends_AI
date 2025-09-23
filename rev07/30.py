# 30.py (SyntaxError 수정 완료)

from __future__ import annotations

# 표준 라이브러리
from typing import List, Dict, Any
from typing_extensions import TypedDict
from datetime import datetime
import json
import os
import re
import traceback

# LangGraph / LangChain
from langgraph.graph import StateGraph, START, END
# ✅ OpenAI → Ollama 전환
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, AIMessage, BaseMessage
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 사용자 모듈 (기존 유지)
from utils import save_state, get_outline, save_outline
from models import Task
from tools import retrieve, web_search, add_web_pages_json_to_chroma  # ← 기존 시그니처 그대로 사용

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
- DeepSeek-R1은 <think>…</think> 사고흐름을 내보낼 수 있으므로,
  프롬프트에서 "최종 결과는 JSON/텍스트만"을 강하게 지시하고
  아래 extract_json_block()으로 파싱을 안정화
"""
llm = ChatOllama(
    model="deepseek-r1:latest",  # 설치된 태그에 맞게 조정 가능 (예: deepseek-r1:7b)
    temperature=0.7,
)


# ──────────────────────────────────────────
# 상태 정의
# ──────────────────────────────────────────

class State(TypedDict):
    messages: List[AnyMessage | str]
    task_history: List[Task]
    references: Dict[str, Any]  # {"queries": List[str], "docs": List[Document-like]}


# ──────────────────────────────────────────
# JSON 유틸: LLM 출력 → 안전 파싱
# ──────────────────────────────────────────

def extract_json_block(text: str) -> Dict[str, Any]:
    """
    DeepSeek-R1이 사고흐름을 포함할 수 있으므로,
    가장 그럴듯한 JSON 객체를 안전하게 추출.
    1) ```json ... ``` 우선
    2) 균형 잡힌 중괄호 탐색 휴리스틱
    실패 시 {} 반환
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

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
                    start_idx = -1
    return {}


def to_task(obj: Dict[str, Any]) -> Task:
    """
    JSON dict → Task(Pydantic)로 엄격 변환.
    필수: agent, description
    선택: done, done_at
    """
    agent = str(obj.get("agent", "")).strip() or "communicator"
    description = str(obj.get("description", "")).strip() or "사용자와 대화를 통해 다음 단계를 파악한다"
    done = bool(obj.get("done", False))
    done_at = str(obj.get("done_at", ""))

    return Task(agent=agent, description=description, done=done, done_at=done_at)


# ──────────────────────────────────────────
# Supervisor 에이전트
# ──────────────────────────────────────────

def supervisor(state: State):
    print("\n\n============ SUPERVISOR ============")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고,
        현재 해야할 일과 사용할 agent를 결정한다.

        사용 가능한 agent:
        - content_strategist: 요구사항이 명확할 때. 콘텐츠 전략/목차(outline) 작성·수정
        - communicator: 판단이 서지 않거나 사용자 피드백 필요할 때. 진행 보고 및 지시 요청
        - web_search_agent: 웹 검색으로 최신/배경 정보를 수집
        - vector_search_agent: 벡터 DB 검색으로 레퍼런스/아이디어 보강

        아래 정보를 참고하라.
        ------------------------------------------
        previous_outline:
        {outline}
        ------------------------------------------
        messages:
        {messages}
        ------------------------------------------

        출력 형식(반드시 이 JSON만 출력하라):
        ```json
        {{
          "agent": "content_strategist" | "communicator" | "vector_search_agent" | "web_search_agent",
          "description": "간단하고 명확한 할 일 설명",
          "done": false,
          "done_at": ""
        }}
        ```
        """
    )

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

    # references 키가 없을 수 있으므로 보장
    return {
        "messages": messages,
        "task_history": task_history,
        "references": state.get("references", {"queries": [], "docs": []}),
    }


# ──────────────────────────────────────────
# 라우터
# ──────────────────────────────────────────

def supervisor_router(state: State):
    task = state["task_history"][-1]
    return task.agent



# ──────────────────────────────────────────
# Vector Search Agent (OpenAI 함수 → JSON 계획 + 파이썬 직접 호출)
# ──────────────────────────────────────────

def vector_search_agent(state: State):
    print("\n\n============ VECTOR SEARCH AGENT ============")

    tasks = state.get("task_history", [])
    task = tasks[-1]
    if task.agent != "vector_search_agent":
        raise ValueError(f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}")

    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 목차(outline) 작성/개선에 필요한 정보를 벡터 검색으로 찾아내는 Agent다.
        다음 맥락을 바탕으로 검색 질의(query) 후보를 3~8개 생성하라.
        - 검색 목적(미션): {mission}
        - 과거 검색 내용(중복 피하기):
          {references}
        - 최근 대화:
          {messages}
        - 현재 목차(있다면 보완점 고려):
          {outline}

        출력 형식(반드시 이 JSON만 출력하라):
        ```json
        {{
          "queries": ["짧고 구체적인 한글/영문 질의", "...", "..."],
          "notes": "질의 의도/보완 포인트(선택)"
        }}
        ```
        """
    )

    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "mission": task.description,
        "references": references,
        "messages": messages,
        "outline": outline,
    }

    chain = vector_search_system_prompt | llm | StrOutputParser()
    raw = chain.invoke(inputs)
    data = extract_json_block(raw)

    new_queries: List[str] = [q for q in data.get("queries", []) if isinstance(q, str)]
    if not new_queries:
        # 폴백: 미션 문장을 그대로 한 개의 질의로 사용
        new_queries = [task.description]

    # 검색 수행 및 결과 누적
    for query in new_queries:
        try:
            print("Vector Query →", query)
            docs = retrieve({"query": query})  # ← 기존 tools.retrieve 사용
            references.setdefault("queries", []).append(query)
            references.setdefault("docs", []).extend(docs)
        except Exception as e:
            print(f"[retrieve 에러] {e}")

    # 문서 중복 제거(페이지 내용 기준)
    unique_docs = []
    seen = set()
    for doc in references.get("docs", []):
        content = getattr(doc, "page_content", None)
        if content and content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    references["docs"] = unique_docs

    # 로그 출력
    print('Queries:--------------------------')
    for q in references.get("queries", []):
        print(q)
    print('References:--------------------------')
    for doc in references.get("docs", [])[:10]:  # 너무 길면 상위 10개만 표시
        preview = getattr(doc, "page_content", "")[:120].replace("\n", " ")
        print(preview)
        print('--------------------------')

    # task 완료 처리 + 다음 task: communicator
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at=""
    )
    tasks.append(new_task)

    # 메시지 기록
    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {new_queries}"
    message = AIMessage(msg_str)
    messages.append(message)
    print(msg_str)

    return {
        "messages": messages,
        "task_history": tasks,
        "references": references
    }


# ──────────────────────────────────────────
# Web Search Agent (OpenAI ToolCalling → JSON 계획 + 파이썬 직접 호출)
# ──────────────────────────────────────────

def web_search_agent(state: State):
    print("\n\n============ WEB SEARCH AGENT ============")

    tasks = state.get("task_history", [])
    task = tasks[-1]

    if task.agent != "web_search_agent":
        raise ValueError(f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")

    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 웹 검색으로 목차(outline) 작성에 필요한 최신/배경 정보를 수집하는 Agent다.
        현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 3~8개의 검색 질의를 제안하라.
        가능한 한 최신성과 신뢰도를 고려한 키워딩을 사용하라.

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용(중복 피하기): {references}
        --------------------------------
        - 이전 대화: {messages}
        --------------------------------
        - 현재 목차: {outline}
        --------------------------------
        - 현재 시각: {current_time}

        출력 형식(반드시 이 JSON만 출력하라):
        ```json
        {{
          "queries": ["검색어1", "검색어2", "..."],
          "notes": "출처 우선순위/수집 기준 등(선택)"
        }}
        ```
        """
    )

    messages = state.get("messages", [])
    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": messages,
        "outline": get_outline(current_path),
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # LLM이 JSON 형태의 '검색 계획'만 출력하도록 유도
    chain = web_search_system_prompt | llm | StrOutputParser()
    raw = chain.invoke(inputs)
    data = extract_json_block(raw)

    queries: List[str] = [q for q in data.get("queries", []) if isinstance(q, str)]
    if not queries:
        # 폴백: 미션 문장 사용
        queries = [task.description]

    # 실제 검색 실행 및 색인
    executed_queries = []
    for q in queries:
        print('-------- web search --------', q)
        try:
            # tools.web_search는 (results, json_path) 등을 반환한다고 가정
            _, json_path = web_search.invoke({"query": q})
            print('json_path:', json_path)

            # 검색 결과 JSON을 Chroma에 추가
            add_web_pages_json_to_chroma(json_path)
            executed_queries.append(q)
        except Exception as e:
            print(f"[web_search/add_to_chroma 에러] {e}")

    # task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 새로운 task: Vector Search Agent로 전환 (방금 색인한 문서들 활용)
    task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색으로 보강한다."
    if executed_queries:
        task_desc += f" 새로 색인된 검색어: {executed_queries}"

    new_task = Task(
        agent="vector_search_agent",
        done=False,
        description=task_desc,
        done_at=""
    )
    tasks.append(new_task)

    # 작업 후기 메시지
    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {executed_queries}"
    messages.append(AIMessage(msg_str))
    print(msg_str)

    return {
        "messages": messages,
        "task_history": tasks,
        "references": state.get("references", {"queries": [], "docs": []}),  # references는 vector_search에서 갱신
    }


# ──────────────────────────────────────────
# 콘텐츠 전략가
# ──────────────────────────────────────────

def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)다.
        이전 대화와 기존 목차, 참고 자료를 바탕으로 사용자의 요구사항을 분석하여,
        책의 세부 목차를 작성하거나(없으면 생성) 기존 목차를 개선한다.

        출력은 '최종 목차 텍스트'만 산출하라. 불필요한 설명은 제외한다.

        --------------------------------
        - 지난 목차:
        {outline}
        --------------------------------
        - 이전 대화 내용:
        {messages}
        --------------------------------
        - 참고 자료(검색 질의/문서 요약 가능):
        {references}
        """
    )

    chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "messages": messages,
        "outline": outline,
        "references": state.get("references", {"queries": [], "docs": []}),
    }

    gathered = ""
    for chunk in chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")
    print()

    save_outline(current_path, gathered)

    # 메시지 및 상태 업데이트
    content_strategist_message = "[Content Strategist] 목차 작성 완료"
    messages.append(AIMessage(content=content_strategist_message))
    task_history = state.get("task_history", [])
    if task_history[-1].agent != "content_strategist":
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at=""
    )
    task_history.append(new_task)

    print(new_task)

    return {
        "messages": messages,
        "task_history": task_history,
        "references": state.get("references", {"queries": [], "docs": []}),
    }


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

    print("\nAI\t: ", end="")
    gathered: BaseMessage | None = None
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end="")
        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    print()

    if gathered is None:
        gathered = AIMessage("죄송합니다. 응답을 생성하지 못했습니다. 다시 시도해 주세요.")

    messages.append(gathered)

    task_history = state.get("task_history", [])
    if task_history[-1].agent != "communicator":
        raise ValueError(f"Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}")
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "messages": messages,
        "task_history": task_history,
        "references": state.get("references", {"queries": [], "docs": []}),
    }


# ──────────────────────────────────────────
# LangGraph 빌드
# ──────────────────────────────────────────

graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
        "vector_search_agent": "vector_search_agent",
        "web_search_agent": "web_search_agent",
    },
)
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# Mermaid PNG 저장(옵션)
try:
    graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))
except Exception:
    pass


# ──────────────────────────────────────────
# 상태 초기화
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
        references={"queries": [], "docs": []},
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
- 우측: 현재 저장된 outline / 최근 검색 질의 / 참조 문서 미리보기
- 하단: 사용자 입력 + 전송/초기화
"""

def render_chat(messages: List[BaseMessage | str]) -> str:
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
            role = msg.__class__.__name__.upper()
            content = getattr(msg, "content", str(msg))
        lines.append(f"**{role}:** {content}")
    return "\n\n".join(lines)


def render_references(refs: Dict[str, Any]) -> str:
    if not refs:
        return "(참조 없음)"
    queries = refs.get("queries", [])
    docs = refs.get("docs", [])
    lines = []
    if queries:
        lines.append("### 🔍 최근 검색 질의")
        for i, q in enumerate(queries[-10:], 1):
            lines.append(f"{i}. {q}")
    if docs:
        lines.append("\n### 📄 참조 문서 미리보기 (상위 8개)")
        for i, d in enumerate(docs[:8], 1):
            preview = getattr(d, "page_content", "")
            meta = getattr(d, "metadata", {})
            src = meta.get("source") or meta.get("url") or ""
            # ✅✅✅ 이 부분이 수정되었습니다 ✅✅✅
            clean_preview = preview[:160].replace('\n', ' ')
            line = f"**{i}.** {clean_preview}" + (f"\n— _{src}_" if src else "")
            lines.append(line)
    return "\n\n".join(lines) or "(참조 없음)"


def ui_on_send(user_text: str, state: State):
    try:
        if state is None:
            state = init_state()

        if user_text.strip():
            state["messages"].append(HumanMessage(user_text.strip()))
            new_state = graph.invoke(state)

            # 상태 저장
            save_state(current_path, new_state)

            chat_md = render_chat(new_state["messages"])
            outline_text = get_outline(current_path) or "(아직 목차가 없습니다.)"
            refs_md = render_references(new_state.get("references"))
            return chat_md, outline_text, refs_md, new_state
        else:
            chat_md = render_chat(state["messages"])
            outline_text = get_outline(current_path) or "(아직 목차가 없습니다.)"
            refs_md = render_references(state.get("references"))
            return chat_md, outline_text, refs_md, state
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"[에러] {e}\n\n{tb}"
        chat_md = render_chat(state["messages"] if state else [])
        return chat_md + "\n\n" + err_msg, get_outline(current_path), "(참조 없음)", state


def ui_on_reset():
    state = init_state()
    return render_chat(state["messages"]), get_outline(current_path) or "(아직 목차가 없습니다.)", "(참조 없음)", state


def launch_gradio():
    with gr.Blocks(title="AI Book Team (Ollama + DeepSeek-R1)") as demo:
        gr.Markdown("## ✍️ AI Book Team (Local Ollama · DeepSeek-R1)\n오른쪽에 최신 목차와 검색 레퍼런스가 표시됩니다.")
        with gr.Row():
            with gr.Column(scale=2):
                chat_md = gr.Markdown(value="")
                user_in = gr.Textbox(label="메시지 입력", placeholder="예) 'AI로 책을 쓰고 싶어요. 주제는 ...'")
                with gr.Row():
                    send_btn = gr.Button("보내기", variant="primary")
                    reset_btn = gr.Button("대화 초기화", variant="secondary")
            with gr.Column(scale=1):
                outline_md = gr.Markdown(label="현재 목차", value="(아직 목차가 없습니다.)")
                refs_md = gr.Markdown(label="검색 레퍼런스", value="(참조 없음)")

        state_box = gr.State(init_state())

        send_btn.click(
            fn=ui_on_send,
            inputs=[user_in, state_box],
            outputs=[chat_md, outline_md, refs_md, state_box],
        )
        user_in.submit(
            fn=ui_on_send,
            inputs=[user_in, state_box],
            outputs=[chat_md, outline_md, refs_md, state_box],
        )
        reset_btn.click(
            fn=ui_on_reset,
            inputs=[],
            outputs=[chat_md, outline_md, refs_md, state_box],
        )

    demo.launch()


# ──────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────

if __name__ == "__main__":
    # 기본: Gradio GUI 실행 (CLI가 필요하면 run_cli() 호출로 교체)
    launch_gradio()
    # run_cli()