"""
[OpenCode 변환본]

- 목적: 기존 Closed LLM(OpenAI ChatOpenAI, gpt-4o) 의존 코드를
        로컬 오픈소스 LLM(DeepSeek-R1, Ollama) 기반으로 동작하도록 변환.
- 핵심 변경:
  1) ChatOpenAI → ChatOllama(model="deepseek-r1") 로 교체 (API Key 불필요, 로컬 추론)
  2) .bind_tools / .with_structured_output 등 OpenAI 전용 기능 의존 제거
     → PydanticOutputParser(JSON 강제) + "검색 계획 JSON" 출력으로 치환
  3) CLI while-loop → Gradio GUI 인터페이스(로컬 실행)
- 전제: PC에 Ollama 및 deepseek-r1 모델이 설치되어 있음.
  * 설치 예시:
    - Ollama: https://ollama.com
    - 모델: `ollama pull deepseek-r1`
- 주의: 본 스크립트는 기존 프로젝트 구조(utils, models, tools 폴더 등)를 그대로 활용합니다.
        (save_state, get_outline, save_outline, Task, retrieve, web_search, add_web_pages_json_to_chroma 존재 가정)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- LangChain / LangGraph ---
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOllama          # ✅ Ollama 기반 Chat 모델
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser

# Pydantic v1 (LangChain 호환)
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError

# --- 프로젝트 의존(기존 코드 그대로 사용) ---
from typing_extensions import TypedDict
from utils import save_state, get_outline, save_outline
from models import Task  # 기존 Pydantic 모델 가정 (fields: agent:str, description:str, done:bool=False, done_at:str="")
from tools import retrieve, web_search, add_web_pages_json_to_chroma

# --- GUI ---
import gradio as gr


# =========================
# 경로/환경 기초 설정
# =========================
filename = os.path.basename(__file__)
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)


# =========================
# LLM 생성기 (Ollama / DeepSeek-R1)
# =========================
def create_llm(
    model_name: str = "deepseek-r1",  # 로컬에 설치된 모델 태그 사용
    temperature: float = 0.3,
    num_ctx: int = 8192,
) -> ChatOllama:
    """
    로컬 Ollama 서버를 이용하는 LangChain Chat 모델을 생성한다.
    - API Key 불필요
    - 모델은 사전 설치 필요: `ollama pull deepseek-r1`
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        # 필요한 경우 추가 파라미터: base_url="http://localhost:11434"
    )


# 전역 LLM 인스턴스
llm = create_llm()


# =========================
# 상태 정의 (원 코드 유지)
# =========================
class State(TypedDict):
    messages: List[AnyMessage | str]
    task_history: List[Task]
    references: Dict[str, Any]
    user_request: str  # 사용자의 요구사항


# =========================
# 공통 유틸: JSON 파서(안전)
# =========================
def safe_json_extract(text: str, fallback: Any) -> Any:
    """
    모델 출력이 JSON이 아닐 경우를 대비한 안전 파서.
    - LangChain JsonOutputParser를 우선 사용, 실패 시 수동 정리.
    """
    parser = JsonOutputParser()
    try:
        return parser.parse(text)
    except Exception:
        # 흔한 패턴: 코드블록 포함, 전/후 설명문 포함 등
        import json, re

        # 코드블록 제거
        cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.S)
        # 첫 { ... } 또는 [ ... ] 블록만 추출 시도
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return fallback
        return fallback


# =========================
# Business Analyst
# =========================
def business_analyst(state: State):
    print("\n\n============ BUSINESS ANALYST ============")

    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 비즈니스 애널리스트로서, 
        AI팀의 진행상황과 "사용자 요구사항"을 토대로,
        현 시점에서 '지난 요청사항 (previous_user_request)'과 최근 사용자의 발언을 바탕으로 요구사항이 무엇인지 판단한다.
        지난 요청사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.

        다음과 같은 템플릿 형태로 반환한다. 
        ```
        - 목표: OOOO \n 방법: OOOO
        ```

        ------------------------------------
        *지난 요청사항(previous_user_request)* : {previous_user_request}
        ------------------------------------
        사용자 최근 발언: {user_last_comment}
        ------------------------------------
        참고자료: {references}
        ------------------------------------
        목차 (outline): {outline}
        ------------------------------------
        "messages": {messages}
        """
    )

    ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

    messages = state["messages"]

    # 최근 사용자 발언 추출
    user_last_comment = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    inputs = {
        "previous_user_request": state.get("user_request", None),
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline": get_outline(current_path),
        "messages": messages,
        "user_last_comment": user_last_comment,
    }

    user_request = ba_chain.invoke(inputs)

    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message)
    messages.append(AIMessage(business_analyst_message))

    save_state(current_path, state)

    return {
        "messages": messages,
        "user_request": user_request,
    }


# =========================
# Supervisor (구조화 출력: PydanticOutputParser)
# =========================
def supervisor(state: State):
    print("\n\n============ SUPERVISOR ============")

    # 기존 Task Pydantic 모델을 이용해 파서 생성
    parser = PydanticOutputParser(pydantic_object=Task)

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고, 
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.     
        - content_strategist: 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다. 
        - web_search_agent: vector_search_agent를 시도하고, 검색 결과(references)에 필요한 정보가 부족한 경우 사용한다. 웹 검색을 통해 해당 정보를 Vector DB에 보강한다. 
        - vector_search_agent: 목차 작성을 위해 필요한 자료를 확보하기 위해 벡터 DB 검색을 한다. 

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.
        반드시 JSON 형식으로 아래 스키마를 엄격히 따를 것.
        {format_instructions}

        ------------------------------------------
        previous_outline: {outline}
        ------------------------------------------
        messages:
        {messages}
        """
    ).partial(format_instructions=parser.get_format_instructions())

    prompt = supervisor_system_prompt.format(
        outline=get_outline(current_path),
        messages=state.get("messages", []),
    )

    # 모델 호출
    raw = llm.invoke(prompt).content

    # 파싱(안전)
    try:
        task: Task = parser.parse(raw)
    except ValidationError:
        data = safe_json_extract(raw, fallback={"agent": "communicator", "description": "사용자에게 다음 지시 문의", "done": False, "done_at": ""})
        task = Task(**data)

    task_history = state.get("task_history", [])
    task_history.append(task)

    supervisor_message = AIMessage(f"[Supervisor] {task}")
    state["messages"].append(supervisor_message)
    print(supervisor_message.content)

    return {
        "messages": state["messages"],
        "task_history": task_history,
    }


def supervisor_router(state: State):
    task = state["task_history"][-1]
    return task.agent


# =========================
# Vector Search Agent (도구 호출 → JSON 질의계획으로 치환)
# =========================
def vector_search_agent(state: State):
    print("\n\n============ VECTOR SEARCH AGENT ============")

    tasks = state.get("task_history", [])
    task = tasks[-1]
    if task.agent != "vector_search_agent":
        raise ValueError(
            f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}"
        )

    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 목차(outline) 작성에 필요한 정보를 벡터 검색으로 보완하는 Agent이다.

        아래 정보를 토대로, 지금 필요한 "벡터 검색 질의(query)"를 1개 이상 제안하라.
        반드시 JSON 배열 형식만 출력한다. 예: ["질문1", "질문2", ...]

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        """
    )

    mission = task.description
    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "mission": mission,
        "references": references,
        "messages": messages,
        "outline": outline,
    }

    plan_prompt = vector_search_system_prompt.format(**inputs)
    raw_plan = llm.invoke(plan_prompt).content
    queries: List[str] = safe_json_extract(raw_plan, fallback=[])

    # 실제 벡터 검색 수행
    for q in queries:
        print("VECTOR QUERY:", q)
        # retrieve() 는 프로젝트 제공 툴 가정 (args: {"query": "..."} 형태)
        retrieved_docs = retrieve({"query": q})
        references["queries"].append(q)
        references["docs"] += retrieved_docs

    # 중복 제거
    unique_docs = []
    seen = set()
    for doc in references["docs"]:
        sig = getattr(doc, "page_content", None)
        if sig and sig not in seen:
            unique_docs.append(doc)
            seen.add(sig)
    references["docs"] = unique_docs

    # 로그 출력
    print('Queries:--------------------------')
    for q in references["queries"]:
        print(q)
    print('References:--------------------------')
    for doc in references["docs"]:
        print(getattr(doc, "page_content", "")[:100])
        print('--------------------------')

    # task 완료 처리
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 다음 task: communicator
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at="",
    )
    tasks.append(new_task)

    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))
    print(msg_str)

    return {
        "messages": messages,
        "task_history": tasks,
        "references": references,
    }


# =========================
# Content Strategist
# =========================
def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    task_history = state.get("task_history", [])
    task = task_history[-1]
    if task.agent != "content_strategist":
        raise ValueError(
            f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}"
        )

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화/요구를 분석하여 세부 목차를 결정한다.

        다음 정보를 활용하여 목차를 작성하라. 
        - 사용자 요구사항(user_request)
        - 작업(task)
        - 검색 자료 (references)
        - 기존 목차 (previous_outline)
        - 이전 대화 내용(messages)

        목표:
        1) 기존 목차가 있으면 수정/보완, 없으면 신규 제안.
        2) chapter/section/sub-section 구성, sub-section 아래 주요 bullet 포함.
        3) 논리적 흐름 점검, 사용자 요구 반영.
        4) references 기반 참고문헌 정리(가능한 한 풍부하게, URL 전체 표기).
        5) 추가 리서치 필요 항목은 supervisor에게 요청.

        출력은 outline_template 양식에 맞추되, 필요 시 섹션을 자유롭게 확장하라.
        각 장은 ':---CHAPTER DIVIDER---:' 로 구분.

        --------------------------------
        - 사용자 요구사항(user_request): 
        {user_request}
        --------------------------------
        - 작업(task): 
        {task}
        --------------------------------
        - 참고 자료 (references)
        {references}
        --------------------------------
        - 기존 목차 (previous_outline)
        {outline}
        --------------------------------
        - 이전 대화 내용(messages)
        {messages}
        --------------------------------

        outline_template:
        {outline_template}

        사용자가 추가 피드백을 제공할 수 있도록 논리 흐름과 주요 아이디어를 제안하라.
        마지막에는 '-----: DONE :-----'를 넣고 간단한 작업 후기를 덧붙여라.
        """
    )

    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    user_request = state.get("user_request", "")
    messages = state["messages"]
    outline = get_outline(current_path)

    # 템플릿 로딩 (프로젝트의 templates/outline_template.md 존재 가정)
    with open(f"{current_path}/templates/outline_template.md", "r", encoding="utf-8") as f:
        outline_template = f.read()

    inputs = {
        "user_request": user_request,
        "task": task,
        "messages": messages,
        "outline": outline,
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline_template": outline_template,
    }

    gathered = ""
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")
    print()

    save_outline(current_path, gathered)

    # 후기 부분 추출
    if "-----: DONE :-----" in gathered:
        review = gathered.split("-----: DONE :-----")[-1]
    else:
        review = gathered[-200:]

    content_strategist_message = f"[Content Strategist] 목차 작성 완료: outline 작성 완료\n {review}"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "messages": messages,
        "task_history": task_history,
    }


# =========================
# Web Search Agent (도구 호출 → JSON 질의계획으로 치환)
# =========================
def web_search_agent(state: State):
    print("\n\n============ WEB SEARCH AGENT ============")

    tasks = state.get("task_history", [])
    task = tasks[-1]
    if task.agent != "web_search_agent":
        raise ValueError(
            f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}"
        )

    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 목차(outline) 작성에 필요한 정보를 웹 검색을 통해 보완하는 Agent이다.
        부족한 정보는 복합 질문을 나누어 검색한다.

        지금 필요한 "웹 검색 질의(query)"들을 1개 이상 제안하라.
        반드시 JSON 배열 형식만 출력한다. 예: ["질문1", "질문2", ...]

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        --------------------------------
        - 현재 시각 : {current_time}
        """
    )

    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": state.get("messages", []),
        "outline": get_outline(current_path),
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    plan_prompt = web_search_system_prompt.format(**inputs)
    raw_plan = llm.invoke(plan_prompt).content
    queries: List[str] = safe_json_extract(raw_plan, fallback=[])

    # 실제 웹 검색 수행 및 Chroma 적재
    for q in queries:
        print("WEB QUERY:", q)
        # web_search(args) → (results, json_path) 를 반환한다고 원 코드에서 가정
        _, json_path = web_search.invoke({"query": q})
        print("json_path:", json_path)
        add_web_pages_json_to_chroma(json_path)

    # task 완료 처리
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 다음 task: vector_search_agent
    task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    task_desc += f" 다음 항목이 새로 추가되었다\n: {queries}"

    new_task = Task(agent="vector_search_agent", done=False, description=task_desc, done_at="")
    tasks.append(new_task)

    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    state["messages"].append(AIMessage(msg_str))

    return {
        "messages": state["messages"],
        "task_history": tasks,
    }


# =========================
# Communicator
# =========================
def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.
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

    gathered = None
    print("\nAI\t: ", end="")
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end="")
        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    print()

    messages.append(gathered)

    task_history = state.get("task_history", [])
    if task_history[-1].agent != "communicator":
        raise ValueError(
            f"Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}"
        )
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "messages": messages,
        "task_history": task_history,
    }


# =========================
# LangGraph 정의 (원 코드 유지)
# =========================
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("business_analyst", business_analyst)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

# Edges
graph_builder.add_edge(START, "business_analyst")
graph_builder.add_edge("business_analyst", "supervisor")
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
graph_builder.add_edge("content_strategist", "business_analyst")
graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "business_analyst")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# Mermaid PNG 출력(선택)
try:
    graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace(".py", ".png"))
except Exception as e:
    print("[경고] 그래프 이미지 생성 실패(무시 가능):", e)


# =========================
# 초기 상태 생성 함수
# =========================
def make_initial_state() -> State:
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
        user_request="",
    )


# =========================
# Gradio 인터페이스
# =========================
def run_pipeline(user_input: str, state: Optional[State]):
    """
    Gradio 핸들러:
    - 대화 1턴을 실행
    - LangGraph 파이프라인 호출
    - 결과 메시지(모델 응답)와 갱신된 상태 반환
    """
    if state is None:
        state = make_initial_state()

    # 종료 명령 처리
    if user_input.strip().lower() in ["exit", "quit", "q"]:
        return "Goodbye!", state

    # 사용자 메시지 추가 및 그래프 실행
    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    # 최근 AI 메시지의 텍스트만 추려서 Chat UI에 표시
    # (여러 Agent 메시지가 추가되므로 마지막 Human 이후의 AI들을 모아 간단 합친다)
    ai_responses = []
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            content = getattr(m, "content", None)
            if content:
                ai_responses.append(content)
            # 충분히 모았으면 종료
            if len(ai_responses) >= 3:
                break

    # 상태 저장(옵션)
    try:
        save_state(current_path, state)
    except Exception as e:
        print("[경고] 상태 저장 실패(무시 가능):", e)

    # 최근 응답들을 위에서부터 순서대로 합치기
    ai_responses = list(reversed(ai_responses))
    display_text = "\n\n".join(ai_responses) if ai_responses else "(응답 생성됨)"

    return display_text, state


def build_ui():
    with gr.Blocks(title="OpenCode • DeepSeek-R1 (Ollama) Book-Team Agent") as demo:
        gr.Markdown(
            """
            # OpenCode • 로컬 LLM(DeepSeek-R1 on Ollama) 기반 Writer-Team Agent
            - ChatOpenAI → ChatOllama 대체 (로컬 추론, API Key 불필요)
            - LangGraph 파이프라인 + Vector/Web Search Agent
            - 좌측 입력에 요청을 적고 전송하세요.
            """
        )
        state = gr.State(make_initial_state())

        with gr.Row():
            chat = gr.Chatbot(height=420, label="대화")
        with gr.Row():
            txt = gr.Textbox(placeholder="요청을 입력하세요. (예: 'AI 입문서를 위한 목차를 잡아줘')", lines=3)
        with gr.Row():
            send = gr.Button("전송", variant="primary")
            clear = gr.Button("초기화")
        with gr.Accordion("진행 상태(디버그)", open=False):
            ref_view = gr.JSON(label="references")
            task_view = gr.JSON(label="task_history")

        def on_send(message, _state):
            reply, new_state = run_pipeline(message, _state)
            # Chatbot 기록 업데이트
            history_add = [(message, reply)]
            # 디버그 뷰
            refs = new_state.get("references", {})
            tasks = [t.dict() if hasattr(t, "dict") else dict(t) for t in new_state.get("task_history", [])]
            return history_add, new_state, refs, tasks, ""

        send.click(
            on_send,
            inputs=[txt, state],
            outputs=[chat, state, ref_view, task_view, txt],
        )

        def on_clear():
            new_state = make_initial_state()
            return [], new_state, {"queries": [], "docs": []}, [], ""

        clear.click(
            on_clear,
            inputs=[],
            outputs=[chat, state, ref_view, task_view, txt],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    # 로컬에서만 접근하는 기본 설정 (필요 시 share=True)
    ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


