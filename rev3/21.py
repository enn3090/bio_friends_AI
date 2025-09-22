"""
[OpenCode 변환본]

목표: Streamlit + OpenAI(ChatOpenAI) + LangChain Tools 기반 코드를
→ **로컬 오픈소스 LLM (Ollama · DeepSeek-R1)** + **Gradio UI** + **수동 Tool-Calling 프로토콜**로 변환

핵심 변경점
1) OpenAI(Closed LLM) 제거 → **Ollama 로컬 모델(DeepSeek-R1)** 사용 (http://localhost:11434)
2) Streamlit UI → **Gradio ChatInterface**
3) LangChain의 tool 바인딩 대신, **JSON 기반 수동 툴 호출 프로토콜** 채택
   - 모델에게 다음 형식만 출력하도록 지시:
     {"tool": "<tool_name>", "args": {...}}  또는  {"tool": null, "answer": "..."}
   - (툴 호출 시) 파이썬에서 실제 함수 실행 → 결과를 시스템 메시지로 주입 → 최종 답변 재생성
4) **스트리밍 출력**은 응답 2단계(최종 답변)에서 가볍게 지원 가능하나,
   DeepSeek-R1 특성상 내부 사고(<think>...</think>)가 섞일 수 있어 **자동 제거** 처리

필요 패키지
pip install gradio langchain langchain-community duckduckgo-search pytz youtube-search

사전 준비
- Ollama 설치: https://ollama.com
- 모델 받기: `ollama pull deepseek-r1:latest`
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import gradio as gr
import pytz

# LangChain 메시지 타입 (대화 기록 관리를 위해 그대로 사용)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 로컬 LLM: Ollama용 LangChain Chat 모델
from langchain_community.chat_models import ChatOllama

# DuckDuckGo 검색
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# YouTube 검색/로더
from youtube_search import YoutubeSearch
from langchain_community.document_loaders import YoutubeLoader


# =========================
# 설정
# =========================
MODEL_NAME = "deepseek-r1:latest"   # 필요시 "deepseek-r1:7b" / "deepseek-r1:14b"
TEMPERATURE = 0.7

SYSTEM_PROMPT = """너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다.

[툴 사용 규칙]
- 네가 외부 도구가 필요하다고 판단하면 **반드시 JSON 한 줄**로만 응답해.
- JSON 스키마:
  {"tool": "<tool_name>", "args": {"key": "value", ...}}
  - 사용 가능 도구:
    1) get_current_time(timezone: str, location: str) -> str
    2) get_web_search(query: str, search_period: str) -> str
    3) get_youtube_search(query: str) -> str
- 도구가 필요 없으면:
  {"tool": null, "answer": "<사용자에게 보여줄 최종 답변>"}
- JSON 외 텍스트/마크다운을 섞지 말 것.

[응답 스타일]
- 한국어로 간결하고 정확하게 답변.
- 확실하지 않으면 불확실함을 명시.
"""

# DeepSeek-R1 내부 사고(<think>...</think>) 제거 패턴
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_TAG_RE.sub("", text).strip()


def extract_json(text: str) -> Dict[str, Any] | None:
    """모델 출력에서 JSON 오브젝트를 추출/파싱."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# =========================
# 도구 구현
# =========================
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각 문자열 반환."""
    try:
        tz = pytz.timezone(timezone)
    except pytz.UnknownTimeZoneError:
        return f"[오류] 알 수 없는 타임존: {timezone}"
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return f"{timezone} ({location}) 현재시각 {now}"


def get_web_search(query: str, search_period: str) -> str:
    """
    DuckDuckGo 웹/뉴스 검색.
    - search_period: 'd'(하루), 'w'(1주), 'm'(1달), 'y'(1년) 등
    반환: DuckDuckGoSearchResults가 생성한 문자열(각 항목은 ';\n'로 구분, 'link:' 포함)
    """
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, results_separator=";\n")
    docs = search.invoke(query)
    return docs


def get_youtube_search(query: str) -> str:
    """
    유튜브 검색 → 상위 5개에서 길이 제한(대략 'mm:ss' 길이 문자열 <= 5) 통과만 채택.
    각 영상의 URL과 로더로 수집한 자막(가능 시) 앞부분을 요약해 텍스트로 반환.
    반환: 사람이 읽기 쉬운 텍스트(최대 수천자 내)
    """
    videos = YoutubeSearch(query, max_results=5).to_dict()

    # 'mm:ss' 또는 'H:MM:SS' 형태의 문자열 길이를 간단 필터(<= 5 → 최대 59:59 가정)
    videos = [v for v in videos if len(v.get("duration", "")) <= 5]

    lines = ["[YouTube 검색 결과 요약]"]
    for v in videos:
        title = v.get("title", "").strip()
        url = "http://youtube.com" + v.get("url_suffix", "")
        channel = v.get("channel", "")
        duration = v.get("duration", "")
        # 자막 로드 시도
        preview = ""
        try:
            loader = YoutubeLoader.from_youtube_url(url, language=["ko", "en"])
            docs = loader.load()
            if docs:
                text = (docs[0].page_content or "").strip().replace("\n", " ")
                preview = text[:800] + ("..." if len(text) > 800 else "")
        except Exception:
            preview = "(자막 로드에 실패했거나 자막이 없습니다.)"

        lines.append(f"- 제목: {title}\n  채널: {channel} · 길이: {duration}\n  URL: {url}\n  미리보기: {preview}")
    return "\n".join(lines) if len(lines) > 1 else "검색 결과가 충분하지 않습니다."


TOOLS = {
    "get_current_time": get_current_time,
    "get_web_search": get_web_search,
    "get_youtube_search": get_youtube_search,
}


# =========================
# 로컬 LLM (Ollama)
# =========================
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    # base_url="http://localhost:11434",  # 기본값 사용 시 생략
)


# =========================
# Gradio 히스토리 ↔ LangChain 메시지 변환
# =========================
def history_to_messages(history: List[Tuple[str, str]]) -> List:
    msgs: List = [SystemMessage(SYSTEM_PROMPT)]
    for u, a in history:
        if u:
            msgs.append(HumanMessage(u))
        if a:
            msgs.append(AIMessage(a))
    return msgs


# =========================
# 메인 응답 루프 (툴 호출 처리)
# =========================
def tool_loop(messages: List) -> str:
    """
    1) 모델에게 JSON-only 형식으로 응답 요구
    2) {"tool": "<name>", "args": {...}} 이면 해당 도구 실행 → 결과를 시스템 컨텍스트로 주입
    3) 최종 답변은 {"tool": null, "answer": "..."} 형태로 받아 출력
    - 안전장치로 최대 3회까지 연속 툴 호출 허용
    """
    max_calls = 3
    local_messages = messages[:]
    for _ in range(max_calls):
        first = llm.invoke(local_messages)
        first_text = getattr(first, "content", str(first))
        parsed = extract_json(first_text)

        # 형식 위반 시, 텍스트 전체를 최종답변으로 사용
        if not parsed:
            return strip_think(first_text)

        # 도구 없이 최종 답
        if parsed.get("tool") is None:
            answer = parsed.get("answer", "")
            return strip_think(answer)

        # 도구 호출 처리
        tool_name = parsed.get("tool")
        args = parsed.get("args", {}) or {}
        tool_fn = TOOLS.get(tool_name)
        if not tool_fn:
            # 알 수 없는 도구면 그대로 설명 유도
            local_messages.append(AIMessage(first_text))
            local_messages.append(SystemMessage(f"알 수 없는 도구 `{tool_name}` 입니다. 도구 없이 답변을 작성하세요."))
            continue

        try:
            result = tool_fn(**args)
        except Exception as e:
            result = f"[도구 실행 오류] {e}"

        # 도구 결과를 모델에 전달하고 **자연어 최종 답변**을 요청
        local_messages.append(AIMessage(first_text))  # 모델이 낸 JSON을 히스토리에 보존
        local_messages.append(SystemMessage(
            "아래는 방금 실행된 도구의 결과다. 이 정보를 바탕으로 한국어로 최종 답변만 작성하라. "
            "JSON 형식은 사용하지 말 것.\n\n"
            f"[{tool_name} 결과]\n{result}"
        ))

        # 다음 루프로 재시도 → 대부분 이 시점에서 {"tool": null, ...} 또는 직접 자연어가 나옴
    # 반복 초과 시 마지막 출력 사용
    fallback = llm.invoke(local_messages)
    return strip_think(getattr(fallback, "content", str(fallback)))


def respond(user_input: str, history: List[Tuple[str, str]]) -> str:
    """
    Gradio ChatInterface 콜백:
    - 히스토리 + 사용자 입력으로 메시지 구성
    - tool_loop로 툴 호출/최종 답변 처리
    - DeepSeek-R1의 <think> 제거 후 반환
    """
    messages = history_to_messages(history)
    messages.append(HumanMessage(user_input))
    answer = tool_loop(messages)
    return answer


# =========================
# Gradio UI
# =========================
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 🤖🔧 Tool-Calling 챗봇 (DeepSeek-R1 · Ollama · Gradio)")

    chat = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            label="AI 도우미",
            height=520,
        ),
        title="",
        description=(
            "로컬 **Ollama(DeepSeek-R1)** + JSON 기반 **수동 Tool-Calling** 예시입니다.\n"
            "사용 가능 도구: `get_current_time(timezone, location)`, `get_web_search(query, search_period)`, `get_youtube_search(query)`\n"
            "※ 모델의 내부 사고(<think>...</think>)는 자동으로 숨겨집니다."
        ),
        theme="soft",
        retry_btn="다시 생성",
        undo_btn="이전 메시지 삭제",
        clear_btn="대화 초기화",
        examples=[
            "Asia/Seoul (서울) 현재 시간 알려줘.",
            "지난 한 달간(검색기간 m) 현대차 미국 시장 관련 뉴스를 요약해줘.",
            "로제 신곡 반응 관련 유튜브 영상 찾아서 핵심만 요약해줘.",
        ],
    )

    with gr.Accordion("시스템/모델/도구 정보", open=False):
        gr.Markdown(
            f"""
- 사용 모델: **{MODEL_NAME}**  
- Temperature: **{TEMPERATURE}**  
- 로컬 서버: **http://localhost:11434** (Ollama 기본)  
- 시스템 프롬프트 요약: JSON-only 도구 호출 프로토콜, 도구 미사용 시 자연어 최종답변  
- 제공 도구:
  1) `get_current_time(timezone, location)` — 예: timezone="Asia/Seoul", location="서울"
  2) `get_web_search(query, search_period)` — search_period: 'd'|'w'|'m'|'y' 등
  3) `get_youtube_search(query)` — 상위 결과에서 자막 미리보기 추출
"""
        )

if __name__ == "__main__":
    # 내부망만 사용할 경우 share=False 권장
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
