"""
[OpenCode 변환본]

목표: OpenAI + Streamlit 기반 검색/RAG 샘플 코드를
→ **로컬 오픈소스 LLM (Ollama: DeepSeek-R1)** + **Gradio UI** + **DuckDuckGo 뉴스 검색** + **웹스크래핑(WebBaseLoader/bs4)** 로 변환

핵심 포인트
1) ChatOpenAI → ChatOllama (로컬 Ollama, API Key 불필요)
2) Streamlit → Gradio ChatInterface (간단한 UI, 비동기 지원)
3) DuckDuckGo 검색(`langchain_community.tools.DuckDuckGoSearchResults`)로 뉴스 위주 검색
4) 검색 결과 링크를 파싱하여 **WebBaseLoader.aload()**(비동기)로 본문 수집 + bs4 백업 파서
5) 수집 문서들을 **context**로 넣어 LLM에 질의(간단한 Stuff-RAG)
6) DeepSeek-R1의 내부 사고(<think>...</think>)는 사용자 출력에서 제거

사전 준비
- Ollama 설치 & 모델 준비:
    - https://ollama.com
    - `ollama pull deepseek-r1:latest`
- 파이썬 패키지 설치:
    pip install gradio langchain langchain-community duckduckgo-search chromadb beautifulsoup4 requests
"""

from __future__ import annotations

import asyncio
import re
from typing import List, Tuple

import gradio as gr

# LangChain 메시지/프롬프트/파서
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 로컬 Ollama 모델
from langchain_community.chat_models import ChatOllama

# DuckDuckGo 검색
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# 웹 문서 로더(뉴스 본문 수집)
from langchain_community.document_loaders import WebBaseLoader

# ========= 설정 =========
MODEL_NAME = "deepseek-r1:latest"        # 필요 시 "deepseek-r1:7b" / "deepseek-r1:14b"
TEMPERATURE = 0.7
SYSTEM_PROMPT = "너는 문서(뉴스/웹페이지)에 기반해 한국어로 답변하는 산업·자동차 정책/시장 분석 전문가야."
DDG_REGION = "kr-kr"  # 한국 결과 우선
DDG_TIME = "w"        # 최근 1주 (d=1일, m=1달 등)
DDG_SOURCE = "news"   # 뉴스에 한정

# DeepSeek-R1 내부 사고 제거
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_TAG_RE.sub("", text).strip()


# ========= LLM 준비 =========
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    # base_url="http://localhost:11434",  # 기본값 사용 시 생략 가능
)


# ========= 프롬프트 & 체인 =========
# 간단 Stuff-RAG: context(뉴스/웹 본문) + 대화 메시지를 합쳐 답변
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "아래는 최신 뉴스/웹에서 수집한 컨텍스트다. 반드시 이 컨텍스트에 근거하여 한국어로 답변하라. "
            "불확실하면 추정을 줄이고 '불확실함'을 명시하라.\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_chain = question_answering_prompt | llm | StrOutputParser()


# ========= DuckDuckGo 검색 유틸 =========
def make_news_search_tool() -> DuckDuckGoSearchResults:
    wrapper = DuckDuckGoSearchAPIWrapper(region=DDG_REGION, time=DDG_TIME)
    return DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        source=DDG_SOURCE,
        results_separator=";\n",
    )


def parse_links_from_search_output(docs_str: str) -> List[str]:
    """
    DuckDuckGoSearchResults.invoke 의 문자열 결과에서 'link:' 이후 부분을 추출.
    각 항목은 '...;\\n' 로 구분됨.
    """
    links: List[str] = []
    if not docs_str:
        return links
    for item in docs_str.split(";\n"):
        item = item.strip()
        if not item:
            continue
        if "link:" in item:
            # 'link:' 이후를 URL로 간주
            try:
                link = item.split("link:")[1].strip()
                if link:
                    links.append(link)
            except Exception:
                continue
    # 중복 제거
    return list(dict.fromkeys(links))


# ========= 웹 본문 수집 (우선 WebBaseLoader, 실패 시 bs4 백업) =========
async def load_pages_async(urls: List[str]) -> List[str]:
    """
    WebBaseLoader.aload()로 비동기 수집. 실패하면 requests+bs4 로 백업 파싱.
    """
    texts: List[str] = []

    if not urls:
        return texts

    try:
        loader = WebBaseLoader(
            web_paths=urls,
            bs_get_text_kwargs={"strip": True},
        )
        # LangChain 비동기 로딩
        docs = await loader.aload()
        for d in docs:
            content = (d.page_content or "").strip()
            if content:
                texts.append(content)
    except Exception:
        # 일부 URL은 로더가 실패할 수 있으므로 bs4 백업 수행
        texts.extend(await fallback_fetch_many(urls))

    # 너무 긴 컨텍스트는 요약/절단 (간단히 앞부분만 사용)
    MAX_CHARS = 12000
    joined = "\n\n---\n\n".join(texts)
    if len(joined) > MAX_CHARS:
        joined = joined[:MAX_CHARS] + "\n\n[컨텍스트가 길어 일부만 사용됨]"
    return [joined]


async def fallback_fetch_many(urls: List[str]) -> List[str]:
    """
    requests + bs4 로 순차 수집 (간단 백업). 비동기 인터페이스에 맞추기 위해 asyncio.to_thread 사용.
    """
    import requests
    from bs4 import BeautifulSoup

    def fetch_one(url: str) -> str:
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            # 우선 article 태그 → 없으면 본문으로 추정되는 큰 div
            art = soup.find("article")
            if art:
                return art.get_text(" ", strip=True)
            # 일부 언론사 커스텀 컨테이너 대응
            cand_ids = ["CmAdContent", "news-contents", "dic_area", "articeBody"]
            for cid in cand_ids:
                node = soup.find("div", id=cid)
                if node:
                    return node.get_text(" ", strip=True)
            # 전체 텍스트 (간단 폴백)
            return soup.get_text(" ", strip=True)
        except Exception:
            return ""

    texts: List[str] = []
    for url in urls:
        txt = await asyncio.to_thread(fetch_one, url)
        if txt:
            texts.append(txt)
    return texts


# ========= Gradio <-> LangChain 히스토리 변환 =========
def history_to_messages(history: List[Tuple[str, str]]) -> List:
    msgs: List = [SystemMessage(SYSTEM_PROMPT)]
    for u, a in history:
        if u:
            msgs.append(HumanMessage(u))
        if a:
            msgs.append(AIMessage(a))
    return msgs


# ========= 메인 응답(비동기) =========
async def respond(user_input: str, history: List[Tuple[str, str]]):
    """
    1) DuckDuckGo 뉴스 검색 → 링크 추출
    2) 링크 본문 수집(WebBaseLoader → bs4 백업)
    3) 컨텍스트와 대화 기록으로 LLM 호출
    4) <think> 제거 후 반환
    """
    # 1) 뉴스 검색
    search_tool = make_news_search_tool()
    search_query = user_input.strip()
    docs_str = search_tool.invoke(search_query)  # 문자열 반환
    links = parse_links_from_search_output(docs_str)

    # 우측 패널(검색 요약) 표시용
    side_info = [
        "### 🔎 검색 요약",
        f"- 쿼리: `{search_query}`",
        f"- 링크 수: **{len(links)}**",
    ]
    if links:
        side_info.append("- 상위 링크:")
        side_info.extend([f"  - {u}" for u in links[:6]])
    side_md = "\n".join(side_info)

    # 2) 본문 수집
    page_texts_joined_list = await load_pages_async(links)
    context_text = page_texts_joined_list[0] if page_texts_joined_list else docs_str

    # 3) LLM 호출
    messages = history_to_messages(history) + [HumanMessage(user_input)]
    try:
        answer = await document_chain.ainvoke(
            {"messages": messages, "context": context_text}
        )
    except Exception as e:
        answer = f"문서 기반 답변 생성 중 오류가 발생했습니다: {e}"

    answer = strip_think(answer)

    # 최종 응답 + 우측 패널 요약
    return answer, side_md


# ========= Gradio UI =========
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 🚗📈 현대차 등 산업 전망 Q&A (DeepSeek-R1 on Ollama + DuckDuckGo RAG)")

    chat = gr.ChatInterface(
        fn=respond,  # async 함수 사용 가능
        additional_outputs=[gr.Markdown(label="검색/컨텍스트 요약")],
        chatbot=gr.Chatbot(
            label="시장분석 전문가",
            height=520,
        ),
        title="",
        description=(
            "로컬 **Ollama(DeepSeek-R1)** 모델로 동작합니다. DuckDuckGo 뉴스 검색 → 본문 수집 → RAG로 답변해요.\n"
            "※ 모델의 내부 사고(<think>...</think>)는 자동 숨김"
        ),
        theme="soft",
        retry_btn="다시 생성",
        undo_btn="이전 메시지 삭제",
        clear_btn="대화 초기화",
        examples=[
            "2025년 현대자동차 미국 시장 전망은 어떻게 되나요?",
            "전기차 인센티브 변화가 국내 완성차 수출에 미칠 영향은?",
            "반도체 공급 이슈가 2025년 자동차 생산에 어떤 리스크가 있나요?",
        ],
    )

    with gr.Accordion("시스템/모델 정보", open=False):
        gr.Markdown(
            f"""
- 사용 모델: **{MODEL_NAME}**  
- 로컬 서버: **http://localhost:11434** (Ollama 기본)  
- 시스템 프롬프트: `{SYSTEM_PROMPT}`  
- Temperature: `{TEMPERATURE}`  
- 검색: DuckDuckGo(뉴스, 지역={DDG_REGION}, 기간={DDG_TIME})
"""
        )

if __name__ == "__main__":
    # 내부망만 사용할 경우 share=False 권장
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)