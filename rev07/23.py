"""
[OpenCode 변환본] 
Closed LLM(OpenAI) → 오픈소스 LLM(로컬 Ollama + DeepSeek-R1) + Gradio UI

- ✅ API Key 불필요 (로컬 PC에 Ollama 및 모델 설치되어 있다고 가정)
- ✅ 임베딩: OllamaEmbeddings (예: nomic-embed-text)
- ✅ LLM: ChatOllama ("deepseek-r1:latest")
- ✅ 벡터DB: Chroma(persist)
- ✅ PDF 로더: PyPDFLoader
- ✅ 텍스트 분할: RecursiveCharacterTextSplitter
- ✅ 간단한 라우팅 + 관련성 그레이더 + RAG 생성
- ✅ Gradio 인터페이스 제공

사전 준비(터미널):
1) Ollama 설치 후 모델 준비
   - ollama pull deepseek-r1:latest
   - ollama pull nomic-embed-text

2) 파이썬 패키지
   - pip install langchain langchain-community langchain-text-splitters chromadb pypdf gradio

폴더 구조 가정:
- ../data/*.pdf          : 색인할 PDF들
- ../chroma_store        : Chroma 퍼시스트 디렉터리
"""

from __future__ import annotations

import os
from glob import glob
from typing import List, Dict, Any, Literal, Tuple

# LangChain - 로컬(Community) 생태계 구성요소
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from langchain_chroma import Chroma

import gradio as gr


# =========================
# 1) PDF 읽기 및 분할 함수
# =========================
def read_pdf_and_split_text(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    주어진 PDF 파일을 읽고 텍스트를 분할합니다.

    매개변수:
        pdf_path (str): PDF 파일의 경로.
        chunk_size (int): 각 텍스트 청크의 크기. 기본값은 1000.
        chunk_overlap (int): 청크 간의 중첩 크기. 기본값은 100.

    반환값:
        List[Document]: 분할된 LangChain Document 리스트.
    """
    print(f"[PDF] {pdf_path} -----------------------------")
    pdf_loader = PyPDFLoader(pdf_path)
    data_from_pdf = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(data_from_pdf)
    print(f" -> Number of splits: {len(splits)}\n")
    return splits


# ====================================
# 2) 임베딩/벡터DB(Chroma) 초기화/로딩
# ====================================
def build_or_load_vectorstore(
    data_glob: str = "../data/*.pdf",
    persist_directory: str = "../chroma_store",
    embedding_model: str = "nomic-embed-text",
) -> Chroma:
    """
    Chroma 벡터스토어를 생성하거나 로드합니다.
    - OllamaEmbeddings를 이용해 로컬 임베딩 수행.
    - persist_directory가 있으면 로드, 없으면 PDF를 임베딩해 생성.

    반환:
        Chroma 인스턴스
    """
    # 🔹 Ollama 임베딩 (사전: ollama pull nomic-embed-text)
    embedding = OllamaEmbeddings(model=embedding_model)

    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print("[VectorStore] 기존 Chroma store 로딩")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vectorstore

    print("[VectorStore] 신규 Chroma store 생성")
    all_docs: List[Document] = []
    pdf_files = sorted(glob(data_glob))
    if not pdf_files:
        print(f"⚠️ PDF를 찾지 못했습니다: {data_glob}")
    for path in pdf_files:
        chunks = read_pdf_and_split_text(path)
        # 메타데이터 보강(파일명)
        for d in chunks:
            d.metadata = {**d.metadata, "source_file": os.path.basename(path)}
        all_docs.extend(chunks)

    # 100개씩 나눠서 초기 색인
    vectorstore: Chroma | None = None
    for i in range(0, len(all_docs), 100):
        batch = all_docs[i : i + 100]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory=persist_directory,
            )
        else:
            vectorstore.add_documents(documents=batch)

    if vectorstore is None:
        # 빈 스토어라도 반환 (오류 방지)
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embedding
        )

    print("[VectorStore] 준비 완료")
    return vectorstore


# ===============================
# 3) LLM (DeepSeek-R1 via Ollama)
# ===============================
def build_llm(model_name: str = "deepseek-r1:latest") -> ChatOllama:
    """
    로컬 Ollama의 DeepSeek-R1 모델을 Chat 인터페이스로 초기화합니다.
    - 사전 준비: ollama pull deepseek-r1:latest
    - R1은 <think>...</think> 내부 사유 텍스트를 포함할 수 있으므로,
      최종 출력에서는 제거(후처리)하는 편이 좋습니다.
    """
    llm = ChatOllama(
        model=model_name,
        temperature=0.3,  # 일관성 위주
        num_ctx=8192,     # 컨텍스트 길이 여유
    )
    return llm


def strip_think_tags(text: str) -> str:
    """
    DeepSeek-R1의 <think>...</think> 내부 사유 텍스트를 제거하여 깔끔한 답변만 남깁니다.
    """
    import re

    # 여러 번 등장할 수 있으므로 전부 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 앞뒤 공백 정리
    return text.strip()


# =====================================
# 4) 라우팅/그레이딩(간단 규칙 + LLM 보조)
# =====================================
VECTORTOPICS_KW = [
    "서울", "서울시", "뉴욕", "자율주행", "온실가스", "발전계획", "도시", "계획", "저감", "교통", "플랜",
]


def simple_router(question: str) -> Literal["vectorstore", "casual_talk"]:
    """
    규칙 기반 라우터:
    - 질문 안에 도시계획/정책 관련 키워드가 포함되면 vectorstore
    - 그 외는 casual_talk
    """
    q = (question or "").lower()
    if any(kw in question for kw in VECTORTOPICS_KW):
        return "vectorstore"
    # 길이가 매우 짧고 인사/안부/잡담이면 casual
    smalltalk_kw = ["안녕", "안녕하세요", "하이", "잘 지냈", "요즘 어때", "뭐해", "고마워"]
    if any(kw in question for kw in smalltalk_kw):
        return "casual_talk"
    # 기본값
    return "vectorstore" if len(q) > 12 else "casual_talk"


def simple_relevance_grader(doc_text: str, question: str) -> bool:
    """
    매우 단순한 관련성 판단(키워드 겹침 기반).
    - 빠르고 의존성 없이 동작.
    - 필요시 LLM 보조 그레이더로 확장 가능.
    """
    q = question.lower()
    d = doc_text.lower()
    hits = 0
    for kw in set([*VECTORTOPICS_KW, *q.split()]):
        if len(kw) < 2:
            continue
        if kw in d:
            hits += 1
    return hits >= 2  # 겹치는 토큰이 일정 이상이면 관련 있다고 간주


# ===============================
# 5) RAG 체인(검색 → 필터 → 생성)
# ===============================
def run_rag_pipeline(
    question: str,
    vectorstore: Chroma,
    llm: ChatOllama,
    k: int = 5,
    max_context_chars: int = 6000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    - 라우팅 → (vectorstore 검색) → 그레이딩 → LLM 생성
    - casual_talk 라우트면 단순 LLM 응답

    반환:
        (final_answer, sources)
        - final_answer: 최종 답변 문자열(think 제거)
        - sources: 사용된 문서 메타/미리보기 리스트
    """
    route = simple_router(question)

    if route == "casual_talk":
        prompt = f"다음 질문/대화에 한국어로 친근하게 답하세요.\n\n질문: {question}"
        raw = llm.invoke(prompt).content
        return strip_think_tags(raw), []

    # vectorstore 경로
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.invoke(question)

    # 간이 그레이딩으로 관련 문서 거르기
    filtered: List[Document] = []
    for d in docs:
        if simple_relevance_grader(d.page_content, question):
            filtered.append(d)

    # 컨텍스트 구성(길이 제한)
    context_parts = []
    total_len = 0
    used_sources: List[Dict[str, Any]] = []
    for d in filtered:
        snippet = d.page_content.strip()
        meta = d.metadata or {}
        # 너무 긴 청크는 일부만
        if len(snippet) > 1200:
            snippet = snippet[:1100] + " ..."

        addition = f"[source: {meta.get('source_file','unknown')} | page: {meta.get('page', 'NA')}]\n{snippet}\n"
        if total_len + len(addition) > max_context_chars:
            break
        context_parts.append(addition)
        total_len += len(addition)

        used_sources.append(
            {
                "source_file": meta.get("source_file", "unknown"),
                "page": meta.get("page", "NA"),
                "preview": snippet[:300],
            }
        )

    context = "\n\n".join(context_parts) if context_parts else "(관련 문서가 충분치 않습니다.)"

    # 답변 프롬프트
    system = (
        "너는 도시 계획/정책 전문가야. 아래 컨텍스트에 기반해서 "
        "질문에 정확하고 간결하게 한국어로 답해줘. "
        "컨텍스트에 없는 사실은 추측하지 말고 모른다고 말해."
    )
    user = f"""[질문]
{question}

[컨텍스트]
{context}

[지침]
- 필요한 경우 서울/뉴욕의 정책 맥락을 간단히 정리.
- 근거가 된 소스 파일명과 페이지를 문장 끝에 (파일명 p.페이지) 형태로 가볍게 표기.
- 모르는 내용은 '제시된 문서에서 확인되지 않았습니다.'라고 명시.
"""

    raw_answer = llm.invoke(f"{system}\n\n{user}").content
    final_answer = strip_think_tags(raw_answer)
    return final_answer, used_sources


# =========================
# 6) Gradio UI 구성
# =========================
# 전역 리소스(앱 시작 시 1회 준비)
VECTORSTORE = build_or_load_vectorstore(
    data_glob="../data/*.pdf",
    persist_directory="../chroma_store",
    embedding_model="nomic-embed-text",
)
LLM = build_llm("deepseek-r1:latest")


def ui_ask(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Gradio용 핸들러. 질문을 받아 RAG 파이프라인 수행.
    """
    if not question or not question.strip():
        return "질문을 입력해 주세요.", []

    answer, sources = run_rag_pipeline(
        question=question.strip(), vectorstore=VECTORSTORE, llm=LLM, k=5
    )
    return answer, sources


# Gradio Blocks
with gr.Blocks(title="OpenCode • 로컬 DeepSeek-R1 RAG") as demo:
    gr.Markdown(
        """
# OpenCode • 로컬 DeepSeek-R1 RAG (Ollama + Chroma)
- 좌측 입력에 질문을 적고 **질의하기**를 누르세요.
- 주제에 따라 자동으로 색인된 PDF(VectorStore) 검색 또는 일상 대화로 라우팅합니다.
- API Key 불필요. 모든 연산은 로컬에서 수행됩니다.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(
                label="질문 (예: 서울 온실가스 저감 계획은 무엇인가요?)",
                placeholder="무엇을 도와드릴까요?",
                lines=3,
            )
            btn = gr.Button("질의하기", variant="primary")

        with gr.Column(scale=5):
            out_md = gr.Markdown(label="응답")
            out_json = gr.JSON(label="참조된 소스(파일/페이지/미리보기)")

    btn.click(fn=ui_ask, inputs=inp, outputs=[out_md, out_json])

# 이 스크립트를 직접 실행할 경우 Gradio 앱을 띄웁니다.
if __name__ == "__main__":
    # share=True가 필요하면 외부 노출 가능(선택)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

