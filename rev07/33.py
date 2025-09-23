"""
[OpenCode 변환본 • RAG + Gradio]

목표
- 사용자가 제공한 노트북 스타일 코드를 로컬 오픈소스 스택으로 그대로 돌릴 수 있는
  **단일 파이썬 스크립트**(API Key 불필요)로 재구성합니다.
- 그래픽 인터페이스는 **Gradio**를 사용합니다.
- 임베딩은 HuggingFace(BAAI/bge-m3), 벡터DB는 Chroma를 사용합니다.
- PDF 분할/색인/검색 전 과정을 버튼 한 번으로 수행하고, 질의를 통해 상위 K개 청크를 조회합니다.

사전 준비(한 번만)
    pip install -U langchain langgraph langchain-community langchain-text-splitters \
                    langchain-chroma langchain-huggingface chromadb pypdf gradio

    # CUDA 환경이라면 PyTorch(버전/쿠다는 환경에 맞게) 설치 후 동작 권장
    # https://pytorch.org/get-started/locally/

로컬 전제
- 로컬 PC에서 실행: 공개키/토큰 필요 없음.
- PDF는 기본적으로 ../data/*.pdf 경로에서 읽습니다(Gradio에서 업로드도 지원).
"""

from __future__ import annotations

import os
from glob import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr

# LangChain & 관련 유틸
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================
# 경로 / 상수
# =========================
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
DEFAULT_PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_store"))
DEFAULT_MODEL_NAME = "BAAI/bge-m3"  # 다국어/멀티벡터 대응 임베딩
DEFAULT_K = 5


# =========================
# 도우미
# =========================
def pick_device() -> str:
    """
    HuggingFaceEmbeddings 용 device 자동 선택.
    - CUDA 가능하면 'cuda', 아니면 'cpu'
    """
    try:
        import torch  # optional
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def log(msg: str) -> str:
    """Gradio 로그창에 보여줄 문자열(개행 포함)."""
    return f"{msg}\n"


# =========================
# PDF 로딩 & 분할
# =========================
def read_pdf_and_split_text(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    주어진 PDF 파일을 읽고 텍스트를 분할합니다.

    Args:
        pdf_path: PDF 경로
        chunk_size: 각 텍스트 청크의 토큰/문자 기준 크기
        chunk_overlap: 청크 간 중첩 크기

    Returns:
        list[Document]: 분할된 LangChain Document 리스트
    """
    print(f"PDF: {pdf_path} -----------------------------")
    loader = PyPDFLoader(pdf_path)
    data_from_pdf = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],  # 기본 분할 규칙
    )
    splits = splitter.split_documents(data_from_pdf)
    print(f"Number of splits: {len(splits)}\n")
    return splits


# =========================
# 임베딩 / 벡터 스토어
# =========================
@dataclass
class RAGResources:
    embeddings: HuggingFaceEmbeddings
    vectorstore: Optional[Chroma]


def make_embeddings(model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    HuggingFace 임베딩 생성 (로컬/오픈 모델, API Key 불필요)
    """
    if device is None:
        device = pick_device()

    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # 코사인 유사도 품질 안정화
    )
    # 간단한 점검: embed_documents는 list[str]을 인자로 받는다.
    _ = emb.embed_documents(["임베딩 점검 문장"])
    return emb


def build_or_load_chroma(
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    extra_pdf_paths: Optional[List[str]] = None,
) -> Tuple[Chroma, str]:
    """
    기존 Chroma 스토어가 있으면 로드, 없으면 PDF를 읽어 새로 생성/영속화.
    extra_pdf_paths 로 Gradio 업로드 파일들도 함께 색인 가능.
    """
    os.makedirs(persist_directory, exist_ok=True)

    if os.listdir(persist_directory):
        vs = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        return vs, "기존 Chroma 스토어를 불러왔습니다."
    else:
        all_pdfs = sorted(glob(os.path.join(data_dir, "*.pdf")))
        if extra_pdf_paths:
            all_pdfs.extend(extra_pdf_paths)

        if not all_pdfs:
            raise FileNotFoundError(
                f"색인할 PDF를 찾지 못했습니다. 폴더에 PDF를 추가하거나 업로드하세요.\n경로: {data_dir}"
            )

        all_docs = []
        for p in all_pdfs:
            try:
                all_docs.extend(read_pdf_and_split_text(p, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            except Exception as e:
                print(f"[경고] {p} 처리 중 오류: {e}")

        if not all_docs:
            raise RuntimeError("문서 분할 결과가 비었습니다. PDF가 손상되었거나 읽을 수 없습니다.")

        vs = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        return vs, f"새 Chroma 스토어 생성 완료. 총 청크 수: {len(all_docs)}"


# =========================
# 검색 함수
# =========================
def run_retrieval(
    vectorstore: Chroma,
    query: str,
    k: int = DEFAULT_K,
) -> List[Dict[str, Any]]:
    """
    질의를 바탕으로 상위 K개 문서를 조회하고 표시용 딕셔너리로 반환.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": int(k)})
    chunks = retriever.invoke(query)

    rows = []
    for i, doc in enumerate(chunks, 1):
        meta = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "")
        rows.append(
            {
                "rank": i,
                "source": meta.get("source", meta.get("file_path", "unknown")),
                "page": meta.get("page", meta.get("pdf_page", "NA")),
                "chunk_chars": len(content),
                "preview": content[:500] + ("…" if len(content) > 500 else ""),
                "metadata": meta,
                "full_content": content,
            }
        )
    return rows


# =========================
# Gradio UI 구성
# =========================
def launch_ui():
    state: Dict[str, Any] = {"embeddings": None, "vectorstore": None}

    with gr.Blocks(title="OpenCode • PDF RAG (Chroma + BGE-M3)") as demo:
        gr.Markdown(
            """
            # OpenCode • 로컬 PDF RAG
            - **임베딩**: BAAI/bge-m3 (HuggingFace, 로컬)
            - **벡터DB**: Chroma (영속 저장)
            - **입력**: `../data/*.pdf` + 업로드 PDF
            - **검색**: 코사인 유사도 Top-K
            """
        )

        with gr.Row():
            data_dir = gr.Textbox(
                label="PDF 폴더 경로",
                value=DEFAULT_DATA_DIR,
                info="기본은 ../data. 필요 시 변경하세요.",
            )
            persist_dir = gr.Textbox(
                label="Chroma 저장 경로",
                value=DEFAULT_PERSIST_DIR,
                info="비워두면 기본 경로를 사용합니다.",
            )

        with gr.Row():
            model_name = gr.Textbox(
                label="임베딩 모델",
                value=DEFAULT_MODEL_NAME,
                info="예: BAAI/bge-m3",
            )
            device_dropdown = gr.Dropdown(
                label="장치",
                choices=["auto", "cpu", "cuda"],
                value="auto",
                info="auto 권장 (CUDA 가능 시 자동 사용)",
            )
            chunk_size = gr.Slider(300, 2000, value=1000, step=50, label="청크 크기")
            chunk_overlap = gr.Slider(0, 400, value=100, step=10, label="청크 오버랩")

        with gr.Row():
            upload = gr.File(label="추가 PDF 업로드(선택)", file_count="multiple", file_types=[".pdf"])
        build_btn = gr.Button("색인 생성/불러오기", variant="primary")

        status = gr.Textbox(label="상태 로그", lines=8)

        gr.Markdown("---")

        with gr.Row():
            query = gr.Textbox(label="질의", placeholder="예) 서울시 쓰레기 저감 정책", lines=2)
            topk = gr.Slider(1, 20, value=DEFAULT_K, step=1, label="Top-K")
            search_btn = gr.Button("검색 실행")

        results = gr.Dataframe(
            headers=["rank", "source", "page", "chunk_chars", "preview"],
            datatype=["number", "str", "str", "number", "str"],
            label="검색 결과(요약)",
            interactive=False,
            row_count=5,
        )
        with gr.Accordion("선택 행 상세 보기", open=False):
            row_index = gr.Number(value=1, precision=0, label="행 번호(rank)")
            show_btn = gr.Button("상세 보기")
            full_content = gr.Textbox(label="전체 내용", lines=16)
            meta_view = gr.JSON(label="메타데이터")

        # -------- 콜백들 --------
        def on_build(idx_data_dir, idx_persist_dir, mdl_name, dev_sel, csize, coverlap, files) -> Tuple[str, Any]:
            logs = ""
            try:
                # 임베딩
                device = None if dev_sel == "auto" else dev_sel
                logs += log(f"[임베딩] 모델: {mdl_name}, 장치: {device or 'auto'}")
                emb = make_embeddings(mdl_name, device=device)
                state["embeddings"] = emb
                logs += log("[임베딩] 준비 완료")

                # 업로드 파일 경로 수집
                extra_paths = []
                if files:
                    for f in files:
                        extra_paths.append(f.name)  # Gradio 임시경로

                # 벡터스토어
                vs, msg = build_or_load_chroma(
                    emb,
                    persist_directory=idx_persist_dir or DEFAULT_PERSIST_DIR,
                    data_dir=idx_data_dir or DEFAULT_DATA_DIR,
                    chunk_size=int(csize),
                    chunk_overlap=int(coverlap),
                    extra_pdf_paths=extra_paths,
                )
                state["vectorstore"] = vs
                logs += log(f"[색인] {msg}")
                logs += log("[완료] 검색 준비가 되었습니다.")
                return logs, None
            except Exception as e:
                logs += log(f"[오류] {e}")
                return logs, None

        def on_search(q, k) -> Tuple[List[List[Any]], str]:
            logs = ""
            if not state.get("vectorstore"):
                logs += log("먼저 '색인 생성/불러오기'를 눌러 색인을 준비하세요.")
                return [], logs
            try:
                rows = run_retrieval(state["vectorstore"], q, int(k))
                table = [[r["rank"], r["source"], r["page"], r["chunk_chars"], r["preview"]] for r in rows]
                # 상세 보기용 캐시
                state["last_rows"] = rows
                logs += log(f"[검색] 상위 {len(rows)}개 결과를 반환했습니다.")
                return table, logs
            except Exception as e:
                logs += log(f"[오류] {e}")
                return [], logs

        def on_show(idx) -> Tuple[str, Dict[str, Any]]:
            rows = state.get("last_rows", [])
            if not rows:
                return "먼저 검색을 실행하세요.", {}
            try:
                i = int(idx) - 1
                if i < 0 or i >= len(rows):
                    return "유효하지 않은 행 번호입니다.", {}
                return rows[i]["full_content"], rows[i]["metadata"]
            except Exception as e:
                return f"[오류] {e}", {}

        build_btn.click(
            on_build,
            inputs=[data_dir, persist_dir, model_name, device_dropdown, chunk_size, chunk_overlap, upload],
            outputs=[status, results],  # results는 초기화(None)
        )

        search_btn.click(
            on_search,
            inputs=[query, topk],
            outputs=[results, status],
        )

        show_btn.click(
            on_show,
            inputs=[row_index],
            outputs=[full_content, meta_view],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


# =========================
# 스크립트 진입점
# =========================
if __name__ == "__main__":
    launch_ui()