# ------------------------------------------------------------
# OpenAI API → Ollama(DeepSeek-R1) 변환 + Gradio 요약 앱
# ------------------------------------------------------------
# ✅ 기능
#   - PDF에서 헤더/푸터 영역 제거 후 본문 텍스트 추출 (PyMuPDF)
#   - 긴 문서를 청크로 분할하여 순차 요약 후 최종 통합 요약 (Ollama deepseek-r1)
#   - Gradio 기반 GUI: PDF 업로드 → 옵션 지정 → 요약/텍스트 다운로드
#
# ⚙️ 사전 준비
#   pip install gradio ollama pymupdf
#   (pymupdf는 코드에서 fitz로 import)
#   Ollama 설치 및 모델 준비:  
#     - https://ollama.com/download  
#     - 모델: `ollama pull deepseek-r1`
# ------------------------------------------------------------

import os
import io
import tempfile
from typing import List

import gradio as gr
import ollama
import fitz  # PyMuPDF

# -------------------------
# 구성값
# -------------------------
MODEL_NAME = "deepseek-r1"  # 로컬에 설치되어 있다고 가정
DEFAULT_TEMPERATURE = 0.1    # 요약은 보수적으로

# -------------------------
# PDF → 텍스트 추출 (헤더/푸터 제거)
# -------------------------

def pdf_to_text(pdf_path: str, header_height: float = 80, footer_height: float = 80) -> str:
    """PDF에서 헤더/푸터 영역을 잘라내고 본문 텍스트만 이어붙여 반환.
    - header_height, footer_height: 포인트 단위(px 유사). 페이지 높이에 맞춰 적절히 조정.
    """
    doc = fitz.open(pdf_path)
    chunks = []

    for page_idx, page in enumerate(doc, start=1):
        rect = page.rect  # (x0, y0, x1, y1)
        # 본문 영역만 clip해서 텍스트 추출
        body = page.get_text(
            "text",
            clip=(0, header_height, rect.width, rect.height - footer_height)
        )
        chunks.append(f"\n[페이지 {page_idx}]\n{body.strip()}\n" + ("-" * 40))

    return "\n".join(chunks)


# -------------------------
# 텍스트 청크 나누기
# -------------------------

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """긴 텍스트를 max_chars 기준으로 문단 단위 분할.
    (간단한 문자 길이 기준. 모델/환경에 따라 조절)
    """
    paragraphs = text.split("\n\n")
    chunks, buf = [], []
    size = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if size + len(p) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# -------------------------
# Ollama 호출 유틸
# -------------------------

def ollama_summarize(user_prompt: str, system_prompt: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """Ollama chat API로 요약 생성. system/user 메시지 사용."""
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": float(temperature)}
    )
    return resp["message"]["content"].strip()


# -------------------------
# 문서 요약 파이프라인 (분할 → 부분요약 → 통합요약)
# -------------------------

def summarize_text(text: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """긴 문서를 청크로 나눠 부분 요약하고, 마지막에 통합 요약."""
    # 1) 시스템 프롬프트 (사용자 요구에 맞춤)
    system_prompt = (
        "너는 다음 글을 요약하는 봇이다. 저자의 문제 인식과 주장, 주요 내용을 정확히 요약하라.\n"
        "출력 형식:\n\n"
        "# 제목\n\n"
        "## 저자의 문제 인식 및 주장 (15문장 이내)\n\n"
        "## 저자 소개\n"
    )

    # 2) 본문이 매우 길다면 덩어리로 요약
    chunks = chunk_text(text, max_chars=6000)

    partials = []
    for i, ch in enumerate(chunks, start=1):
        user_prompt = (
            f"아래는 문서의 일부(청크 {i}/{len(chunks)})이다. 핵심 주장과 근거를 보존하며 요약하라.\n"
            f"분량: 7~12문장 권장.\n\n=== 청크 시작 ===\n{ch}\n=== 청크 끝 ==="
        )
        partial = ollama_summarize(user_prompt, system_prompt, temperature)
        partials.append(partial)

    if len(partials) == 1:
        return partials[0]

    # 3) 부분 요약들을 통합 요약
    merged_source = "\n\n".join(partials)
    merge_prompt = (
        "다음은 문서의 부분 요약들을 모아 둔 것이다. 중복을 제거하고, 논리적 흐름을 정리하여"
        " 하나의 완성된 최종 요약을 작성하라. 표제/소제목 형식은 지키고, 구체적 수치/지표/연도는 보존하라.\n\n"
        f"=== 부분 요약들 시작 ===\n{merged_source}\n=== 부분 요약들 끝 ==="
    )
    final_summary = ollama_summarize(merge_prompt, system_prompt, temperature)
    return final_summary


# -------------------------
# Gradio 이벤트 핸들러
# -------------------------

def process_pdf(pdf_file, header_h, footer_h, temperature):
    """PDF 파일을 받아 텍스트 추출 후 요약. 결과 텍스트/요약, 다운로드 파일 반환."""
    if pdf_file is None:
        return gr.update(value=""), gr.update(value=""), None, None

    # 1) 임시 파일로 저장 경로 확보 (gradio 파일 객체는 temp 경로 포함)
    pdf_path = pdf_file.name

    # 2) 텍스트 추출
    extracted_text = pdf_to_text(pdf_path, header_height=header_h, footer_height=footer_h)

    # 3) 요약
    summary = summarize_text(extracted_text, temperature=temperature)

    # 4) 다운로드 파일 준비
    tmpdir = tempfile.mkdtemp(prefix="pdfsum_")
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    txt_path = os.path.join(tmpdir, f"{base}_extracted.txt")
    sum_path = os.path.join(tmpdir, f"{base}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary, extracted_text, txt_path, sum_path


# -------------------------
# Gradio UI 구성
# -------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📄 PDF 요약기 (Ollama DeepSeek-R1)
        - **헤더/푸터 제거** 후 본문만 추출해서 요약합니다.
        - 모델: `deepseek-r1` (로컬 Ollama)
        - 긴 문서도 **청크 요약 → 통합 요약**으로 안정 처리
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            pdf_in = gr.File(label="PDF 업로드", file_types=[".pdf"], type="filepath")
            header_h = gr.Number(value=80, label="헤더 높이(px)")
            footer_h = gr.Number(value=80, label="푸터 높이(px)")
            temp_in = gr.Slider(0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.1, label="Temperature (창의성)")
            run_btn = gr.Button("요약 실행", variant="primary")
        with gr.Column(scale=3):
            summary_out = gr.Textbox(label="최종 요약", lines=18)
            text_out = gr.Textbox(label="추출 텍스트(헤더/푸터 제거)", lines=18)
            with gr.Row():
                dl_text = gr.File(label="추출 텍스트 파일")
                dl_sum = gr.File(label="요약 결과 파일")

    run_btn.click(process_pdf, [pdf_in, header_h, footer_h, temp_in], [summary_out, text_out, dl_text, dl_sum])


# -------------------------
# 실행 엔트리
# -------------------------
if __name__ == "__main__":
    # 공유 필요 시: demo.launch(share=True)
    demo.launch()