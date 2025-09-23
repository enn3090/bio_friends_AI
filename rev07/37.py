"""
OpenCode 변환본 • DeepSeek-R1(로컬 Ollama) + Gradio + PaddleOCR 로 일반 테이블/영수증 OCR & 표 정리

기능
- 이미지(사진/스캔)에서 **일반 테이블/영수증**을 OCR로 인식
- 테이블은 PaddleOCR **PP-Structure**로 구조(셀)까지 추출 → DataFrame/HTML로 표시
- 영수증/일반 OCR 텍스트는 줄 단위 병합 + 박스 오버레이
- DeepSeek-R1(Ollama)로 **헤더 추론/정규화**(옵션) → 깨끗한 JSON/표로 출력

설치
    pip install -U gradio paddleocr pandas pillow numpy langchain langchain-community
    # (선택) GPU가 있다면 PyTorch를 환경에 맞게 설치해두면 속도 향상
    # Ollama & 모델: https://ollama.com → `ollama pull deepseek-r1`

실행
    python deepseek_r1_gradio_table_ocr.py

메모
- PaddleOCR는 처음 실행 시 모델을 다운로드합니다(인터넷 필요). 이후 캐시 사용.
- 완전 로컬 처리가 목적이라 외부 API Key는 쓰지 않습니다.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import gradio as gr

# LLM (로컬 Ollama)
from langchain_community.chat_models import ChatOllama

# OCR (PaddleOCR)
from paddleocr import PPStructure, PaddleOCR

# =========================
# 환경/유틸
# =========================

def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

USE_GPU = cuda_available()


def draw_boxes(image: Image.Image, boxes: List[Tuple[List[List[int]], str, float]]) -> Image.Image:
    """OCR/레이아웃 박스를 이미지에 그립니다."""
    vis = image.convert("RGBA").copy()
    drw = ImageDraw.Draw(vis)
    for bbox, text, conf in boxes:
        poly = [(int(x), int(y)) for x, y in bbox]
        color = (14, 165, 233, 180) if conf is None or conf >= 0.6 else (245, 158, 11, 180)
        drw.polygon(poly, outline=color)
    return vis


# =========================
# LLM: DeepSeek-R1 (Ollama)
# =========================

def make_llm(model: str = "deepseek-r1", temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature)

LLM = make_llm()

PROMPT_TABLE_CLEAN = (
    "당신은 표 정규화 도우미입니다. 아래 CSV 데이터를 바탕으로, 첫 행이 진짜 헤더인지 판단하고,\n"
    "적절한 헤더를 추론해 표를 깔끔히 정리하세요. 모든 수치는 숫자형으로 파싱하고, 빈칸은 null.\n"
    "JSON만 출력하세요. 스키마: {\"columns\": [..], \"rows\": [[..],[..],...]}\n"
)

PROMPT_TABLE_USER = """
원본 CSV:
---
{csv}
---
요구사항:
- columns는 문자열 배열
- rows는 2차원 배열(각 행은 columns 순서와 동일)
- 숫자는 숫자형으로, 날짜/시간/통화 기호는 텍스트로 남기되 일관성 유지
- JSON 이외의 텍스트는 절대 출력하지 말 것
"""


# =========================
# OCR: 일반 텍스트 (영수증 등)
# =========================

def run_general_ocr(img: Image.Image, lang: str = "korean") -> Tuple[Image.Image, str]:
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=USE_GPU, show_log=False)
    np_img = np.array(img)
    result = ocr.ocr(np_img, cls=True)

    flat = []  # (bbox, text, conf)
    lines = []

    for block in result:
        for (bbox, (text, conf)) in block:
            flat.append((bbox, text, float(conf)))
            lines.append((bbox, text, float(conf)))

    # 줄 정렬: y-중심, 그 후 x-중심
    def centroid(b):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        return (sum(xs) / 4.0, sum(ys) / 4.0)

    lines2 = []
    for bbox, text, conf in lines:
        cx, cy = centroid(bbox)
        lines2.append((round(cy / 15), cx, text, conf, bbox))
    lines2.sort(key=lambda x: (x[0], x[1]))
    merged = "\n".join([t[2] for t in lines2 if t[2]])

    overlay = draw_boxes(img, [(l[4], l[2], l[3]) for l in lines2])
    return overlay, merged


# =========================
# OCR: 테이블(구조 포함)
# =========================

def area(box):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
    w = max(abs(x2 - x1), abs(x3 - x4))
    h = max(abs(y4 - y1), abs(y3 - y2))
    return max(1, w) * max(1, h)


def run_table_ocr(img: Image.Image, lang: str = "korean") -> Tuple[List[pd.DataFrame], List[str], Image.Image]:
    """PP-Structure 로 표 구조 추출 → (DataFrames, HTMLs, overlay)"""
    pp = PPStructure(show_log=False, use_gpu=USE_GPU, lang=lang)
    np_img = np.array(img)
    results = pp(np_img)

    tables = []
    htmls = []
    boxes = []

    for r in results:
        if r.get('type') == 'table' and 'res' in r and r['res'] is not None:
            html = r['res'].get('html', '')
            if html:
                try:
                    dfs = pd.read_html(io.StringIO(html))
                    if dfs:
                        tables.append(dfs[0])
                        htmls.append(html)
                        if 'bbox' in r:
                            # PP-Structure bbox는 [x1, y1, x2, y2]
                            x1, y1, x2, y2 = r['bbox']
                            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            boxes.append((bbox, 'table', None))
                except Exception:
                    pass

    # 박스 오버레이
    overlay = draw_boxes(img, boxes) if boxes else img
    return tables, htmls, overlay


# =========================
# 정규화(LLM)
# =========================

def normalize_table_with_llm(df: pd.DataFrame) -> Dict[str, Any]:
    csv = df.to_csv(index=False)
    raw = LLM.invoke(PROMPT_TABLE_CLEAN + "\n\n" + PROMPT_TABLE_USER.format(csv=csv)).content
    try:
        data = json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group(0)) if m else {"columns": list(df.columns), "rows": df.fillna("").values.tolist()}
    # 안전 체크
    cols = data.get("columns")
    rows = data.get("rows")
    if not isinstance(cols, list) or not isinstance(rows, list):
        cols = list(df.columns)
        rows = df.fillna("").values.tolist()
    return {"columns": cols, "rows": rows}


# =========================
# Gradio 핸들러
# =========================

def handle(image: Image.Image, mode: str, lang: str, use_llm: bool) -> tuple:
    if image is None:
        return None, "이미지를 업로드하세요.", gr.update(visible=False), pd.DataFrame(), ""

    if mode == "테이블 OCR":
        dfs, htmls, overlay = run_table_ocr(image, lang=lang)
        if not dfs:
            return overlay, "표를 감지하지 못했습니다.", gr.update(visible=False), pd.DataFrame(), ""
        # 가장 큰 표 1개를 대표로 사용(필요 시 확장 가능)
        # 여기서는 첫 번째 표를 사용
        df = dfs[0]
        info = f"감지된 표 수: {len(dfs)} / shape: {df.shape}"
        if use_llm:
            cleaned = normalize_table_with_llm(df)
            cols, rows = cleaned["columns"], cleaned["rows"]
            df2 = pd.DataFrame(rows, columns=cols)
            json_out = json.dumps(cleaned, ensure_ascii=False, indent=2)
            return overlay, info, gr.update(value=json_out, visible=True), df2, htmls[0]
        else:
            json_out = json.dumps({"columns": list(df.columns), "rows": df.fillna("").values.tolist()}, ensure_ascii=False, indent=2)
            return overlay, info, gr.update(value=json_out, visible=True), df, htmls[0]

    else:  # 일반 OCR
        overlay, text = run_general_ocr(image, lang=lang)
        return overlay, text, gr.update(visible=False), pd.DataFrame(), ""


# =========================
# UI 구성
# =========================

def build_ui():
    with gr.Blocks(title="OpenCode • General Table & Receipt OCR (PaddleOCR + DeepSeek-R1)") as demo:
        gr.Markdown(
            """
            # OpenCode • 테이블/영수증 OCR 스튜디오
            - **PaddleOCR**: 일반 OCR + **PP-Structure** 테이블 인식
            - **DeepSeek-R1(Ollama)**: 헤더 추론·정규화(옵션)
            - 표는 **DataFrame/JSON/HTML** 모두 확인 가능
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="이미지 업로드", height=420)
                mode = gr.Radio(["테이블 OCR", "일반 OCR"], value="테이블 OCR", label="모드")
                lang = gr.Dropdown(["korean", "ch", "en", "japan"], value="korean", label="언어")
                use_llm = gr.Checkbox(value=True, label="DeepSeek-R1로 표 정규화")
                run = gr.Button("실행", variant="primary")

            with gr.Column(scale=1):
                overlay = gr.Image(label="오버레이", height=420)
                info_or_text = gr.Textbox(label="정보 / OCR 텍스트", lines=8)
                json_out = gr.Code(label="정규화 결과(JSON)", language="json", visible=False)
                df_out = gr.Dataframe(label="표(DataFrame)", interactive=False)
                html_out = gr.HTML(label="원본 표(HTML)")

        run.click(handle, inputs=[image, mode, lang, use_llm], outputs=[overlay, info_or_text, json_out, df_out, html_out])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)

