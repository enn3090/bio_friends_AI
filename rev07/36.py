"""
OpenCode 변환본 • DeepSeek-R1(로컬 Ollama) + Gradio + EasyOCR로 영수증 OCR & 표 구조화

기능
- 영수증/장바구니/명세서 등 사진에서 텍스트를 추출(EasyOCR)하고 DeepSeek-R1로 항목/금액/일자 등을 구조화
- 결과를 테이블(DataFrame)과 요약(JSON)으로 출력, 원본+박스 오버레이 이미지 제공
- 완전 로컬 동작(API Key 불필요). 한국어/영어 혼합 인식 지원

설치(한 번만)
    pip install -U gradio langchain langchain-community easyocr pillow numpy pandas
    # (선택) GPU 가속: PyTorch를 환경에 맞게 설치 → https://pytorch.org/get-started/locally/
    # Ollama & 모델 준비: https://ollama.com → `ollama pull deepseek-r1`

실행
    python deepseek_r1_gradio_receipt_ocr.py
    # 브라우저가 자동 열립니다 (http://localhost:7860)

주의
- EasyOCR은 첫 실행 시 모델을 자동 다운로드합니다(인터넷 필요). 이후 캐시 사용.
- 개인정보가 포함될 수 있으니 로컬 보관/폐기 정책을 준수하세요.
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
from langchain_community.chat_models import ChatOllama

# OCR
import easyocr

# =========================
# LLM (DeepSeek-R1 via Ollama)
# =========================

def make_llm(model: str = "deepseek-r1", temperature: float = 0.2) -> ChatOllama:
    """로컬 Ollama용 LangChain Chat 모델. API Key 불필요."""
    return ChatOllama(model=model, temperature=temperature)

LLM = make_llm()

# =========================
# 유틸
# =========================

def draw_boxes(image: Image.Image, ocr_results: List[Tuple[List[List[int]], str, float]]) -> Image.Image:
    """EasyOCR 결과(bbox, text, conf)를 받아 박스를 덧그린 이미지를 반환."""
    vis = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(vis)
    for bbox, text, conf in ocr_results:
        # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        poly = [(int(x), int(y)) for x, y in bbox]
        color = (14, 165, 233, 180) if conf >= 0.6 else (245, 158, 11, 180)
        draw.polygon(poly, outline=color)
    return vis


def ocr_image(image: Image.Image, langs: List[str], detail: int = 1, text_threshold: float = 0.5) -> Tuple[List[Tuple], str]:
    """EasyOCR로 이미지에서 텍스트 추출. 반환: (raw_results, merged_text)"""
    # EasyOCR 리더는 생성 비용이 크므로 매 호출 캐시를 피하려면 글로벌/상태화 가능(여기선 간단화)
    reader = easyocr.Reader(langs, gpu=False)  # GPU=True로도 변경 가능
    results = reader.readtext(np.array(image), detail=detail, text_threshold=text_threshold)

    # 텍스트 라인 병합(좌->우, 상->하 대략 정렬)
    def centroid(b):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        return (sum(xs)/4.0, sum(ys)/4.0)

    lines = []
    for r in results:
        bbox, text, conf = r
        cx, cy = centroid(bbox)
        lines.append((cy, cx, text.strip(), conf))
    lines.sort(key=lambda x: (round(x[0]/20), x[1]))  # 대략적 줄 정렬
    merged = "\n".join([l[2] for l in lines if l[2]])
    return results, merged


PROMPT_SYS = (
    "당신은 영수증 OCR 정규화 도우미입니다.\n"
    "다음 OCR 텍스트를 분석해 구조화된 JSON으로 변환하세요.\n"
    "키는 vendor, date, time, currency, items, subtotal, tax, tip, total, notes 입니다.\n"
    "items는 [{description, qty, unit_price, line_total}] 배열로 작성하고, 숫자는 숫자형으로.\n"
    "날짜/시간/통화 기호는 최대한 추론하고, 없으면 null.\n"
    "출력은 반드시 JSON만. 다른 텍스트 금지.\n"
)

PROMPT_USER = (
    """
OCR 텍스트:
---
{ocr_text}
---
요구사항:
- 항목(Items)을 최대한 추출하고, 없는 값은 null로 채우되 합계(total)는 텍스트에서 우선.
- 금액이 여러 통화 기호를 섞어 쓰면 가장 그럴듯한 하나를 선택.
- 소수점/쉼표 문제를 정규화.
- 수량/단가/라인합계가 일치하지 않으면 라인합계 우선.
- JSON 외 문구는 절대 출력하지 마세요.
    """
)


def normalize_with_llm(ocr_text: str) -> Dict[str, Any]:
    try:
        raw = LLM.invoke(PROMPT_SYS + "\n\n" + PROMPT_USER.format(ocr_text=ocr_text)).content
        try:
            data = json.loads(raw)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(m.group(0)) if m else {}
        if not isinstance(data, dict):
            raise ValueError("파싱 실패")
        return data
    except Exception:
        # 최소 스키마 기본값
        return {
            "vendor": None,
            "date": None,
            "time": None,
            "currency": None,
            "items": [],
            "subtotal": None,
            "tax": None,
            "tip": None,
            "total": None,
            "notes": "LLM 정규화 실패"
        }


def items_to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["description", "qty", "unit_price", "line_total"]
    df = pd.DataFrame(items, columns=cols)
    # 타입 정리
    for c in ["qty", "unit_price", "line_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# Gradio 핸들러
# =========================

def handle(image: Image.Image, lang_choice: List[str], use_llm: bool, text_thr: float) -> Tuple[Image.Image, str, str, pd.DataFrame, str]:
    if image is None:
        return None, "이미지를 업로드하세요.", "", pd.DataFrame(), ""

    # 1) OCR
    raw_results, merged_text = ocr_image(image, lang_choice or ["ko", "en"], detail=1, text_threshold=float(text_thr))

    # 2) 시각화
    overlay = draw_boxes(image, raw_results)

    # 3) LLM 정규화(선택)
    meta_json = {}
    if use_llm:
        meta_json = normalize_with_llm(merged_text)
    else:
        meta_json = {
            "vendor": None,
            "date": None,
            "time": None,
            "currency": None,
            "items": [],
            "subtotal": None,
            "tax": None,
            "tip": None,
            "total": None,
            "notes": "LLM 미사용"
        }

    # 4) 테이블
    df = items_to_df(meta_json.get("items", []))

    # 5) 헤더 요약
    header_lines = []
    for k in ["vendor", "date", "time", "currency", "subtotal", "tax", "tip", "total"]:
        header_lines.append(f"{k}: {meta_json.get(k)}")
    header = "\n".join(header_lines)

    return overlay, merged_text, json.dumps(meta_json, ensure_ascii=False, indent=2), df, header


# =========================
# UI 구성
# =========================

def build_ui():
    with gr.Blocks(title="OpenCode • Receipt OCR (DeepSeek-R1 + EasyOCR)") as demo:
        gr.Markdown(
            """
            # OpenCode • 영수증 OCR 스튜디오
            - **EasyOCR**로 텍스트 추출 → **DeepSeek-R1(Ollama)** 로 구조화(JSON)
            - 테이블, 합계, 일자/가맹점 등 자동 정리
            - 완전 로컬, API Key 불필요
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="영수증 이미지 업로드", height=420)
                with gr.Row():
                    lang_choice = gr.CheckboxGroup(
                        choices=["ko", "en", "ja", "zh", "de", "fr", "es"],
                        value=["ko", "en"],
                        label="OCR 언어",
                    )
                    text_thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="텍스트 임계값")
                use_llm = gr.Checkbox(value=True, label="DeepSeek-R1로 구조화(JSON) 수행")
                run = gr.Button("인식 실행", variant="primary")

            with gr.Column(scale=1):
                overlay = gr.Image(label="인식 박스", height=420)
                raw_text = gr.Textbox(label="OCR 텍스트", lines=10)
                header = gr.Textbox(label="요약 헤더", lines=6)
                json_out = gr.Code(label="구조화 결과(JSON)", language="json")
                table = gr.Dataframe(label="항목 테이블", interactive=False)

        run.click(
            handle,
            inputs=[image, lang_choice, use_llm, text_thr],
            outputs=[overlay, raw_text, json_out, table, header],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
