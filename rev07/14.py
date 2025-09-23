"""
Ollama + Gradio 기반 이미지 설명/비교 앱
- 이미지 이해(비전)는 qwen2.5-vl 같은 VLM을 사용
- 심화 요약/정리는 DeepSeek-R1(텍스트 전용)로 후처리
- 로컬 PC에 Ollama가 설치되어 있고, qwen2.5-vl 및 deepseek-r1 모델이 pull 되어 있다고 가정

설치 예시(터미널):
    ollama pull qwen2.5-vl
    ollama pull deepseek-r1
    pip install gradio pillow

주의: DeepSeek-R1은 텍스트 전용이므로, 이미지는 VLM(qwen2.5-vl)이 먼저 처리합니다.
"""

import base64
import os
from typing import Optional

import gradio as gr
import ollama
from PIL import Image

# -----------------------------
# 유틸: 이미지 -> Base64 data URL
# -----------------------------

def encode_image_to_data_url(image_path: str) -> str:
    """이미지 파일 경로를 받아 data URL(base64) 문자열로 변환합니다.
    Ollama의 비전 모델은 images 필드에 파일 경로나 data URL을 받을 수 있습니다.
    """
    mime = "image/jpeg"
    try:
        # Pillow로 열어 JPEG로 일단 통일(용량/호환성 고려), 원본 포맷 그대로 쓰고 싶으면 이 부분 수정 가능
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            from io import BytesIO
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
    except Exception:
        # 포맷 변환 실패 시, 바이너리 그대로 인코딩 시도
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


# -----------------------------
# 모델 호출 래퍼
# -----------------------------

def call_vlm(images_data_urls: list, prompt: str, model: str = "qwen2.5-vl") -> str:
    """비전 모델(VLM) 호출. 여러 장의 이미지를 한 번에 전달하여 비교도 가능.
    images_data_urls: data URL(base64) 문자열 리스트
    prompt: 사용자 프롬프트(한국어)
    model: Ollama에 설치된 비전 모델 이름
    """
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": images_data_urls,
        }
    ]
    resp = ollama.chat(model=model, messages=messages)
    return resp.get("message", {}).get("content", "")


def refine_with_deepseek(text: str, system_hint: Optional[str] = None, model: str = "deepseek-r1") -> str:
    """DeepSeek-R1로 후처리(정리/요약/구조화). 텍스트 전용.
    system_hint: 출력 스타일/형식을 안내하는 시스템 메시지
    """
    messages = []
    if system_hint:
        messages.append({"role": "system", "content": system_hint})
    messages.append({"role": "user", "content": text})
    resp = ollama.chat(model=model, messages=messages)
    return resp.get("message", {}).get("content", "")


# -----------------------------
# 태스크 함수들(Gradio에서 호출)
# -----------------------------

def describe_image(image_path: str, user_prompt: str, vlm_name: str, use_refine: bool) -> str:
    if not image_path:
        return "이미지가 필요합니다."
    data_url = encode_image_to_data_url(image_path)

    # 1) 비전 모델로 1차 분석
    base_prompt = user_prompt.strip() or "이 이미지의 핵심 요소(장소/사물/행동/분위기)를 한국어로 설명해줘."
    vlm_out = call_vlm([data_url], base_prompt, model=vlm_name)

    if not use_refine:
        return vlm_out

    # 2) DeepSeek-R1로 정리
    system_hint = (
        "너는 전문 데이터 해설가이다. 한국어로 간결하지만 정보 밀도가 높은 요약을 작성해라. "
        "소제목, 불릿포인트, 중요 수치/디테일을 강조하고 과도한 추측은 피하라."
    )
    return refine_with_deepseek(vlm_out, system_hint=system_hint)


def compare_two_images(img1_path: str, img2_path: str, user_prompt: str, vlm_name: str, use_refine: bool) -> str:
    if not img1_path or not img2_path:
        return "두 이미지를 모두 업로드하세요."

    data_urls = [encode_image_to_data_url(img1_path), encode_image_to_data_url(img2_path)]

    base_prompt = (
        user_prompt.strip()
        or "두 이미지를 비교해 차이점과 공통점을 항목별로 정리해줘. 구도, 조명, 색감, 주요 사물/텍스처, 분위기, 사용 목적(추정)을 포함해 한국어로 설명해줘."
    )

    vlm_out = call_vlm(data_urls, base_prompt, model=vlm_name)

    if not use_refine:
        return vlm_out

    system_hint = (
        "두 이미지 비교 결과를 한국어로 정리한다. \n"
        "- 공통점 \n- 차이점(구도/조명/색감/사물/텍스처/분위기) \n- 요약 결론(한 문장)\n"
        "가능한 한 관찰 가능한 사실에 근거하고 과도한 추론은 피한다."
    )
    return refine_with_deepseek(vlm_out, system_hint=system_hint)


def compare_oecd_charts(img2021_path: str, img2022_path: str, focus_country: str, vlm_name: str, use_refine: bool) -> str:
    if not img2021_path or not img2022_path:
        return "두 개의 차트 이미지를 모두 업로드하세요."

    data_urls = [encode_image_to_data_url(img2021_path), encode_image_to_data_url(img2022_path)]

    # 차트 비교에 특화된 프롬프트(한국 중심 설명 기본값)
    focus_country = focus_country.strip() or "한국"
    base_prompt = f"""
    두 개의 지표/차트 이미지를 비교해줘. 첫 번째는 2021년 데이터, 두 번째는 2022년 데이터이다.
    다음을 한국어로, {focus_country} 중심으로 설명하라:
    1) {focus_country}의 절대 값/순위/변화 방향(증가·감소·정체) 
    2) {focus_country}의 상대적 위치(주요국 대비 상위/중위/하위, 순위 변동)
    3) 연도 간 변화에서 눈에 띄는 포인트(급등/급락/역전/격차 확대·축소)
    4) 그래프 읽기 주의점(축 범위, 단위, 표기법, 표본 변화 등)
    가능한 경우 수치와 국가명을 명시하라. 모호한 경우는 추정을 자제하고 근거를 밝혀라.
    """

    vlm_out = call_vlm(data_urls, base_prompt, model=vlm_name)

    if not use_refine:
        return vlm_out

    system_hint = (
        f"OECD 스타일의 비교 리포트를 한국어로 작성한다. {focus_country}를 중심으로 섹션을 구성하고, "
        "불릿과 간단한 표(텍스트 기반)를 활용해 가독성을 높여라. 과도한 확신을 피하고, 시각적 한계나 해석상의 주의점을 마지막에 제시해라."
    )
    return refine_with_deepseek(vlm_out, system_hint=system_hint)


# -----------------------------
# Gradio UI
# -----------------------------

def build_ui():
    with gr.Blocks(title="이미지 설명/비교 - Ollama (DeepSeek-R1 + VLM)") as demo:
        gr.Markdown("""
        # 🖼️ 이미지 설명/비교 도구
        - **비전 분석**: `qwen2.5-vl` (로컬 VLM)
        - **심화 요약/정리**: `deepseek-r1` (텍스트 전용)
        
        ⚠️ DeepSeek-R1은 이미지를 직접 읽지 못합니다. 
        먼저 VLM이 이미지를 분석하고, 그 결과 텍스트를 DeepSeek-R1이 요약/정리합니다.
        """)

        with gr.Row():
            vlm_name = gr.Dropdown(
                label="비전 모델(VLM)",
                choices=["qwen2.5-vl", "llava:13b", "llava:7b", "minicpm-v"],
                value="qwen2.5-vl",
                info="Ollama에 설치된 비전 모델을 선택하세요."
            )
            use_refine = gr.Checkbox(label="DeepSeek-R1로 후처리", value=True)

        with gr.Tab("단일 이미지 설명"):
            with gr.Row():
                img_single = gr.Image(type="filepath", label="이미지 업로드")
                prompt_single = gr.Textbox(
                    label="프롬프트",
                    value="이 이미지에 대해 설명해주세요.",
                    lines=4,
                )
            out_single = gr.Markdown(label="결과")
            btn_single = gr.Button("분석하기", variant="primary")
            btn_single.click(
                fn=describe_image,
                inputs=[img_single, prompt_single, vlm_name, use_refine],
                outputs=out_single,
            )

        with gr.Tab("두 이미지 비교"):
            with gr.Row():
                img1 = gr.Image(type="filepath", label="이미지 1")
                img2 = gr.Image(type="filepath", label="이미지 2")
            prompt_compare = gr.Textbox(
                label="프롬프트",
                value="두 카페의 차이점을 설명해주세요.",
                lines=3,
            )
            out_compare = gr.Markdown(label="결과")
            btn_compare = gr.Button("비교하기", variant="primary")
            btn_compare.click(
                fn=compare_two_images,
                inputs=[img1, img2, prompt_compare, vlm_name, use_refine],
                outputs=out_compare,
            )

        with gr.Tab("OECD 차트(2021 vs 2022) 비교"):
            gr.Markdown("OECD R&D 등 지표 차트 두 장을 업로드하고, 한국 중심으로 연도 간 변화를 설명받아보세요.")
            with gr.Row():
                chart2021 = gr.Image(type="filepath", label="2021 차트 이미지")
                chart2022 = gr.Image(type="filepath", label="2022 차트 이미지")
            focus = gr.Textbox(label="중점 국가", value="한국")
            out_oecd = gr.Markdown(label="결과")
            btn_oecd = gr.Button("차트 비교 분석", variant="primary")
            btn_oecd.click(
                fn=compare_oecd_charts,
                inputs=[chart2021, chart2022, focus, vlm_name, use_refine],
                outputs=out_oecd,
            )

        gr.Markdown("""
        ---
        **팁**
        - 결과가 너무 장황하면 후처리 체크 해제(원문 유지) 또는 프롬프트에 "간결하게"를 추가하세요.
        - 모델이 설치되어 있지 않다면 `ollama pull <모델명>`으로 먼저 내려받으세요.
        - `llava` 계열은 이미지를 경로로 넘겨도 동작하나, 본 스크립트는 호환성 위해 data URL 방식을 기본 사용합니다.
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # 서버 실행. 로컬 네트워크 공개가 필요하면 share=True
    demo.launch()

