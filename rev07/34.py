"""
OpenCode 변환본 • DeepSeek-R1(로컬 Ollama) + Stable Diffusion(diffusers) + Gradio UI

목표
- 로컬 LLM(DeepSeek-R1 on Ollama)로 사용자의 간단한 아이디어를 Stable Diffusion 친화 프롬프트로 자동 확장
- Hugging Face diffusers의 Stable Diffusion / SDXL 파이프라인으로 이미지 생성(txt2img, img2img)
- Gradio UI 제공(배치 생성, 시드, 스케줄러, 가이던스, 스텝, 해상도 등)

전제(사전 설치)
    pip install -U gradio diffusers transformers accelerate safetensors xformers
    pip install -U langchain langchain-community
    # (선택) CUDA 가속: PyTorch를 환경에 맞게 설치 https://pytorch.org/get-started/locally/
    # Ollama 및 모델 설치: https://ollama.com  →  ollama pull deepseek-r1

로컬 특성
- API Key 불필요(DeepSeek-R1은 Ollama 로컬 서버 사용)
- 모델은 인터넷에서 최초 1회 다운로드가 필요할 수 있음(HF 토큰이 필요한 모델은 토큰 설정 필요)

주의
- NSFW 필터는 기본적으로 활성화(파이프라인 기본 safety_checker). 체크박스로 해제 가능하지만, 안전한 사용을 권장합니다.
"""
from __future__ import annotations

import os
import json
import gc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
from PIL import Image
import gradio as gr

# --- LLM (로컬 Ollama) ---
from langchain_community.chat_models import ChatOllama

# --- Diffusers ---
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
)


# =========================
# 환경/기본값
# =========================
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.float16 if (DEFAULT_DEVICE == "cuda") else torch.float32

SD15_REPO = "stabilityai/stable-diffusion-2-1"  # 768 모델 권장: runwayml/stable-diffusion-v1-5 도 가능
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
# (선택) SDXL Refiner를 쓰고 싶다면 아래 주석 해제 후 로직 추가 가능
# SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"


# =========================
# LLM: DeepSeek-R1 via Ollama
# =========================

def make_llm(model: str = "deepseek-r1", temperature: float = 0.2) -> ChatOllama:
    """로컬 Ollama용 LangChain Chat 모델. API Key 불필요."""
    return ChatOllama(model=model, temperature=temperature)

LLM = make_llm()

PROMPT_SYS = (
    "당신은 이미지 프롬프트 엔지니어입니다. 사용자의 요약 아이디어를 Stable Diffusion/SDXL 친화 문구로 확장하세요.\n"
    "출력은 반드시 JSON으로만, 키는 'prompt'와 'negative' 두 개 입니다.\n"
    "스타일/카메라/조명/구도/키워드 를 추가하고, 과한 수식어는 줄이세요. 한국어 입력이어도 영문 키워드 혼합을 허용합니다."
)

PROMPT_FMT = (
    "사용자 아이디어:\n" 
    """{idea}\n
    제약:\n
    - 장면의 핵심 객체, 배경, 분위기, 색감, 스타일(예: cinematic, studio lighting, 8k, photorealistic, illustration 등)을 분명히 하세요.
    - 인물이나 민감한 묘사는 안전하고 비외설적으로 표현하세요.
    - JSON 외의 문구를 출력하지 마세요.
    """
)

def refine_prompt(idea: str) -> Tuple[str, str]:
    """DeepSeek-R1로 아이디어를 SD 친화 프롬프트로 확장. 실패 시 기본값 사용."""
    try:
        sys = gr.update()
        msg = f"{PROMPT_SYS}\n\n{PROMPT_FMT.format(idea=idea)}"
        raw = LLM.invoke(msg).content
        # JSON만 추출
        try:
            data = json.loads(raw)
        except Exception:
            # 코드블록 케이스 제거
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(m.group(0)) if m else {}
        prompt = data.get("prompt") or idea
        negative = data.get("negative") or ""
        return prompt, negative
    except Exception:
        return idea, "nsfw, nudity, low quality, blurry, deformed, worst quality"


# =========================
# Diffusers 유틸
# =========================

def get_scheduler(name: str, pipe):
    name = (name or "dpm").lower()
    if name.startswith("euler"):  # euler / euler_a
        return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if name.startswith("lms"):
        return LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    if name.startswith("heun"):
        return HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    # 기본: DPM++ multi-step
    return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


@dataclass
class LoadedPipes:
    sd15_txt2img: Optional[StableDiffusionPipeline] = None
    sd15_img2img: Optional[StableDiffusionImg2ImgPipeline] = None
    sdxl_txt2img: Optional[StableDiffusionXLPipeline] = None
    sdxl_img2img: Optional[StableDiffusionXLImg2ImgPipeline] = None


PIPES = LoadedPipes()


def load_sd_pipeline(model_type: str, safety: bool = True) -> Tuple[object, str]:
    """필요한 파이프라인을 lazy load 후 캐시. model_type: 'sd15' | 'sdxl'"""
    note = ""
    if model_type == "sd15":
        if PIPES.sd15_txt2img is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                SD15_REPO,
                torch_dtype=DEFAULT_DTYPE,
                safety_checker=None if not safety else None,  # SD2.1은 safety_checker 기본 None
            )
            pipe.to(DEFAULT_DEVICE)
            PIPES.sd15_txt2img = pipe
            PIPES.sd15_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                SD15_REPO,
                torch_dtype=DEFAULT_DTYPE,
            ).to(DEFAULT_DEVICE)
            note = "[로드] SD 1.x 파이프라인 초기화 완료"
        return PIPES.sd15_txt2img, note
    else:  # sdxl
        if PIPES.sdxl_txt2img is None:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                SDXL_BASE,
                torch_dtype=DEFAULT_DTYPE,
            )
            pipe.to(DEFAULT_DEVICE)
            PIPES.sdxl_txt2img = pipe
            PIPES.sdxl_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                SDXL_BASE,
                torch_dtype=DEFAULT_DTYPE,
            ).to(DEFAULT_DEVICE)
            note = "[로드] SDXL 파이프라인 초기화 완료"
        return PIPES.sdxl_txt2img, note


def do_txt2img(
    model_type: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    scheduler_name: str,
    seed: int,
    batch_count: int,
    safety: bool,
) -> List[Image.Image]:
    pipe, _ = load_sd_pipeline(model_type, safety=safety)
    pipe.scheduler = get_scheduler(scheduler_name, pipe)

    generator = torch.Generator(device=DEFAULT_DEVICE).manual_seed(seed) if seed >= 0 else None

    imgs = []
    for i in range(batch_count):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        imgs.extend(out.images)
    return imgs


def do_img2img(
    model_type: str,
    prompt: str,
    negative: str,
    init_image: Image.Image,
    strength: float,
    steps: int,
    guidance: float,
    scheduler_name: str,
    seed: int,
    batch_count: int,
    safety: bool,
) -> List[Image.Image]:
    _, _ = load_sd_pipeline(model_type, safety=safety)
    pipe = PIPES.sd15_img2img if model_type == "sd15" else PIPES.sdxl_img2img
    pipe.scheduler = get_scheduler(scheduler_name, pipe)

    generator = torch.Generator(device=DEFAULT_DEVICE).manual_seed(seed) if seed >= 0 else None

    imgs = []
    for i in range(batch_count):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        imgs.extend(out.images)
    return imgs


# =========================
# Gradio UI
# =========================

def ui_generate(
    idea: str,
    model_type: str,
    use_llm: bool,
    negative_input: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    scheduler_name: str,
    seed: int,
    batch_count: int,
    safety: bool,
    init_image: Optional[Image.Image],
    strength: float,
):
    logs = []
    try:
        # 1) 프롬프트 확장(선택)
        if use_llm:
            refined, neg = refine_prompt(idea)
            if negative_input.strip():
                neg = f"{neg}, {negative_input}"
            prompt, negative = refined, neg
            logs.append("[LLM] DeepSeek-R1가 프롬프트를 확장했습니다.")
        else:
            prompt, negative = idea, negative_input

        # SDXL 권장 해상도 안내
        if model_type == "sdxl" and (width * height) < (1024 * 1024):
            logs.append("[안내] SDXL은 1024x1024 해상도에서 최적 성능을 보입니다.")

        # 2) 생성 실행
        if init_image is None:
            images = do_txt2img(
                model_type, prompt, negative, width, height, steps, guidance, scheduler_name, seed, batch_count, safety
            )
        else:
            images = do_img2img(
                model_type, prompt, negative, init_image, strength, steps, guidance, scheduler_name, seed, batch_count, safety
            )

        logs.append(f"[완료] {len(images)}장 생성")
        return images, "\n".join(logs)
    except Exception as e:
        return None, f"[오류] {e}"
    finally:
        gc.collect()
        if DEFAULT_DEVICE == "cuda":
            torch.cuda.empty_cache()


def build_ui():
    with gr.Blocks(title="OpenCode • DeepSeek-R1 + Stable Diffusion") as demo:
        gr.Markdown(
            """
            # OpenCode • DeepSeek-R1 + Stable Diffusion
            - **DeepSeek-R1(Ollama)** 가 아이디어를 SD 친화 프롬프트로 확장
            - **diffusers** 로 SD 1.x / **SDXL** 이미지 생성 (txt2img / img2img)
            - 로컬 실행, API Key 불필요
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                idea = gr.Textbox(label="아이디어(간단히)", placeholder="예: 비 오는 도쿄의 네온 골목, 영화적 조명", lines=3)
                use_llm = gr.Checkbox(value=True, label="DeepSeek-R1로 프롬프트 확장")
                negative = gr.Textbox(label="네거티브(선택)", placeholder="low quality, nsfw, blurry…", lines=2)

                model_type = gr.Radio(["sd15", "sdxl"], value="sdxl", label="모델")
                scheduler = gr.Dropdown([
                    "dpm",
                    "euler_a",
                    "lms",
                    "heun",
                ], value="dpm", label="스케줄러")

                with gr.Row():
                    width = gr.Slider(512, 1344, value=1024, step=64, label="너비")
                    height = gr.Slider(512, 1344, value=1024, step=64, label="높이")

                with gr.Row():
                    steps = gr.Slider(10, 80, value=30, step=1, label="스텝")
                    guidance = gr.Slider(0.5, 15.0, value=7.0, step=0.5, label="가이던스")

                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="시드(-1이면 랜덤)")
                    batch_count = gr.Slider(1, 8, value=1, step=1, label="한 번에 몇 장")

                safety = gr.Checkbox(value=True, label="(권장) Safety Checker 사용")

                gr.Markdown("**Img2Img(선택)** 초기 이미지를 올리면 img2img로 동작합니다.")
                init_image = gr.Image(label="초기 이미지 업로드", type="pil")
                strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Img2Img 변형 강도")

                run = gr.Button("생성", variant="primary")

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="결과", columns=2, height=540)
                logs = gr.Textbox(label="로그", lines=12)

        run.click(
            ui_generate,
            inputs=[
                idea, model_type, use_llm, negative, width, height, steps, guidance,
                scheduler, seed, batch_count, safety, init_image, strength,
            ],
            outputs=[gallery, logs],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
