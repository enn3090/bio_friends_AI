# -*- coding: utf-8 -*-
"""
DeepSeek-R1 (Ollama) + Gradio
Veo 3용 고도화 JSON 프롬프트 생성기

- 로컬 PC에 Ollama와 deepseek-r1 모델이 설치되어 있다고 가정합니다.
  설치 예:
    1) https://ollama.com 다운로드/설치
    2) 터미널에서: ollama pull deepseek-r1
    3) ollama serve (일반적으로 자동 실행됨, 기본 포트 11434)

- 이 앱은 사용자의 '러프 동영상 아이디어' 문장을 입력받아
  Veo 3 스타일의 '구조화된 JSON 프롬프트'를 생성합니다.
- 출력은 항상 '유효한 단일 JSON 객체'가 되도록 강제합니다.

필요 패키지:
    pip install gradio requests
"""

import os
import re
import json
import time
import tempfile
from typing import Tuple, Any, Dict

import requests
import gradio as gr


# =========================
# Ollama(로컬 LLM) 호출 유틸
# =========================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "deepseek-r1")  # 로컬에 설치된 모델 이름


def _ollama_generate(prompt: str, temperature: float = 0.6, max_tokens: int = 2048, stop=None) -> str:
    """
    Ollama /api/generate 호출 (비스트리밍). deepseek-r1 모델을 사용합니다.
    - prompt: 단일 문자열 프롬프트 (system + user 지시 포함)
    - temperature, max_tokens: 생성 옵션
    - stop: 중단 토큰 리스트 (예: ["```"])
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_ctx": 8192,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "stop": stop or ["```", "</think>", "</thinking>", "<|eot_id|>"],
            # deepseek-r1이 사고흐름을 출력하지 않도록 유도
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.RequestException as e:
        raise RuntimeError(f"[Ollama 연결 오류] {e}\n"
                           f"- Ollama가 실행 중인지 확인하세요 (기본: {OLLAMA_URL}).\n"
                           f"- deepseek-r1 모델이 설치되어 있는지 확인하세요: `ollama pull deepseek-r1`") from e


# =========================
# 시스템 프롬프트(역할 지시)
# =========================

def build_system_instructions(structure_mode: str, force_no_text: bool) -> str:
    """
    사용자 러프 아이디어를 바탕으로, Veo 3용 고품질 JSON 프롬프트만 산출하도록 하는
    시스템 레벨 지시문을 구성합니다.
    """
    advanced_hint = ""
    if structure_mode == "Standard (Single Shot)":
        advanced_hint = (
            "Use ONLY the Standard JSON Structure.\n"
            "Output must be a single JSON object with the keys: "
            '["description","style","camera","lighting","environment","elements","motion","ending","text","keywords"].\n'
        )
    elif structure_mode == "Sequential (Multi-Part)":
        advanced_hint = (
            "Use ONLY the Sequential Shot Structure.\n"
            "Output must be a single JSON object with keys at minimum: "
            '["scene","style","sequence","audio"].\n'
            "The 'sequence' must be an array of objects describing continuous shots and transitions without hard cuts.\n"
        )
    elif structure_mode == "Timeline (Timestamped)":
        advanced_hint = (
            "Use ONLY the Timeline-Based Structure.\n"
            "Output must be a single JSON object with keys at minimum: "
            '["metadata","camera_setup","key_elements","timeline"].\n'
            'Include "aspect_ratio" inside "metadata".\n'
        )
    else:
        # Auto: 모델이 아이디어 복잡도에 따라 최적 구조를 선택
        advanced_hint = (
            "Choose the best-fitting structure (Standard / Sequential / Timeline) based on user intent.\n"
            "Always output a SINGLE JSON object representing the chosen structure.\n"
        )

    text_rule = 'The "text" field MUST be "none".\n' if force_no_text else \
        'If no overlay is needed, set "text" to "none".\n'

    # 원문 지시를 그대로 포함하여 LLM이 잊지 않도록 고정
    core_instruction = r"""
Agent Instructions: Generating High-Quality Prompts for Veo 3
Your primary goal is to generate a structured and highly descriptive prompt in JSON format. This allows the AI to parse distinct creative and technical elements for precise video generation. The most successful prompts follow a pattern of "object-centric magical transformation," where a single product or object triggers a dynamic and cinematic unfolding of a larger scene.

There are two primary structures to use: the Standard JSON Structure for single, continuous shots, and the Advanced Sequential/Timeline Structure for more complex, multi-part scenes.

1. The Standard JSON Structure
This is the default and most common format. Use it for concepts that can be captured in a single, continuous camera motion, even if the action within the frame is complex.

Your output must be a single JSON object with the following keys. Be descriptive and evocative in your values.

"description": (String) A comprehensive, one-paragraph summary of the entire video concept from start to finish. This is the high-level narrative.
"style": (String) Overall visual aesthetic and mood.
"camera": (String) Angle, movement, and lens style (filmmaking terminology).
"lighting": (String) Lighting style, time of day, and color palette.
"environment": (String) Setting/location; static or transforming.
"elements": (Array of Strings) Key nouns/visual components.
"motion": (String) Sequence of actions/transformations (A > B > C).
"ending": (String) Final shot/lasting image.
"text": (String) Overlay text; if none, explicitly "none".
"keywords": (Array of Strings) Tags including aspect ratio ("16:9"), core subject, actions, style, technical attributes.

2. Advanced Structures (Use When Necessary)
A) Sequential Shot Structure (multi-part, continuous motion without hard cuts):
{
  "scene": "animation",
  "style": "Futuristic Apple-style minimalism, photorealistic...",
  "sequence": [
    {"shot": "Logo Reveal", "camera": "slow push-in", "description": "Begin with the brand logo floating..."},
    {"transition": "Without any cut, the camera smoothly moves closer..."},
    {"shot": "Product Formation", "camera": "continuous motion, no cut", "description": "The particles condense and materialize..."}
  ],
  "audio": {"soundtrack": "Soft, futuristic ambient music..."}
}

B) Timeline-Based Structure (precise timing; timestamps and beats):
{
  "metadata": {"prompt_name": "NYC City Assembly", "base_style": "cinematic, photorealistic, 4K", "aspect_ratio": "16:9"},
  "camera_setup": "A single, fixed, wide-angle shot...",
  "key_elements": {...},
  "timeline": [
    {"sequence": 1, "timestamp": "00:00-00:01", "action": "In the center of the barren plaza...", "audio": "Deep, resonant rumble..."},
    {"sequence": 2, "timestamp": "00:01-00:02", "action": "The container's steel doors burst open...", "audio": "Sharp metallic clang..."}
  ]
}

Guiding Principles:
- ALWAYS OUTPUT VALID JSON: a single well-formed JSON object. No explanations or comments.
- Embrace "magical realism / assembly" centered on an object catalyst.
- Think like a director: dolly, crane, orbit, low angle, golden hour, lens flare.
- Be hyper-specific and sensory. Mention condensation, glistening surfaces, steam, sparks, textures.
- Deconstruct the motion: describe the chain precisely (A > B > C).
- Use "keywords" to reinforce concepts including "16:9" or another aspect ratio plus style/tech specs (e.g., 4K).
"""

    # 최종 시스템 지시 조합
    system = (
        "You are a prompt-engineering director for Veo 3 video generation.\n"
        "Your job: Return ONLY a single valid JSON object tailored to the user's idea.\n"
        "Do NOT include any prose before or after the JSON. No markdown fences.\n"
        f"{advanced_hint}"
        f"{text_rule}"
        + core_instruction
    )
    return system


# =========================
# JSON 정제/검증 유틸
# =========================

def extract_first_json_blob(text: str) -> str:
    """
    LLM이 혹시라도 주변 텍스트를 섞어 보냈을 경우, 첫 번째 JSON 객체 블록만 추출.
    중괄호 괄호수 카운팅으로 가장 그럴듯한 JSON을 복구 시도.
    """
    # 빠른 경로: 이미 올바른 JSON일 수 있음
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 코드블록 제거
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()

    # 중괄호 균형으로 첫 객체 찾기
    start_indices = [m.start() for m in re.finditer(r"\{", text)]
    for start in start_indices:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
    # 실패 시 원문 반환 (후속 단계에서 예외 표시)
    return text


def ensure_json_object(structure_mode: str, data: Any) -> Dict[str, Any]:
    """
    구조 모드에 맞는 최소 키가 있는지 확인하고, 부족하면 기본 값을 채워 넣습니다.
    - 모델 출력이 살짝 누락해도 UI가 깨지지 않도록 가벼운 보정만 수행합니다.
    """
    if not isinstance(data, dict):
        raise ValueError("LLM 출력이 JSON 객체가 아닙니다.")

    if structure_mode == "Standard (Single Shot)" or (
        structure_mode == "Auto (Let Model Decide)" and all(k not in data for k in ["scene", "metadata"])
    ):
        # 표준 구조 최소 키
        defaults = {
            "description": "",
            "style": "",
            "camera": "",
            "lighting": "",
            "environment": "",
            "elements": [],
            "motion": "",
            "ending": "",
            "text": "none",
            "keywords": ["16:9", "cinematic", "4K"]
        }
        for k, v in defaults.items():
            data.setdefault(k, v)

    elif structure_mode == "Sequential (Multi-Part)" or ("sequence" in data and isinstance(data.get("sequence"), list)):
        data.setdefault("scene", "animation")
        data.setdefault("style", "")
        data.setdefault("sequence", [])
        data.setdefault("audio", {"soundtrack": ""})

    elif structure_mode == "Timeline (Timestamped)" or ("timeline" in data and isinstance(data.get("timeline"), list)):
        meta = data.setdefault("metadata", {})
        meta.setdefault("prompt_name", "Untitled Prompt")
        meta.setdefault("base_style", "cinematic, photorealistic, 4K")
        meta.setdefault("aspect_ratio", "16:9")
        data.setdefault("camera_setup", "")
        data.setdefault("key_elements", {})
        data.setdefault("timeline", [])

    return data


# =========================
# 프롬프트 결합
# =========================

def compose_prompt(system_msg: str, user_idea: str) -> str:
    """
    Ollama의 단일 프롬프트에 system/user 역할을 함께 넣는 포맷.
    deepseek-r1이 사고흐름을 출력하지 않도록 '최종 JSON만' 강조.
    """
    return (
        "<system>\n" + system_msg.strip() + "\n</system>\n"
        "<user>\n"
        "Below is the user's rough video concept. Transform it into a SINGLE valid JSON object, "
        "following the above rules and structures. Be richly descriptive, cinematic, and sensory. "
        "Do NOT include explanations. JSON only.\n\n"
        f"User Idea:\n{user_idea.strip()}\n"
        "</user>\n"
        "<assistant>\n"
    )


# =========================
# Gradio 액션 함수
# =========================

def generate_json_prompt(user_idea: str,
                         structure_mode: str = "Auto (Let Model Decide)",
                         force_no_text: bool = True,
                         temperature: float = 0.6,
                         max_tokens: int = 2048) -> Tuple[Dict[str, Any], str]:
    """
    Gradio에서 호출되는 메인 함수.
    - user_idea: 사용자의 러프 아이디어(한 문장 이상)
    - structure_mode: 표준/시퀀셜/타임라인/자동
    - force_no_text: 텍스트 오버레이를 강제로 "none"으로
    - temperature, max_tokens: LLM 생성 옵션
    반환:
      (JSON 객체(dict), 디버그 원문 텍스트)
    """
    if not user_idea or not user_idea.strip():
        raise gr.Error("러프 동영상 아이디어를 입력하세요.")

    system_msg = build_system_instructions(structure_mode, force_no_text)
    prompt = compose_prompt(system_msg, user_idea)

    # LLM 호출
    raw = _ollama_generate(prompt, temperature=temperature, max_tokens=max_tokens)

    # JSON만 추출/검증
    blob = extract_first_json_blob(raw)
    try:
        data = json.loads(blob)
    except Exception as e:
        # 한 번 더 엄격 지시로 재시도 (자동 복구)
        stricter = system_msg + "\nSTRICT RULE: Output ONLY a single valid JSON object. No extra text."
        prompt2 = compose_prompt(stricter, user_idea)
        raw2 = _ollama_generate(prompt2, temperature=temperature, max_tokens=max_tokens)
        blob2 = extract_first_json_blob(raw2)
        try:
            data = json.loads(blob2)
            raw = raw2
            blob = blob2
        except Exception:
            raise gr.Error(f"모델 출력이 JSON 형식이 아닙니다.\n--- 원문 ---\n{raw[:1500]}") from e

    # 구조에 맞는 최소 키 보정
    data = ensure_json_object(structure_mode, data)

    # 텍스트 강제 옵션 적용
    if force_no_text and isinstance(data, dict):
        if "text" in data and isinstance(data["text"], str):
            data["text"] = "none"

    # 깔끔하게 정렬된 JSON과 디버그 원문을 반환
    return data, blob


def save_json_to_file(data: Dict[str, Any]) -> str:
    """
    다운로드 버튼용 JSON 파일 생성.
    """
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp.name, "w", encoding="utf-8") as f:
        f.write(pretty)
    return tmp.name


# =========================
# Gradio UI
# =========================

with gr.Blocks(theme=gr.themes.Soft(), title="Veo 3 JSON Prompt Generator (DeepSeek-R1 · Ollama)") as demo:
    gr.Markdown(
        """
# Veo 3 JSON 프롬프트 생성기
**DeepSeek-R1 (Ollama, 로컬) + Gradio**

- 사용법: 러프한 동영상 아이디어를 입력하면, Veo 3용 고도화된 **단일 JSON 객체**를 생성합니다.  
- 구조 선택:
  - **Standard**: 한 번의 연속 쇼트(카메라 모션은 복잡해도 컷 없이)인 경우
  - **Sequential**: 컷 없이 이어지는 다중 단계/쇼트
  - **Timeline**: 타임스탬프별 정밀 제어 (사운드 큐 동기화 등)
  - **Auto**: 모델이 아이디어에 맞춰 최적 구조를 선택
        """
    )

    with gr.Row():
        structure_mode = gr.Dropdown(
            choices=[
                "Auto (Let Model Decide)",
                "Standard (Single Shot)",
                "Sequential (Multi-Part)",
                "Timeline (Timestamped)",
            ],
            value="Auto (Let Model Decide)",
            label="구조 선택"
        )
        force_no_text = gr.Checkbox(value=True, label='텍스트 오버레이를 항상 "none"으로', info="브랜드 텍스트 금지 시 유용")

    user_idea = gr.Textbox(
        label="러프 동영상 아이디어 입력",
        placeholder="예) 차가운 콜라 캔이 테이블 중앙에서 서서히 성에가 맺히고, 탭이 천천히 열리며 주변이 네온 도시로 변모하는 마법적 변환...",
        lines=6
    )

    with gr.Accordion("고급 옵션", open=False):
        temperature = gr.Slider(0.0, 1.2, value=0.6, step=0.05, label="창의성 (temperature)")
        max_tokens = gr.Slider(256, 4096, value=2048, step=64, label="최대 토큰(힌트)")

    with gr.Row():
        gen_btn = gr.Button("🚀 프롬프트 생성", variant="primary")
        sample_btn = gr.Button("✨ 예시 아이디어 채우기")

    with gr.Row():
        json_view = gr.JSON(label="생성된 JSON (검증/보정 완료)")
        raw_view = gr.Code(label="LLM 원문(JSON만 추출)", language="json")

    download_btn = gr.DownloadButton(label="⬇️ JSON 다운로드", file_name="veo3_prompt.json")

    # 이벤트 바인딩
    def on_generate(user_idea, structure_mode, force_no_text, temperature, max_tokens):
        data, raw = generate_json_prompt(
            user_idea=user_idea,
            structure_mode=structure_mode,
            force_no_text=force_no_text,
            temperature=temperature,
            max_tokens=int(max_tokens),
        )
        return data, raw, gr.update(value=save_json_to_file(data))

    gen_btn.click(
        on_generate,
        inputs=[user_idea, structure_mode, force_no_text, temperature, max_tokens],
        outputs=[json_view, raw_view, download_btn]
    )

    def fill_sample():
        return (
            "탁자 중앙의 유리 병(무표기)이 차갑게 빛나며 미세한 물방울이 맺힌다. 병의 마개가 "
            "딱 소리를 내며 천천히 떠오르고, 병 입구에서 미세한 금빛 입자들이 분출되어 공중에서 "
            "리본처럼 소용돌이친다. 순간 방 전체가 황금빛 석양 톤으로 물들며 벽이 사라지고, "
            "넓은 모래 사막과 현대적 도시의 스카이라인이 이어진 초현실 공간으로 연속 변환된다. "
            "카메라는 낮은 로우 앵글의 느린 오비트로 시작해 상공 탑다운으로 연결되며, 입자들이 "
            "유리병 라벨 형태를 스스로 조립했다가 도시 전체의 네온 축제로 증폭된다. 텍스트 오버레이는 불필요."
        )

    sample_btn.click(fill_sample, outputs=[user_idea])


if __name__ == "__main__":
    # 공유 모드 필요 시 share=True
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

