"""
[OpenCode 변환본]

- 목적: OpenAI(Closed LLM) + Streamlit 기반 코드를, **로컬 오픈소스 LLM(Ollama의 DeepSeek-R1)** + **Gradio UI**로 변환
- 특징:
  1) API Key 불필요 (로컬 Ollama 서버 사용, 기본: http://localhost:11434)
  2) 모델: deepseek-r1 (원하는 태그로 교체 가능, 예: "deepseek-r1:latest" 또는 "deepseek-r1:14b")
  3) 간단한 **툴 호출(get_current_time)** 데모 포함
     - LLM이 JSON 형식으로 {"tool": "get_current_time", "args": {...}} 를 출력하면
       실제 파이썬 함수를 실행하고 결과를 다시 모델에 제공한 뒤 최종 답변을 생성
  4) **Gradio ChatInterface** 로 대화형 UI 제공
  5) DeepSeek-R1의 <think> 내부 사고 텍스트를 사용자 출력에서 자동 제거

사전 준비:
1) Ollama 설치: https://ollama.com
2) 모델 다운로드:
   - 터미널에서: `ollama pull deepseek-r1:latest`
     (또는 원하는 버전/사이즈로 pull)
3) (선택) gpt_functions.py 파일에 다음과 같이 준비되어 있다고 가정:
   - get_current_time(timezone: str) -> str
   - tools: List[dict]  # 각 툴의 이름/설명/매개변수 스키마 정보 (문서화를 위해 시스템 프롬프트에 사용)

실행:
`python app.py` 로 실행하면 Gradio UI가 브라우저에서 열립니다.
"""

import json
import re
from typing import Any, Dict, List, Optional

import gradio as gr

# Ollama 파이썬 클라이언트 (pip install ollama)
# 로컬에서 ollama serve 가 동작 중이어야 합니다.
import ollama

# 사용자가 제공한 함수/툴 정의 불러오기 (없을 경우를 대비해 안전하게 처리)
try:
    from gpt_functions import get_current_time, tools as REGISTERED_TOOLS
except Exception:
    # 예비(fallback) 구현: gpt_functions 모듈이 없는 경우에도 동작하도록
    import datetime
    import zoneinfo

    def get_current_time(timezone: str = "UTC") -> str:
        try:
            tz = zoneinfo.ZoneInfo(timezone)
        except Exception:
            tz = zoneinfo.ZoneInfo("UTC")
        now = datetime.datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")

    REGISTERED_TOOLS = [
        {
            "name": "get_current_time",
            "description": "지정한 타임존의 현재 시간을 문자열로 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "예: Asia/Seoul, UTC, America/Los_Angeles",
                    }
                },
                "required": ["timezone"],
            },
        }
    ]


# ====== 설정 ======
MODEL_NAME = "deepseek-r1:latest"  # 필요 시 "deepseek-r1:14b" 등으로 교체
SYSTEM_PROMPT = """너는 사용자를 도와주는 상담사야.

아래 '도구 사용 규칙'을 반드시 따라:
[도구 사용 규칙]
- 네가 외부 도구가 필요하다고 판단하면 **반드시 JSON 한 줄**로만 응답해.
- JSON 스키마:
  {"tool": "<tool_name>", "args": {"key": "value", ...}}
- 도구가 필요 없으면:
  {"tool": null, "answer": "<사용자에게 보여줄 최종 답변>"}
- JSON 이외의 추가 텍스트/설명/마크다운을 절대 섞지 마.

[사용 가능 도구]
{}
""".format(
    json.dumps(REGISTERED_TOOLS, ensure_ascii=False, indent=2)
)


# ====== 유틸: DeepSeek-R1의 <think> 토큰 제거 ======
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(content: str) -> str:
    """DeepSeek-R1이 생성하는 내부 사고(<think>...</think>)를 제거하여 사용자에게 깔끔하게 보여준다."""
    return THINK_TAG_PATTERN.sub("", content).strip()


# ====== 유틸: 모델 응답에서 JSON 찾아 파싱 ======
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def extract_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """
    모델이 출력한 텍스트에서 JSON 오브젝트를 추출/파싱.
    - 모델이 순수 JSON만 내놓는 것이 이상적이지만, 안전하게 정규식으로 첫 JSON 블록을 잡아본다.
    """
    # 우선 전체를 JSON으로 시도
    try:
        return json.loads(text)
    except Exception:
        pass

    # 블록으로 검색
    m = JSON_BLOCK_PATTERN.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ====== LLM 호출 ======
def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Ollama 로컬 모델에 대화 메시지를 전달하고 응답 텍스트를 반환한다.
    messages 형식: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    resp = ollama.chat(model=MODEL_NAME, messages=messages)
    return resp.get("message", {}).get("content", "")


# ====== 툴 실행 라우터 ======
def run_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """
    지원하는 툴을 실행하고 문자열 결과를 반환.
    필요한 만큼 여기에 if/elif로 추가 가능.
    """
    if tool_name == "get_current_time":
        tz = args.get("timezone", "UTC")
        return get_current_time(timezone=tz)
    # 확장 지점: 다른 툴 추가
    raise ValueError(f"지원하지 않는 도구: {tool_name}")


# ====== 대화 상태 관리 ======
def build_system_message() -> Dict[str, str]:
    return {"role": "system", "content": SYSTEM_PROMPT}


def history_to_messages(history: List[List[str]]) -> List[Dict[str, str]]:
    """
    Gradio의 history 형식을 Ollama chat 메시지로 변환.
    history: [[user, assistant], [user, assistant], ...]
    """
    msgs = [build_system_message()]
    for user_msg, assistant_msg in history:
        if user_msg:
            msgs.append({"role": "user", "content": user_msg})
        if assistant_msg:
            msgs.append({"role": "assistant", "content": assistant_msg})
    return msgs


# ====== 메인 응답 로직 ======
def respond(user_input: str, history: List[List[str]]) -> str:
    """
    1) 사용자 입력을 포함해 LLM에 질의
    2) LLM이 툴 호출 JSON을 내면 툴 실행 → 결과를 컨텍스트에 추가 → 최종 답변 재요청
    3) 최종 답변에서 <think> 제거 후 반환
    """
    # 1) 1차 호출
    messages = history_to_messages(history)
    messages.append({"role": "user", "content": user_input})

    first = call_llm(messages)
    parsed = extract_tool_json(first)

    # 2) 툴 호출 분기
    if parsed and isinstance(parsed, dict) and parsed.get("tool") is not None:
        # 도구 호출 케이스
        tool_name = parsed.get("tool")
        args = parsed.get("args", {}) or {}

        try:
            tool_result = run_tool(tool_name, args)
        except Exception as e:
            # 도구 호출 실패 시 모델에게 오류를 알려 재시도/설명 유도
            tool_result = f"[도구 실행 오류] {e}"

        # 도구 실행 결과를 모델에게 '사실 정보'로 제공 후 최종 답변을 요청
        tool_feedback = (
            f"도구 `{tool_name}` 실행 결과:\n"
            f"{json.dumps(tool_result, ensure_ascii=False)}\n\n"
            "위 결과를 바탕으로 사용자에게 도움 되는 최종 답변만 자연스럽게 한국어로 작성하세요. "
            "JSON 형식은 사용하지 마세요."
        )
        messages.append({"role": "assistant", "content": first})  # 모델의 툴 호출 JSON도 히스토리에 남김
        messages.append({"role": "system", "content": tool_feedback})

        final = call_llm(messages)
        return strip_think(final)

    # 3) 도구 미사용(바로 답변) 케이스
    if parsed and parsed.get("tool") is None:
        # {"tool": null, "answer": "..."} 형식 준수 시
        answer = parsed.get("answer", "")
        return strip_think(answer or "")

    # 모델이 형식을 안 지킨 경우도 방어적으로 처리
    return strip_think(first or "답변을 생성하지 못했어요. 다시 시도해 주세요.")


# ====== Gradio UI 구성 ======
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 💬 Chatbot (DeepSeek-R1 on Ollama + Gradio)")

    chat = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            label="상담사",
            height=520,
            avatar_images=(None, None),  # 필요 시 아바타 이미지 경로 지정 가능
        ),
        title="",
        description=(
            "로컬 **Ollama** 의 **DeepSeek-R1** 모델을 사용합니다. "
            "필요 시 `get_current_time` 도구를 자동 호출해 현재 시간을 알려줄 수 있어요."
        ),
        theme="soft",
        retry_btn="다시 생성",
        undo_btn="이전 메시지 삭제",
        clear_btn="대화 초기화",
        additional_inputs=[],
        examples=[
            "안녕하세요, 오늘 일정 정리 도와줄 수 있어요?",
            "서울 시간으로 지금 몇 시야?",
            "스트레스 관리 팁 알려줘.",
        ],
    )

    # 초기 시스템 안내 출력 (옵션)
    with gr.Accordion("시스템/모델 정보", open=False):
        gr.Markdown(
            f"""
- 사용 모델: **{MODEL_NAME}**  
- 로컬 서버: **http://localhost:11434** (Ollama 기본)  
- 사용 가능 도구: `get_current_time(timezone: str)`  
- 도구 호출 프로토콜:  
  `{{"tool": "<tool_name>", "args": {{...}}}}` 또는 `{{"tool": null, "answer": "..."}}`
"""
        )

if __name__ == "__main__":
    # share=True 로 외부 접근 가능(테스트용). 내부망만 사용할 땐 False.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

