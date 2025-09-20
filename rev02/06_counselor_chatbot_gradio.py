# ------------------------------------------------------------
# OpenAI API → Ollama + DeepSeek-R1 변환 (Gradio UI 버전)
# ------------------------------------------------------------
# ✅ 특징
#   - 로컬에 설치된 Ollama의 deepseek-r1 모델 사용 (API Key 불필요)
#   - Gradio 기반 대화형 UI 제공
#   - 시스템 프롬프트: "너는 사용자를 도와주는 상담사야."
#   - temperature(창의성) 조절 가능
# ------------------------------------------------------------
# 준비물
#   pip install gradio ollama
#   (Ollama 및 deepseek-r1 모델은 로컬에 설치돼 있어야 합니다)
#   예) 모델 설치:  ollama pull deepseek-r1
# ------------------------------------------------------------

import gradio as gr
import ollama

# -------------------------
# 설정: 시스템 프롬프트 & 옵션
# -------------------------
SYSTEM_PROMPT = "너는 사용자를 도와주는 상담사야."
# Ollama의 sampling 옵션 (OpenAI의 temperature와 동일한 개념)
OLLAMA_OPTIONS = {
    "temperature": 0.9,  # 창의성 정도 (0.0 ~ 2.0 권장)
}

# -------------------------
# 모델 호출 함수
# -------------------------

def call_llm(messages):
    """
    Ollama deepseek-r1와 대화하여 마지막 assistant 응답 텍스트를 반환합니다.
    messages: [{role: "system"|"user"|"assistant", content: str}, ...]
    """
    response = ollama.chat(
        model="deepseek-r1",
        messages=messages,
        options=OLLAMA_OPTIONS,
    )
    return response["message"]["content"]


# -------------------------
# Gradio 이벤트 핸들러
# -------------------------

def on_submit(user_message, chat_history, msg_state):
    """
    - user_message: 사용자가 입력한 텍스트 (str)
    - chat_history: Gradio Chatbot에 표시되는 [(user, bot), ...] 형식의 리스트
    - msg_state: Ollama에 전달할 전체 메시지 히스토리 (system 포함)
    반환값: (업데이트된 chat_history, 업데이트된 msg_state, 입력창 초기화)
    """
    if not user_message or user_message.strip() == "":
        # 공백 입력 방지
        return gr.update(), msg_state, gr.update(value="")

    # 최초 호출 시 system 메시지 초기화
    if not msg_state:
        msg_state = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 사용자 메시지 추가
    msg_state.append({"role": "user", "content": user_message})

    # 모델 호출
    assistant_text = call_llm(msg_state)

    # 모델 응답을 히스토리와 상태에 반영
    msg_state.append({"role": "assistant", "content": assistant_text})

    # Gradio에 표시될 대화 목록에 추가
    chat_history = chat_history + [(user_message, assistant_text)]

    # 입력창 비우기
    return chat_history, msg_state, gr.update(value="")


def on_clear():
    """대화 전체 초기화 (화면, 내부 상태 둘 다)"""
    empty_hist = []
    # 시스템 프롬프트만 남긴 초기 상태로 리셋하고 싶다면 아래처럼 변경하세요:
    # init_state = [{"role": "system", "content": SYSTEM_PROMPT}]
    init_state = []  # 사용자가 메시지를 보낼 때 시스템 프롬프트가 주입되도록 비워둠
    return empty_hist, init_state


# -------------------------
# Gradio UI 구성
# -------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 DeepSeek-R1 상담 챗봇 (로컬 Ollama)
    - 시스템 프롬프트: **너는 사용자를 도와주는 상담사야.**
    - 모델: `deepseek-r1` (로컬)
    - 옵션: `temperature=0.9`
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="상담 대화",
                type="messages",  # 메시지형 내부 구조 사용
                height=480,
                avatar_images=(None, None),
            )
            with gr.Row():
                user_in = gr.Textbox(
                    placeholder="메시지를 입력하고 Enter를 누르세요...",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("보내기", variant="primary")
                clear_btn = gr.Button("초기화", variant="secondary")
        with gr.Column(scale=2):
            gr.Markdown("""
            ### ⚙️ 설정
            - **Temperature**는 현재 코드 상수로 0.9로 설정되어 있습니다.
            - 필요시 `OLLAMA_OPTIONS`를 수정해 다양한 샘플링 옵션을 적용하세요.
            
            #### 추가 팁
            - 모델 교체: `model="deepseek-r1"` → `deepseek-r1:32b` 등
            - 시스템 프롬프트 변경: `SYSTEM_PROMPT` 수정
            - 스트리밍이 필요하다면 `ollama.chat` 대신 `ollama.generate(stream=True, ...)` 패턴을 적용할 수 있습니다.
            """)

    # 내부 상태: Ollama에 보낼 원시 메시지 형식 저장
    msg_state = gr.State([])  # [{role, content}, ...]

    # 이벤트 연결 (Enter 및 버튼)
    user_in.submit(on_submit, [user_in, chatbot, msg_state], [chatbot, msg_state, user_in])
    send_btn.click(on_submit, [user_in, chatbot, msg_state], [chatbot, msg_state, user_in])

    # 초기화 버튼
    clear_btn.click(on_clear, outputs=[chatbot, msg_state])


# -------------------------
# 앱 실행
# -------------------------
if __name__ == "__main__":
    # 공유 필요 시: demo.launch(share=True)
    demo.launch()