import gradio as gr
import ollama

"""
Ollama + DeepSeek-R1 모델을 활용한 유치원생 역할놀이 챗봇 (연속 대화 + 예시 버튼)
--------------------------------------------------------------------------
- OpenAI API, dotenv, API 키(sk-...) 사용 없음. (완전 로컬)
- Ollama 로컬 서버와 deepseek-r1 모델을 사용합니다. (사전 준비: `ollama pull deepseek-r1:latest`)
- Gradio UI 제공: 히스토리 유지, Temperature 슬라이더, 빠른 예시 버튼(참새/말/개구리/뱀/오리), 초기화 버튼.
"""

# 시스템 프롬프트: 유치원생 캐릭터 지시
SYSTEM_PROMPT = "너는 유치원 학생이야. 유치원생처럼 답변해줘."


def ensure_system(messages):
    """메시지 맨 앞에 system 프롬프트를 보장"""
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def chat_with_kindergartener(user_input: str, history: list, state_messages: list, temperature: float):
    """
    유저 입력을 받아 DeepSeek-R1 모델을 호출하고 유치원생처럼 답변합니다.
    - history: (user, assistant) 튜플로 이루어진 대화 기록 (Gradio용)
    - state_messages: Ollama 모델에 전달하는 원시 메시지 리스트 (역할/내용)
    - temperature: 생성 창의성 조절 값
    """
    # state_messages 초기화 및 시스템 프롬프트 보장
    state_messages = state_messages or []
    ensure_system(state_messages)

    # 사용자 발화 추가
    state_messages.append({"role": "user", "content": user_input})

    # Ollama + deepseek-r1 호출
    response = ollama.chat(
        model="deepseek-r1:latest",
        messages=state_messages,
        options={"temperature": float(temperature)},
    )

    assistant_text = response["message"]["content"]

    # Gradio 히스토리 및 모델 메시지 갱신
    history = history + [(user_input, assistant_text)]
    state_messages.append({"role": "assistant", "content": assistant_text})

    return history, state_messages


def reset_history():
    """대화 초기화"""
    return [], []


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    ## 🧒 유치원생 역할놀이 챗봇
    동물 이름이나 간단한 말을 해보세요! 챗봇이 유치원생처럼 귀엽게 대답합니다.
    """)

    with gr.Row():
        clear_btn = gr.Button("🧹 대화 초기화", variant="secondary")
        temperature = gr.Slider(0.0, 1.5, value=0.9, step=0.1, label="창의성(Temperature)")

    chatbot = gr.Chatbot(label="유치원생", height=480, show_copy_button=True)
    state_messages = gr.State([])

    # 빠른 예시 버튼들 (원문의 시나리오 반영: 참새/말/개구리/뱀 + 이번 요청의 오리)
    with gr.Row():
        ex_bird = gr.Button("참새 🐦")
        ex_horse = gr.Button("말 🐴")
        ex_frog = gr.Button("개구리 🐸")
        ex_snake = gr.Button("뱀 🐍")
        ex_duck = gr.Button("오리 🦆")

    user_box = gr.Textbox(placeholder="예: 오리", label="입력", lines=1)

    # 제출 핸들러
    def on_submit(user_input, chat_hist, state_msgs, temp):
        if not user_input or not user_input.strip():
            return gr.update(), state_msgs
        return chat_with_kindergartener(user_input.strip(), chat_hist, state_msgs, temp)

    user_box.submit(
        fn=on_submit,
        inputs=[user_box, chatbot, state_messages, temperature],
        outputs=[chatbot, state_messages]
    ).then(lambda: gr.update(value=""), None, user_box)

    # 예시 버튼 핸들러들 (각 버튼은 지정 텍스트로 on_submit을 호출)
    for btn, text in [
        (ex_bird, "참새"),
        (ex_horse, "말"),
        (ex_frog, "개구리"),
        (ex_snake, "뱀"),
        (ex_duck, "오리"),
    ]:
        btn.click(
            fn=on_submit,
            inputs=[gr.State(text), chatbot, state_messages, temperature],
            outputs=[chatbot, state_messages]
        )

    # 초기화 버튼
    clear_btn.click(fn=reset_history, inputs=None, outputs=[chatbot, state_messages])

    gr.Markdown(
        """
        **TIP**: 같은 동물 이름을 여러 번 입력하면, 이전 문맥을 이어받아 더 자연스러운 말투로 변형될 수 있어요.
        """
    )

if __name__ == "__main__":
    demo.launch()