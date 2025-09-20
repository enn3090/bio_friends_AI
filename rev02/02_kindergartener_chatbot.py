import gradio as gr
import ollama

"""
Ollama + DeepSeek-R1 모델을 활용한 유치원생 역할놀이 챗봇
-----------------------------------------------------
- OpenAI API 키 불필요 (로컬 실행)
- 대화 맥락에 따라 유치원생처럼 답변
- Gradio 인터페이스 제공
"""

# 시스템 프롬프트: 유치원생 캐릭터 지시
SYSTEM_PROMPT = "너는 유치원 학생이야. 유치원생처럼 답변해줘."


def ensure_system(messages):
    """메시지 맨 앞에 system 프롬프트를 보장"""
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def chat_with_kindergartener(user_input: str, history: list, state_messages: list):
    """
    유저 입력을 받아 DeepSeek-R1 모델을 호출하고 유치원생처럼 답변
    - history: (user, assistant) 튜플로 이루어진 대화 기록 (Gradio용)
    - state_messages: Ollama 모델에 전달하는 원시 메시지 리스트
    """
    # state_messages 초기화
    state_messages = state_messages or []
    ensure_system(state_messages)

    # 사용자 발화 추가
    state_messages.append({"role": "user", "content": user_input})

    # Ollama 호출
    response = ollama.chat(
        model="deepseek-r1:latest",
        messages=state_messages,
        options={"temperature": 0.9},
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
    말해보세요! 챗봇이 유치원생처럼 귀엽게 대답해줄 거예요.
    """)

    clear_btn = gr.Button("🧹 대화 초기화", variant="secondary")
    chatbot = gr.Chatbot(label="유치원생", height=480)
    state_messages = gr.State([])
    user_box = gr.Textbox(
        placeholder="예: 참새", label="입력", lines=1
    )

    # 입력 처리
    def on_submit(user_input, chat_hist, state_msgs):
        if not user_input or not user_input.strip():
            return gr.update(), state_msgs
        return chat_with_kindergartener(user_input.strip(), chat_hist, state_msgs)

    user_box.submit(
        fn=on_submit,
        inputs=[user_box, chatbot, state_messages],
        outputs=[chatbot, state_messages]
    ).then(
        lambda: gr.update(value=""),
        None,
        user_box
    )

    # 초기화 버튼
    clear_btn.click(
        fn=reset_history,
        inputs=None,
        outputs=[chatbot, state_messages]
    )

if __name__ == "__main__":
    demo.launch()