import gradio as gr
import ollama

"""
연속 대화(대화 히스토리 유지) + Gradio UI + Ollama(DeepSeek-R1)
-----------------------------------------------------------------
- OpenAI API 키(sk-...)를 전혀 사용하지 않습니다.
- 로컬에 설치된 Ollama + deepseek-r1 모델을 사용합니다.
- 사용자가 보낸 이전 대화를 모두 유지하여 문맥 기반 답변을 생성합니다.
- 온도(창의성) 슬라이더 제공.
- "초기화" 버튼으로 히스토리/상태를 완전히 리셋할 수 있습니다.

사전 준비(터미널):
  1) Ollama 설치 후 실행
  2) deepseek-r1 모델 준비:  `ollama pull deepseek-r1:latest`
  3) 이 스크립트 실행:        `python app.py`
"""

# 대화의 기본 성격을 규정하는 시스템 프롬프트
SYSTEM_PROMPT = (
    "너는 백설공주 이야기 속의 마법 거울이야. "
    "그 이야기 속의 마법 거울의 캐릭터에 부합하게, 품위 있고 운율감 있는 말투로 답변해줘."
)


def ensure_system(messages):
    """messages(list[dict])에 system 역할 프롬프트가 없다면 삽입합니다."""
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def chat_with_mirror(user_input: str, history: list, state_messages: list, temperature: float):
    """
    - user_input: 사용자가 방금 입력한 텍스트
    - history: Gradio Chatbot이 화면에 보여줄 (user, assistant) 튜플 리스트
    - state_messages: Ollama로 보낼 원시 메시지 리스트(역할/콘텐츠 딕셔너리). 대화 컨텍스트 저장용.
    - temperature: 창의성 조절 값

    반환:
    - 업데이트된 history(list[tuple])
    - 업데이트된 state_messages(list[dict])
    """
    # 1) state_messages에 시스템 프롬프트 보장
    state_messages = state_messages or []
    ensure_system(state_messages)

    # 2) 사용자 발화 추가
    state_messages.append({"role": "user", "content": user_input})

    # 3) Ollama + DeepSeek-R1 호출
    #    (옵션에 temperature를 전달해 생성 창의성 조절)
    response = ollama.chat(
        model="deepseek-r1:latest",
        messages=state_messages,
        options={
            "temperature": float(temperature),
        },
    )

    assistant_text = response["message"]["content"]

    # 4) 히스토리(화면 표시용)와 state_messages(모델 컨텍스트용) 동기화
    history = history + [(user_input, assistant_text)]
    state_messages.append({"role": "assistant", "content": assistant_text})

    return history, state_messages


def reset_history():
    """대화 초기화: 화면 히스토리와 내부 상태를 모두 비웁니다."""
    return [], []


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    ## 🪞 마법 거울과의 대화 (연속 대화 지원)
    백설공주 이야기 속 *마법 거울*에게 무엇이든 물어보세요. 이전 대화를 기억하여 더 자연스럽게 이어서 답합니다.
    """)

    with gr.Row():
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.5,
            value=0.9,
            step=0.1,
            label="창의성(Temperature)",
            info="값이 높을수록 더 창의적이고 예측 불가한 답변"
        )
        clear_btn = gr.Button("🧹 대화 초기화", variant="secondary")

    chatbot = gr.Chatbot(
        label="마법 거울",
        avatar_images=(None, None),
        bubble_full_width=False,
        height=480,
        show_copy_button=True,
         type='messages'
    )

    # state_messages: Ollama로 보낼 원시 메시지(역할/콘텐츠) 저장소
    state_messages = gr.State([])

    user_box = gr.Textbox(
        placeholder="거울아 거울아, 세상에서 누가 제일 아름답니?",
        label="질문 입력",
        lines=2,
    )

    # 제출 동작: user_box -> 모델 호출 -> chatbot/상태 갱신
    def on_submit(user_input, chat_hist, state_msgs, temp):
        # 빈 입력은 무시
        if not user_input or not user_input.strip():
            return gr.update(), state_msgs
        new_hist, new_state = chat_with_mirror(user_input.strip(), chat_hist, state_msgs, temp)
        return new_hist, new_state

    user_box.submit(
        fn=on_submit,
        inputs=[user_box, chatbot, state_messages, temperature],
        outputs=[chatbot, state_messages]
    ).then(
        lambda: gr.update(value=""),
        None,
        user_box
    )

    # 초기화 버튼: 히스토리/상태 모두 초기화
    clear_btn.click(
        fn=reset_history,
        inputs=None,
        outputs=[chatbot, state_messages]
    )

    gr.Markdown(
        """
        **TIP**: 더 길게 이어지는 대화를 원하면 질문을 짧게 여러 번 던져보세요. 
        모델이 이전 맥락을 바탕으로 점진적으로 응답 품질을 높입니다.
        """
    )

# 앱 실행
if __name__ == "__main__":
    demo.launch()
