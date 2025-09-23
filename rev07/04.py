# ------------------------------------------------------------
# OpenAI API 코드 → Ollama + DeepSeek-R1 변환 예제
# ------------------------------------------------------------
# ✅ 주요 변경점:
#   1. OpenAI API Key 사용 제거 → 로컬에 설치된 Ollama 모델 활용
#   2. 모델명: "deepseek-r1" (로컬 환경에 설치되어 있다고 가정)
#   3. 인터페이스: Gradio로 간단한 대화 UI 구현
# ------------------------------------------------------------

import gradio as gr
import ollama  # Ollama 라이브러리 (pip install ollama)

# ------------------------------------------------------------
# 모델과 대화하는 함수 정의
# ------------------------------------------------------------
def chat_with_model(user_input, history):
    """
    사용자가 입력한 메시지를 DeepSeek-R1 모델에 전달하고,
    응답을 받아서 대화 기록(history)에 추가하는 함수
    """

    # Ollama 모델 호출
    response = ollama.chat(
        model="deepseek-r1",  # 로컬에 설치된 DeepSeek-R1 모델 사용
        messages=[
            {"role": "system", "content": "너는 유치원 학생이야. 유치원생처럼 답변해줘."},
            *history,  # 기존 대화 기록 유지
            {"role": "user", "content": user_input},
        ]
    )

    # 모델 응답 텍스트 추출
    bot_message = response["message"]["content"]

    # Gradio가 이해할 수 있도록 history 반환 형식 맞추기
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_message})

    # Gradio의 Chatbot 형식은 (질문, 답변) 튜플 리스트
    display_history = []
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            display_history.append((history[i]["content"], history[i+1]["content"]))

    return display_history, history


# ------------------------------------------------------------
# Gradio 인터페이스 구축
# ------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🐤 DeepSeek-R1 유치원생 모드 대화하기")

    chatbot = gr.Chatbot()  # 대화창
    state = gr.State([])    # 대화 기록 저장 (리스트 형태)

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="메시지를 입력하세요...")

    # 입력창에서 엔터 → 모델 응답 실행
    txt.submit(chat_with_model, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)  # 입력창 초기화

# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
