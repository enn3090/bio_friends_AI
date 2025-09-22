# ------------------------------------------------------------
# OpenAI API 코드 → Ollama + DeepSeek-R1 변환 예제
# ------------------------------------------------------------
# ✅ 주요 변경점:
#   1. OpenAI API Key 제거 → 로컬 Ollama 모델 사용
#   2. 모델명: "deepseek-r1" (로컬 PC에 설치되어 있다고 가정)
#   3. 터미널에서 입력/출력 방식 유지 (원래 코드와 동일)
# ------------------------------------------------------------

import ollama  # Ollama 라이브러리 (pip install ollama)

# ①
def get_ai_response(messages):
    """
    Ollama를 통해 DeepSeek-R1 모델과 대화하고,
    가장 최신 응답 내용을 반환하는 함수
    """
    response = ollama.chat(
        model="deepseek-r1",
        messages=messages  # 전체 대화 기록 전달
    )
    return response["message"]["content"]  # 모델 응답 텍스트 반환


# 초기 대화 설정
messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},  # 시스템 프롬프트
]

# 대화 루프
while True:
    user_input = input("사용자: ")

    if user_input.lower() == "exit":  # ② 종료 조건
        print("대화를 종료합니다. 👋")
        break

    # 사용자 메시지를 기록에 추가
    messages.append({"role": "user", "content": user_input})

    # 모델 응답 가져오기
    ai_response = get_ai_response(messages)

    # AI 응답도 기록에 추가
    messages.append({"role": "assistant", "content": ai_response})

    # 결과 출력
    print("AI:", ai_response)