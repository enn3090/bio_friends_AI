# ------------------------------------------------------------
# OpenAI API → Ollama(DeepSeek-R1) 변환 버전
# ------------------------------------------------------------
# ✅ 기능
#   - 텍스트 파일을 읽어서 요약
#   - Ollama DeepSeek-R1 모델을 사용
#   - 출력 포맷: 제목 / 저자의 문제 인식 및 주장 / 저자 소개
# ------------------------------------------------------------

import ollama

def summarize_txt(file_path: str, temperature: float = 0.1):
    # ① 텍스트 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # ② 요약을 위한 시스템 프롬프트
    system_prompt = '''
    너는 다음 글을 요약하는 봇이다. 
    아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라. 

    작성해야 하는 포맷은 다음과 같다. 
    
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)
    
    ## 저자 소개
    '''

    # ③ Ollama에 전달할 메시지
    response = ollama.chat(
        model="deepseek-r1",  # 로컬 모델 사용
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": txt},
        ],
        options={"temperature": temperature}
    )

    summary = response["message"]["content"].strip()
    return summary


if __name__ == '__main__':
    file_path = './chap04/output/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축_with_preprocessing.txt'

    summary = summarize_txt(file_path, temperature=0.1)
    print(summary)

    # ④ 요약된 내용을 파일로 저장
    with open('./chap04/output/crop_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    print("✅ 요약이 완료되어 crop_model_summary.txt 에 저장되었습니다.")
