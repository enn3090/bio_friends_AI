import base64
import gradio as gr
import ollama

# 이미지 파일을 Base64로 인코딩하는 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 모델에게 이미지 설명을 요청하는 함수
def analyze_image(image, prompt="이 이미지에 대해 설명해주세요."):
    # 업로드된 이미지를 base64로 변환
    image_base64 = encode_image(image)

    # Ollama 모델 호출
    response = ollama.chat(
        model="deepseek-r1",   # 로컬에 설치된 DeepSeek-R1 모델 사용
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n(이미지는 base64로 전달됩니다)",
            },
            {
                "role": "user",
                "content": f"data:image/jpeg;base64,{image_base64}",
            }
        ],
    )

    return response["message"]["content"]

# Gradio 인터페이스 정의
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ 이미지 설명 생성기 (DeepSeek-R1 + Ollama)")
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="이미지를 업로드하세요")
        prompt_input = gr.Textbox(label="프롬프트 입력", value="이 이미지에 대해 설명해주세요.")
    
    output = gr.Textbox(label="모델 응답")

    analyze_btn = gr.Button("이미지 분석하기")
    analyze_btn.click(fn=analyze_image, inputs=[image_input, prompt_input], outputs=output)

# 실행
if __name__ == "__main__":
    demo.launch()
