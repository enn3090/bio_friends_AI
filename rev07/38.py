# 오류 수정된 전체 코드

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Tuple

import gradio as gr
from langchain_community.chat_models import ChatOllama

# =========================
# LLM (DeepSeek-R1 via Ollama)
# =========================

def make_llm(model: str = "deepseek-r1", temperature: float = 0.2) -> ChatOllama:
    """로컬 Ollama용 LangChain Chat 모델. API Key 불필요."""
    return ChatOllama(model=model, temperature=temperature)

# =========================
# 프롬프트 템플릿
# =========================
SYS_ANALYZE = (
    "당신은 신중하고 정확한 전문가입니다.\n"
    "요청을 해결하기 위해 핵심 쟁점과 접근계획을 **간결한 불릿 요약**으로 정리하세요.\n"
    "불필요한 추론 과정이나 장황한 사고과정은 배제하고, 사용자가 활용 가능한 **요약 분석**만 제시하세요.\n"
    "가능하면 근거/가정, 잠재 리스크, 대안 경로를 포함하세요.\n"
)

USR_ANALYZE = (
    """
질의:
{query}

요구사항:
- 결과 형식은 Markdown. 섹션 헤더를 사용하세요.
- 섹션 예시: "문제 재정의", "핵심 요약(최대 {max_bullets} bullets)", "가정", "해결 접근", "잠재 리스크".
- 과도한 내부 사고 전개 대신 **핵심 요약** 중심으로 작성.
    """
)

SYS_ANSWER = (
    "지금부터는 위 분석 요약을 바탕으로 **최종 답변**만을 명확하고 실행가능하게 제시하세요.\n"
    "필요시 단계별 절차/코드/수식/정책안을 포함하되, 불필요한 사색은 배제하세요. Markdown 형식으로 작성."
)

USR_ANSWER = (
    """
사용자 질의:
{query}

참고용 분석 요약:
{analysis}

요구사항:
- 최종 답변(결론/솔루션/권고)을 Markdown으로 작성.
- 필요 시 예시, 표, 코드블록, 체크리스트를 포함.
- 모호성/전제조건은 먼저 명시.
    """
)

SYS_REFLECT = (
    "전문 감리자 역할입니다. 아래 최종 답변을 **검증**하고 개선점을 제시하세요.\n"
    "검증 항목: 정확성, 누락/엣지케이스, 일관성, 실행가능성, 안전/윤리.\n"
    "결과는 Markdown 섹션으로: '빠른 점검표', '핵심 지적사항', '수정 제안', '신뢰도(0-100)' 형식."
)

USR_REFLECT = (
    """
사용자 질의:
{query}

최종 답변:
{answer}

요구사항:
- 간결하고 수행가능한 지침 위주로 작성.
- 사실 오류가 의심되면 근거와 함께 표시.
    """
)

# =========================
# 헬퍼
# =========================

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name, flags=re.I)
    name = name.strip("_")
    return name or "deep_thinking"


def run_pipeline(query: str, temperature: float, max_bullets: int, language: str) -> Tuple[str, str, str, str]:
    """Sequential: Analyze → Answer → Reflect. 각 단계는 별도의 LLM 호출."""
    if not query or not query.strip():
        return "", "", "", ""

    llm = make_llm(temperature=temperature)

    # 1) Analyze
    analyze_prompt = f"[SYSTEM]\n{SYS_ANALYZE}\n\n[USER]\n" + USR_ANALYZE.format(query=query, max_bullets=max_bullets)
    analysis = llm.invoke(analyze_prompt).content

    # 2) Answer
    answer_prompt = f"[SYSTEM]\n{SYS_ANSWER}\n\n[USER]\n" + USR_ANSWER.format(query=query, analysis=analysis)
    answer = llm.invoke(answer_prompt).content

    # 3) Reflect
    reflect_prompt = f"[SYSTEM]\n{SYS_REFLECT}\n\n[USER]\n" + USR_REFLECT.format(query=query, answer=answer)
    reflection = llm.invoke(reflect_prompt).content

    # 4) Markdown 합치기
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Deep Thinking Report\n\n- Model: deepseek-r1 (Ollama)\n- Time: {now}\n- Language: {language}\n\n---\n"

    md = (
        header
        + f"\n## 1) Query\n\n{query}\n\n"
        + f"## 2) Analysis Summary\n\n{analysis}\n\n"
        + f"## 3) Final Answer\n\n{answer}\n\n"
        + f"## 4) Reflection (QA/Review)\n\n{reflection}\n"
    )

    return analysis, answer, reflection, md


def save_markdown_to_temp(md: str, filename_hint: str = "") -> str:
    """Gradio File 컴포넌트와 호환되도록 임시 파일에 저장하고 경로를 반환합니다."""
    # outputs 디렉토리가 없으면 생성합니다.
    os.makedirs("outputs", exist_ok=True)
    
    base = sanitize_filename(filename_hint) if filename_hint else "deep_thinking"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 파일명만 생성합니다. Gradio가 임시 디렉토리에 이 이름으로 파일을 생성합니다.
    final_filename = f"{base}_{ts}.md"
    
    # 임시로 파일을 저장할 실제 경로를 구성합니다.
    temp_path = os.path.join("outputs", final_filename)
    
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    # 저장된 파일의 경로를 반환합니다.
    return temp_path


# =========================
# Gradio UI
# =========================

def build_ui():
    with gr.Blocks(title="OpenCode • DeepSeek-R1 Sequential Thinking + Reflection") as demo:
        gr.Markdown(
            """
            # OpenCode • Sequential Deep Thinking → Reflection
            - **Analyze → Answer → Reflect** 3단계로 질의를 처리합니다.
            - 결과는 Markdown으로 제공되며, **.md 파일 저장**이 가능합니다.
            - 로컬 **DeepSeek-R1(Ollama)** 사용 (API Key 불필요)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query = gr.Textbox(label="질의 입력", placeholder="무엇을 도와드릴까요?", lines=8)
                with gr.Row():
                    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                    max_bullets = gr.Slider(3, 20, value=8, step=1, label="분석 요약 bullet 상한")
                language = gr.Dropdown(["ko", "en"], value="ko", label="언어(표시용)")
                run = gr.Button("생성", variant="primary")
                fname = gr.Textbox(label="파일명 힌트(선택)", placeholder="report, plan, spec 등")
                save_btn = gr.Button("Markdown 저장")

            with gr.Column(scale=1):
                analysis_md = gr.Markdown(label="Analysis Summary")
                answer_md = gr.Markdown(label="Final Answer")
                reflect_md = gr.Markdown(label="Reflection (검증)")
                file_out = gr.File(label="다운로드(.md)")

        state_md = gr.State("")

        def on_run(q, t, mb, lang):
            a, b, c, md = run_pipeline(q, t, int(mb), lang)
            # state_md에만 전체 마크다운을 저장합니다.
            return a, b, c, md

        run.click(
            on_run,
            inputs=[query, temperature, max_bullets, language],
            outputs=[analysis_md, answer_md, reflect_md, state_md],
        )

        def on_save(md_str, hint):
            if not md_str:
                return None
            # 임시 파일 저장 함수를 호출합니다.
            path = save_markdown_to_temp(md_str, hint)
            return path

        save_btn.click(on_save, inputs=[state_md, fname], outputs=[file_out])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)