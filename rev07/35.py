"""
OpenCode 변환본 • DeepSeek-R1(로컬 Ollama) + Gradio로 Mermaid 다이어그램 코드 생성 & 이미지로 보기/저장

기능
- DeepSeek-R1이 사용자의 아이디어를 Mermaid 코드로 자동 생성/보정
- Gradio UI에서 Mermaid 다이어그램을 즉시 렌더링(SVG)
- 브라우저 측 JavaScript로 SVG→PNG로 저장(이미지 다운로드 버튼 제공)
- 다이어그램 타입/테마/크기 제어

전제(설치)
    pip install -U gradio langchain langchain-community
    # Ollama & 모델 준비: https://ollama.com
    #   1) ollama pull deepseek-r1
    #   2) ollama serve (보통 자동 실행)

참고
- 서버(파이썬)에서 PNG 변환 없이, 프론트(브라우저)에서 PNG로 저장합니다.
- 완전 오프라인이 필요하면 mermaid.min.js 파일을 로컬에 두고 <script src="/local/path"> 로 교체하세요.
"""
from __future__ import annotations

import json
import re
from typing import Tuple

import gradio as gr
from langchain_community.chat_models import ChatOllama

# =========================
# LLM (DeepSeek-R1 via Ollama)
# =========================

def make_llm(model: str = "deepseek-r1", temperature: float = 0.2) -> ChatOllama:
    """로컬 Ollama용 LangChain Chat 모델. API Key 불필요."""
    return ChatOllama(model=model, temperature=temperature)

LLM = make_llm()

SYS_PROMPT = (
    "당신은 Mermaid 다이어그램 프롬프트 엔지니어입니다.\n"
    "사용자의 의도를 바탕으로 **유효한 Mermaid 코드만** 생성하세요.\n"
    "반드시 JSON만 출력: {\"code\": \"```mermaid...```\", \"title\": \"...\"}. 다른 텍스트 금지.\n"
    "지원 타입 예: flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, journey, gantt, pie.\n"
)

USER_TEMPLATE = (
    """
요청 요약: {idea}
원하는 다이어그램 타입(힌트): {diagram_type}
제약:
- Mermaid 문법 유효해야 함 (렌더 오류 금지)
- 코드블록으로 감싸기: ```mermaid ...```
- 간결한 노드/엣지/레이블, 과도한 텍스트 금지
- title(선택)을 JSON에 함께 포함
    """
)


def extract_mermaid_from_json(raw: str) -> Tuple[str, str]:
    """LLM JSON 응답에서 mermaid 코드와 제목을 추출한다."""
    try:
        data = json.loads(raw)
    except Exception:
        # 코드블록 포함 등 잡텍스트 제거
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group(0)) if m else {"code": raw}

    code = data.get("code", "").strip()
    title = data.get("title", "").strip()

    # ```mermaid ... ``` 내부만 뽑기
    m2 = re.search(r"```mermaid\s*([\s\S]*?)\s*```", code)
    if m2:
        code = m2.group(1).strip()
    return code, title


# =========================
# Mermaid 렌더용 HTML 생성(브라우저에서 SVG→PNG 저장 지원)
# =========================

def build_mermaid_html(code: str, theme: str = "default", width: int = 1024, height: int = 768) -> str:
    code_js = json.dumps(code)  # 안전 이스케이프
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <style>
    body {{ font-family: ui-sans-serif, system-ui; margin: 0; padding: 0; background: #0b1020; color: #e6edf7; }}
    .toolbar {{ display:flex; gap:8px; align-items:center; padding:10px 12px; background:#0f172a; position:sticky; top:0; z-index:1; }}
    .btn {{ background:#1e293b; color:#cbd5e1; border:1px solid #334155; padding:6px 10px; border-radius:10px; cursor:pointer; }}
    .btn:hover {{ background:#0ea5e9; color:#0b1020; border-color:#38bdf8; }}
    #wrap {{ padding:12px; }}
    .box {{ background:#0a122b; border:1px solid #1f2a44; border-radius:16px; padding:12px; overflow:auto; }}
    #svgBox {{ width:{width}px; height:{height}px; overflow:auto; background:#0a122b; }}
    .hint {{ opacity:.7; font-size:12px; margin-left:8px; }}
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs\" type=\"module\"></script>
</head>
<body>
  <div class=\"toolbar\">
    <button class=\"btn\" id=\"btnRender\">렌더</button>
    <button class=\"btn\" id=\"btnPNG\">PNG 저장</button>
    <span class=\"hint\">테마: {theme} · 크기: {width}×{height}</span>
  </div>
  <div id=\"wrap\">
    <div class=\"box\">
      <div id=\"svgBox\"></div>
    </div>
  </div>

  <script type=\"module\">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: false, theme: '{theme}' }});

    const code = {code_js};
    const svgBox = document.getElementById('svgBox');

    async function renderNow() {{
      try {{
        const { '{' } svg { '}' } = await mermaid.render('theGraph', code);
        svgBox.innerHTML = svg;
      }} catch (e) {{
        svgBox.innerHTML = `<pre style=\"color:#fca5a5\">렌더 오류: ${'{'}String(e){'}'}</pre>`;
      }}
    }}

    async function downloadPNG() {{
      const svgEl = svgBox.querySelector('svg');
      if (!svgEl) return;
      const svgData = new XMLSerializer().serializeToString(svgEl);
      const svgBlob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      const scale = 2; // 더 선명한 PNG를 위해 2배 스케일
      await new Promise((res) => {{ img.onload = res; img.src = url; }});

      const canvas = document.createElement('canvas');
      canvas.width = Math.max({width}, img.width) * scale;
      canvas.height = Math.max({height}, img.height) * scale;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0a122b';
      ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {{
        const a = document.createElement('a');
        a.download = 'mermaid.png';
        a.href = URL.createObjectURL(blob);
        a.click();
      }}, 'image/png');

      URL.revokeObjectURL(url);
    }}

    document.getElementById('btnRender').addEventListener('click', renderNow);
    document.getElementById('btnPNG').addEventListener('click', downloadPNG);

    // 초기 렌더
    renderNow();
  </script>
</body>
</html>
    """
    return html


# =========================
# Gradio 핸들러
# =========================

def generate_mermaid(idea: str, diagram_type: str, use_llm: bool, theme: str, width: int, height: int):
    idea = (idea or "").strip()
    if not idea:
        return gr.update(value=""), gr.update(value="무엇을 그릴지 설명을 입력하세요."), gr.update(visible=False)

    if use_llm:
        msg = f"{SYS_PROMPT}\n\n" + USER_TEMPLATE.format(idea=idea, diagram_type=diagram_type)
        raw = LLM.invoke(msg).content
        code, title = extract_mermaid_from_json(raw)
    else:
        # 사용자가 이미 Mermaid 문법을 직접 작성한다고 가정
        code, title = idea, ""

    if not code.strip():
        return gr.update(value=""), gr.update(value="Mermaid 코드를 생성할 수 없습니다."), gr.update(visible=False)

    html = build_mermaid_html(code, theme=theme, width=int(width), height=int(height))
    code_block = f"""```mermaid\n{code}\n```"""
    title_txt = title or "(제목 없음)"
    header = f"제목: {title_txt}\n타입 힌트: {diagram_type}\n테마: {theme} / 크기: {width}x{height}"

    return gr.update(value=code_block), gr.update(value=header), gr.update(value=html, visible=True)


# =========================
# UI 구성
# =========================

def build_ui():
    with gr.Blocks(title="OpenCode • DeepSeek-R1 Mermaid Studio") as demo:
        gr.Markdown(
            """
            # OpenCode • DeepSeek-R1 Mermaid Studio
            - 로컬 **DeepSeek-R1(Ollama)** 가 아이디어→Mermaid 코드 자동 생성
            - **Gradio**에서 즉시 렌더링 및 **PNG 저장** 지원(브라우저 변환)
            - 타입/테마/크기 제어
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                idea = gr.Textbox(label="아이디어 또는 Mermaid 코드", placeholder="예) 결제 플로우: 사용자→장바구니→결제→완료", lines=6)
                with gr.Row():
                    diagram_type = gr.Dropdown(
                        [
                            "flowchart",
                            "sequenceDiagram",
                            "classDiagram",
                            "stateDiagram-v2",
                            "erDiagram",
                            "journey",
                            "gantt",
                            "pie",
                        ],
                        value="flowchart",
                        label="다이어그램 타입(힌트)",
                    )
                    use_llm = gr.Checkbox(value=True, label="DeepSeek-R1로 코드 생성/보정")
                with gr.Row():
                    theme = gr.Dropdown(["default", "neutral", "dark", "forest", "base"], value="dark", label="테마")
                    width = gr.Slider(600, 2000, value=1024, step=10, label="가로(px)")
                    height = gr.Slider(400, 1500, value=720, step=10, label="세로(px)")
                run = gr.Button("렌더 / 코드 생성", variant="primary")
                info = gr.Textbox(label="메타 정보", interactive=False)
                code_out = gr.Code(label="Mermaid 코드", language="markdown")

            with gr.Column(scale=1):
                gr.Markdown("### 미리보기 & PNG 저장")
                html_out = gr.HTML(visible=False)

        run.click(
            generate_mermaid,
            inputs=[idea, diagram_type, use_llm, theme, width, height],
            outputs=[code_out, info, html_out],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)

