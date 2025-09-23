import fitz  # PyMuPDF (pip install pymupdf)
import os

# PDF 경로
pdf_file_path = "chap04/data/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf"
doc = fitz.open(pdf_file_path)

# 헤더, 푸터 높이(px)
header_height = 80
footer_height = 80

full_text = ""

for page_num, page in enumerate(doc, start=1):
    rect = page.rect  # 페이지 크기 (x0, y0, x1, y1)
    
    # clip=(x0, y0, x1, y1) 좌표 기준으로 잘라서 추출
    header = page.get_text("text", clip=(0, 0, rect.width, header_height))
    footer = page.get_text("text", clip=(0, rect.height - footer_height, rect.width, rect.height))
    text = page.get_text("text", clip=(0, header_height, rect.width, rect.height - footer_height))

    # 페이지별 구분 + 본문 텍스트만 저장
    full_text += f"\n[페이지 {page_num}]\n{text}\n" + ("-" * 40) + "\n"

# 파일명만 추출
pdf_file_name = os.path.splitext(os.path.basename(pdf_file_path))[0]

# 출력 디렉토리 보장
output_dir = "chap04/output"
os.makedirs(output_dir, exist_ok=True)

txt_file_path = os.path.join(output_dir, f"{pdf_file_name}_with_preprocessing.txt")

# 저장
with open(txt_file_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ 저장 완료: {txt_file_path}")