import fitz  # PyMuPDF
import os

# ① PDF 파일 경로 지정
pdf_file_path = "chap04/data/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf"

# PDF 열기
doc = fitz.open(pdf_file_path)

# 출력 경로 설정
pdf_file_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
output_dir = f"chap04/output/{pdf_file_name}"
os.makedirs(output_dir, exist_ok=True)

# 텍스트 파일 경로
txt_file_path = os.path.join(output_dir, f"{pdf_file_name}.txt")

full_text = ""

# ② 페이지 반복
for page_num, page in enumerate(doc, start=1):
    # --- 텍스트 추출 ---
    text = page.get_text("text")  # 텍스트만
    full_text += f"\n--- [페이지 {page_num}] ---\n{text}\n"

    # --- 이미지 추출 ---
    image_list = page.get_images(full=True)
    if image_list:
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]  # 이미지 XREF 번호
            pix = fitz.Pixmap(doc, xref)

            # PNG 저장 경로
            img_filename = f"page{page_num}_img{img_index}.png"
            img_path = os.path.join(output_dir, img_filename)

            # 색상 처리 (CMYK → RGB 변환 필요할 수 있음)
            if pix.n < 5:  # RGB or Gray
                pix.save(img_path)
            else:  # CMYK → RGB 변환
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(img_path)

            pix = None  # 메모리 해제

            # 텍스트 파일 안에 이미지 경로 삽입
            full_text += f"[이미지 포함 → {img_filename}]\n"

# ③ TXT 저장
with open(txt_file_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ 변환 완료: {txt_file_path}")
print(f"✅ 이미지 저장 폴더: {output_dir}")
