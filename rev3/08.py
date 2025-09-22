# -*- coding: utf-8 -*-
"""
pip install camelot-py[cv] tabula-py pdfplumber pandas openpyxl

PDF의 표를 엑셀/CSV로 구조적으로 추출하는 통합 스크립트

✔️ 전략
1) Camelot (lattice → stream 순)으로 페이지별 표 감지/추출
2) 보조로 Tabula도 시도 (lattice/stream)
3) 필요시 OCR(ocrmypdf)로 가짜 텍스트 부여 후 다시 1)~2) 시도 (옵션)

⚙️ 사전 준비
- pip install camelot-py[cv] tabula-py pdfplumber pandas openpyxl
- 시스템 의존성: Java(탭룰라), Ghostscript(일부 환경에서 필요), poppler(OpenCV/렌더링 보조)
- OCR 사용 시: ocrmypdf 설치 필요 (예: brew install ocrmypdf 또는 apt-get install ocrmypdf)

💾 출력
- output_dir/
  ├─ tables/ (페이지별-테이블별 CSV 저장)
  ├─ all_tables.xlsx (모든 표를 한 파일로, 시트: p{page}_t{idx})
  └─ extract_report.json (무엇을 몇 개 추출했는지 요약)

참고: 스캔 PDF(이미지 기반)는 OCR 옵션을 켜야 인식률이 올라갑니다.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import camelot
import tabula
import pdfplumber


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_scanned_pdf(pdf_path: Path, sample_pages: int = 3) -> bool:
    """텍스트가 거의 없는(=스캔 가능성이 높은) PDF인지 대략 판단.
    앞쪽 일부 페이지만 샘플링해 텍스트 유무 검사.
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            n = min(sample_pages, len(pdf.pages))
            for i in range(n):
                txt = pdf.pages[i].extract_text() or ""
                if txt.strip():
                    return False
        return True
    except Exception:
        # 열기 실패 시 스캔 가능성 높다고 가정 (보수적으로 True)
        return True


def run_ocr_if_needed(pdf_path: Path, ocr: bool) -> Path:
    """ocr=True면 ocrmypdf로 강제 OCR PDF 생성 후 경로 반환. 아니면 원본 경로 반환."""
    if not ocr:
        return pdf_path

    ocr_out = pdf_path.with_suffix("")
    ocr_out = ocr_out.parent / f"{pdf_path.stem}_ocr.pdf"

    try:
        subprocess.run([
            "ocrmypdf", "--skip-text", "--force-ocr", "--quiet",
            str(pdf_path), str(ocr_out)
        ], check=True)
        return ocr_out
    except FileNotFoundError:
        print("[경고] ocrmypdf 미설치로 OCR 단계 건너뜁니다. (brew/apt로 설치 가능)")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"[경고] ocrmypdf 실행 실패: {e}\n원본 PDF로 계속 진행합니다.")
        return pdf_path


def camelot_try(pdf_path: Path, pages: str, flavor: str, strip_text: str = "\n") -> List[pd.DataFrame]:
    """Camelot으로 표 추출 시도 후 DataFrame 리스트 반환."""
    try:
        tables = camelot.read_pdf(str(pdf_path), pages=pages, flavor=flavor, strip_text=strip_text)
        return [t.df for t in tables]
    except Exception as e:
        print(f"[정보] Camelot {flavor} 시도 실패: {e}")
        return []


def tabula_try(pdf_path: Path, pages: str, lattice: bool) -> List[pd.DataFrame]:
    """Tabula로 표 추출 시도 후 DataFrame 리스트 반환."""
    try:
        dfs = tabula.read_pdf(str(pdf_path), pages=pages, lattice=lattice, multiple_tables=True, stream=not lattice)
        return dfs or []
    except Exception as e:
        print(f"[정보] Tabula {('lattice' if lattice else 'stream')} 시도 실패: {e}")
        return []


def extract_tables(pdf_path: str, output_dir: str, ocr: bool = False) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    ensure_dir(output_dir)
    tables_dir = output_dir / "tables"
    ensure_dir(tables_dir)

    # 0) 스캔 여부 판단 후, OCR 옵션이 켜져 있으면 선 OCR
    scanned_guess = is_scanned_pdf(pdf_path)
    working_pdf = run_ocr_if_needed(pdf_path, ocr and scanned_guess)

    # 페이지 수 계산(보고용)
    try:
        with pdfplumber.open(str(working_pdf)) as pdf:
            total_pages = len(pdf.pages)
    except Exception:
        total_pages = None

    all_tables: List[Dict[str, Any]] = []

    # 1) Camelot lattice → stream 순으로 전체 페이지 시도
    pages_param = "all"
    for method in [
        ("camelot", "lattice"),
        ("camelot", "stream"),
        ("tabula", "lattice"),
        ("tabula", "stream"),
    ]:
        engine, mode = method
        dfs: List[pd.DataFrame] = []

        if engine == "camelot":
            dfs = camelot_try(working_pdf, pages_param, mode)
        else:
            dfs = tabula_try(working_pdf, pages_param, lattice=(mode == "lattice"))

        if dfs:
            # 추출 성공 시 바로 정리 저장 후 종료
            idx = 0
            excel_path = output_dir / "all_tables.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for df in dfs:
                    idx += 1
                    # 공백 행/열 정리(간단 전처리)
                    df = df.replace({"\u00a0": " "}, regex=True)
                    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

                    csv_name = f"tables_pall_t{idx}_{engine}_{mode}.csv"
                    csv_path = tables_dir / csv_name
                    df.to_csv(csv_path, index=False)

                    sheet_name = f"pALL_t{idx}_{engine[:1]}{mode[:1]}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                    all_tables.append({
                        "page": "all",
                        "index": idx,
                        "engine": engine,
                        "mode": mode,
                        "csv": str(csv_path),
                        "sheet": sheet_name,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    })
            break  # 어떤 엔진/모드로든 한 번 성공하면 종료 (필요시 제거 가능)

    # 2) 엔진 일괄 시도에서 실패했다면, 페이지별 세밀 시도
    if not all_tables:
        print("[정보] 전체 페이지 일괄 추출 실패 → 페이지별 세밀 추출 시도")
        # 세밀 시도: 각 페이지마다 lattice → stream (camelot, tabula)
        try:
            with pdfplumber.open(str(working_pdf)) as pdf:
                total_pages = len(pdf.pages)
        except Exception:
            total_pages = 1

        excel_path = output_dir / "all_tables.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            table_counter = 0
            for p in range(1, (total_pages or 1) + 1):
                page_str = str(p)
                for engine, mode in [("camelot", "lattice"), ("camelot", "stream"), ("tabula", "lattice"), ("tabula", "stream")]:
                    if engine == "camelot":
                        dfs = camelot_try(working_pdf, page_str, mode)
                    else:
                        dfs = tabula_try(working_pdf, page_str, lattice=(mode == "lattice"))

                    if not dfs:
                        continue

                    for df in dfs:
                        table_counter += 1
                        df = df.replace({"\u00a0": " "}, regex=True)
                        df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

                        csv_name = f"tables_p{p}_t{table_counter}_{engine}_{mode}.csv"
                        csv_path = tables_dir / csv_name
                        df.to_csv(csv_path, index=False)

                        sheet_name = f"p{p}_t{table_counter}_{engine[:1]}{mode[:1]}"
                        # 시트명 길이 제한(엑셀 31자) 처리
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                        all_tables.append({
                            "page": p,
                            "index": table_counter,
                            "engine": engine,
                            "mode": mode,
                            "csv": str(csv_path),
                            "sheet": sheet_name,
                            "rows": int(df.shape[0]),
                            "cols": int(df.shape[1]),
                        })

    # 3) 리포트 저장
    report = {
        "pdf": str(pdf_path),
        "working_pdf": str(working_pdf),
        "total_pages": total_pages,
        "scanned_guess": bool(scanned_guess),
        "tables_found": len(all_tables),
        "details": all_tables,
    }
    with open(output_dir / "extract_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ 표 추출 완료 — 총 {len(all_tables)}개. 결과: {output_dir}")
    if not all_tables:
        print("⚠️ 표를 찾지 못했습니다. 스캔 PDF일 경우 ocr=True로 재시도하거나, 테이블 선/칸이 분명한 고화질 PDF를 사용해보세요.")
    return report


if __name__ == "__main__":
    # 예시 경로 — 사용자 파일명에 맞게 수정하세요
    pdf_file_path = "chap04/data/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf"
    out_dir = "chap04/output_tables"

    # 스캔 PDF라면 ocr=True 권장
    report = extract_tables(pdf_file_path, out_dir, ocr=True)
    # print(json.dumps(report, ensure_ascii=False, indent=2))
