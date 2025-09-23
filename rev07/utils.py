# utils.py

import os

def _runs_dir(base_path: str) -> str:
    """_runs 디렉토리를 생성하고 경로를 반환합니다."""
    path = os.path.join(base_path, "_runs")
    os.makedirs(path, exist_ok=True)
    return path

def save_state(path: str, state_obj: dict):
    """실행 상태(state)를 state.txt 파일에 저장합니다."""
    run_dir = _runs_dir(path)
    with open(os.path.join(run_dir, "state.txt"), "w", encoding="utf-8") as f:
        f.write(str(state_obj))

def get_outline(path: str) -> str:
    """outline.md 파일의 내용을 읽어 반환합니다."""
    run_dir = _runs_dir(path)
    fpath = os.path.join(run_dir, "outline.md")
    return open(fpath, "r", encoding="utf-8").read() if os.path.exists(fpath) else ""

def save_outline(path: str, content: str):
    """목차(outline) 내용을 outline.md 파일에 저장합니다."""
    run_dir = _runs_dir(path)
    with open(os.path.join(run_dir, "outline.md"), "w", encoding="utf-8") as f:
        f.write(content)