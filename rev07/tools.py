# tools.py (29, 30, 31, 32번 파일을 모두 만족하는 통합본)

import json
import os
from typing import List, Dict, Any, Tuple

# LangChain의 Document 클래스와 비슷한 구조를 가진 임시 클래스
# 모든 파일이 이 구조를 기대하고 있으므로 공통으로 사용합니다.
class MockDocument:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"MockDocument(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

# 
# 1. retrieve 함수 (모든 파일에서 필요)
#
def retrieve(params: Dict[str, Any]) -> List[MockDocument]:
    """
    [임시] 벡터 데이터베이스 검색을 흉내 냅니다.
    - 입력: {"query": "검색어"}
    - 출력: MockDocument 객체의 리스트
    """
    query = params.get("query", "")
    print(f"--- [임시 Vector DB 검색] '{query}'에 대한 검색을 수행합니다. ---")
    return [
        MockDocument(
            page_content=f"'{query}'에 대한 벡터 검색 결과 문서 1번입니다.",
            metadata={"source": "local-vector-db/doc1"}
        ),
        MockDocument(
            page_content=f"'{query}'와 관련된 벡터 검색 문서 2번입니다.",
            metadata={"source": "local-vector-db/doc2"}
        )
    ]

#
# 2. web_search 함수 (30, 31, 32번 파일에서 필요)
#
def web_search(params: Dict[str, Any]) -> Tuple[List[Dict], str]:
    """
    [임시] 웹 검색을 흉내 내고, 결과를 JSON 파일로 저장합니다.
    - 입력: {"query": "검색어"}
    - 출력: (검색 결과 리스트, 결과가 저장된 JSON 파일 경로)
    """
    query = params.get("query", "")
    print(f"--- [임시 웹 검색] '{query}'에 대한 웹 검색을 수행합니다. ---")
    
    # 가짜 검색 결과 데이터 생성
    results = [
        {"title": f"'{query}' 관련 뉴스 기사", "url": "https://example.com/news/1", "snippet": "최신 뉴스입니다..."},
        {"title": f"'{query}'에 대한 블로그 글", "url": "https://example.com/blog/2", "snippet": "자세한 설명이 담긴 글입니다..."},
    ]
    
    # 결과를 JSON 파일로 저장
    os.makedirs("outputs", exist_ok=True)
    json_path = os.path.join("outputs", f"web_search_{query.replace(' ', '_')}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"--- [임시 웹 검색] 결과를 '{json_path}' 파일에 저장했습니다. ---")
    
    return results, json_path

#
# 3. add_web_pages_json_to_chroma 함수 (30, 31, 32번 파일에서 필요)
#
def add_web_pages_json_to_chroma(json_path: str):
    """
    [임시] 웹 검색 결과가 담긴 JSON 파일을 읽어 DB에 추가하는 과정을 흉내 냅니다.
    - 입력: web_search가 생성한 JSON 파일 경로
    """
    print(f"--- [임시 DB 색인] '{json_path}' 파일의 내용을 Vector DB에 추가합니다. ---")
    if not os.path.exists(json_path):
        print(f"--- [경고] '{json_path}' 파일을 찾을 수 없습니다. ---")
        return
    
    # 실제로는 이 파일 내용을 읽어서 Vector DB에 넣는 로직이 들어갑니다.
    # 지금은 성공했다는 메시지만 출력합니다.
    print(f"--- [임시 DB 색인] 성공적으로 작업을 마쳤습니다. ---")