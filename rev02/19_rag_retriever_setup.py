"""
[OpenCode 변환본]

목표: OpenAI 기반 임베딩 + GPT 모델을 이용한 RAG 설정 코드를
→ **로컬 오픈소스 LLM (Ollama DeepSeek-R1)** + **SentenceTransformers 임베딩** + **Chroma** 기반으로 변환

핵심 변경점
1) OpenAIEmbeddings → HuggingFace 임베딩 모델 사용 (`SentenceTransformerEmbeddings`)
   - 추천: "sentence-transformers/all-MiniLM-L6-v2" (작고 빠르며 CPU에서도 동작)
   - 필요 시 KoSimCSE 등 한국어 특화 모델로 교체 가능
   - 설치: `pip install sentence-transformers`
2) ChatOpenAI → **ChatOllama** (로컬 Ollama DeepSeek-R1 사용)
3) 나머지 RAG 파이프라인 (Chroma, retriever, document_chain, query_augmentation_chain)은 동일
4) OutputParser, ChatPromptTemplate, MessagesPlaceholder 그대로 활용

사전 준비
- Ollama 설치: `ollama pull deepseek-r1:latest`
- HuggingFace 모델: `pip install sentence-transformers`
- ChromaDB: `pip install chromadb`
"""

# ====== 임베딩 모델 선언 (OpenAIEmbeddings → HuggingFace SentenceTransformers) ======
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 한국어/영어 다국어 지원 소형 모델
embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ====== 언어 모델 불러오기 (ChatOpenAI → ChatOllama) ======
from langchain_community.chat_models import ChatOllama

# Ollama 로컬 DeepSeek-R1 모델
llm = ChatOllama(
    model="deepseek-r1:latest",   # 필요 시 "deepseek-r1:7b" 또는 "deepseek-r1:14b"
    temperature=0.7,
    # base_url="http://localhost:11434",  # Ollama 기본값이면 생략 가능
)

# ====== Load Chroma store ======
from langchain_chroma import Chroma

print("Loading existing Chroma store")
persist_directory = 'C:/github/gpt_agent_2025_easyspub/chap09/chroma_store'

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# ====== Create retriever ======
retriever = vectorstore.as_retriever(k=3)

# ====== Create document chain ======
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 문서 기반 QA용 프롬프트
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

# ====== Query augmentation chain ======
query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),  # 기존 대화 내용
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자의 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라. "
            "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. :\n\n{query}",
        ),
    ]
)

query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()