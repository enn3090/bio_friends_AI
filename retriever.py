# ====== 임베딩 모델 선언 (OpenAIEmbeddings → HuggingFace SentenceTransformers) ======
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 한국어/영어 다국어 지원 소형 모델
embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ====== 언어 모델 불러오기 (ChatOpenAI → ChatOllama) ======
from langchain_community.chat_models import ChatOllama

# Ollama 로컬 DeepSeek-R1 모델
llm = ChatOllama(
    model="deepseek-r1:latest",
    temperature=0.7,
)

# ====== Load Chroma store ======
from langchain_chroma import Chroma

print("Loading existing Chroma store")
# 중요: 이 경로 'C:/...' 부분은 본인 컴퓨터의 실제 경로로 바꿔야 할 수 있습니다.
# 지금은 일단 그대로 두고, 실행 시 오류가 나면 그때 수정하겠습니다.
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
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자의 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라. "
            "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. :\n\n{query}",
        ),
    ]
)

query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()