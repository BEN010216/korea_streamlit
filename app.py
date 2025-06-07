import os
import tempfile
import streamlit as st

# dotenv optional import
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# import PDF loader 유연하게 처리
try:
    from langchain_community.document_loaders import PyPDFLoader
except ModuleNotFoundError:
    try:
        from langchain.document_loaders.pdf import PyPDFLoader
    except ModuleNotFoundError:
        PyPDFLoader = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from googleapiclient.discovery import build

# ───────── 환경 변수 로드 ─────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not OPENAI_API_KEY:
    st.error("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    st.error("환경변수 GOOGLE_API_KEY 또는 GOOGLE_CSE_ID가 설정되지 않았습니다.")

# ───────── PDF → Document 리스트 ─────────
def load_and_split_pdf(file_path: str):
    if PyPDFLoader is None:
        st.error("PDF 기능 사용 시 'pypdf'와 'langchain-community' 설치 필요")
        st.stop()
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return splitter.split_documents(pages)

# ───────── 웹검색 → Document 리스트 ─────────
def web_search_docs(query: str, num_results: int = 5):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
    items = res.get("items", [])
    docs = []
    for item in items:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        content = f"{title}\n{snippet}"
        docs.append(Document(page_content=content, metadata={"source": link}))
    return docs

# ───────── QA 체인 생성 ─────────
def setup_qa_chain(documents, model_name: str = "gpt-4"):
    vector_store = FAISS.from_documents(documents, OpenAIEmbeddings())
    llm = ChatOpenAI(model=model_name, openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

# ───────── rerun 호환성 처리 ─────────
def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ───────── Streamlit 앱 ─────────
def run_app():
    st.set_page_config(page_title="회원가입 고객상담 챗봇", page_icon="🤖", layout="wide")

    st.markdown("""
        <h1 style="text-align:center; color:#6C63FF;">📚 회원가입 고객상담 챗봇</h1>
        <p style="text-align:center;">업로드한 <strong>PDF FAQ</strong>와 <strong>웹검색</strong>을 통해 답변을 제공합니다.</p>
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ 설정")
    st.sidebar.markdown("FAQ PDF 업로드 및 웹검색 질문 기능 제공")

    if "history" not in st.session_state:
        st.session_state.history = []

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    st.sidebar.subheader("📄 FAQ PDF 업로드")
    uploaded = st.sidebar.file_uploader("PDF 업로드", type=["pdf"])
    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.sidebar.success("📁 파일 업로드 완료!")

    st.markdown("### ❓ 질문 입력")
    query = st.text_input("질문을 입력하세요", value="", key="query_input")

    if st.button("🤔 질문하기"):
        actual_query = query.strip()
        pdf_answer = None
        pdf_sources = []

        if actual_query:
            if st.session_state.uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(st.session_state.uploaded_file.read())
                    tmp_path = tmp.name

                pdf_docs = load_and_split_pdf(tmp_path)
                pdf_chain = setup_qa_chain(pdf_docs)
                pdf_res = pdf_chain.invoke({"query": actual_query})
                pdf_answer = pdf_res.get("result", "").strip()
                pdf_sources = pdf_res.get("source_documents", [])

            st.session_state.history.append({
                "query": actual_query,
                "pdf_answer": pdf_answer,
                "pdf_sources": pdf_sources,
                "web_answer": None,
                "web_sources": [],
                "web_searched": False
            })

            rerun()

    if st.session_state.history:
        st.markdown("## 📜 이전 질문")
        for i, qa in enumerate(st.session_state.history):
            with st.container():
                st.markdown(f"### ❓ 질문 {i+1}: `{qa['query']}`")

                with st.expander("📄 PDF 기반 답변 보기"):
                    if qa["pdf_answer"]:
                        st.success(qa["pdf_answer"])
                    else:
                        st.info("PDF 기반 답변 없음.")

                if qa["web_searched"]:
                    with st.expander("🌐 웹검색 기반 답변 보기"):
                        if qa["web_answer"]:
                            st.warning(qa["web_answer"])
                        else:
                            st.info("웹검색 답변 없음.")
                        st.markdown("📑 **참고 자료:**")
                        for doc in qa["web_sources"]:
                            url = doc.metadata.get("source", "")
                            st.markdown(f"- 🌍 [출처]({url})")

                if not qa["web_searched"]:
                    if st.button(f"🌐 웹검색 답변 - 질문 {i+1}", key=f"web_search_{i}"):
                        web_docs = web_search_docs(qa["query"])
                        web_chain = setup_qa_chain(web_docs)
                        web_res = web_chain.invoke({"query": qa["query"]})
                        qa["web_answer"] = web_res.get("result", "").strip()
                        qa["web_sources"] = web_res.get("source_documents", [])
                        qa["web_searched"] = True
                        rerun()

    if st.sidebar.button("🧹 업로드 초기화"):
        st.session_state.uploaded_file = None
        rerun()

    st.markdown("---")
    st.markdown("## 🔍 웹검색 전용")
    web_query = st.text_input("웹에서 검색할 질문을 입력하세요", key="web_query_input")
    if st.button("🌐 웹검색만 하기", key="web_search_direct"):
        if web_query.strip():
            with st.spinner("웹에서 정보를 수집 중입니다..."):
                web_docs = web_search_docs(web_query.strip())
                web_chain = setup_qa_chain(web_docs)
                web_res = web_chain.invoke({"query": web_query.strip()})

                st.markdown("### 🌐 웹검색 답변:")
                st.code(web_res.get("result", "").strip(), language="markdown")

                st.markdown("### 📑 참고 자료:")
                for doc in web_res.get("source_documents", []):
                    url = doc.metadata.get("source", "")
                    st.markdown(f"- 🌍 [참고 자료]({url})")

if __name__ == "__main__":
    run_app()
