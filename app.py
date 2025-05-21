import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# .env에서 API 키 로드
load_dotenv()

# ✅ PDF 로딩 함수
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# ✅ FAISS 벡터스토어 생성
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# ✅ 체인 초기화 함수 (업로드 파일 받도록 수정)
def initialize_components(selected_model, uploaded_file):
    if uploaded_file is None:
        st.warning("📎 PDF 파일을 업로드해 주세요.")
        st.stop()

    # 파일을 임시 경로에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.info(f"📄 불러온 파일: {uploaded_file.name}")  # 디버깅용 파일 이름 표시
    pages = load_and_split_pdf(file_path)
    vectorstore = create_vector_store(pages)
    retriever = vectorstore.as_retriever()

    # 프롬프트 설정
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                   "which might reference context in the chat history, "
                   "formulate a standalone question."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI answering based on the following documents.\n"
                   "If you don’t know, just say so.\n"
                   "답변은 한국어로, 존댓말을 사용하세요. 이모지도 넣어줘요.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# ✅ Streamlit 실행 메인 함수
def main():
    st.set_page_config(page_title="헌법 Q&A", page_icon="📘")
    st.header("헌법 Q&A 챗봇 💬📚")

    # 파일 업로드 UI
    uploaded_file = st.file_uploader("📎 PDF 파일 업로드", type="pdf")

    model_option = st.selectbox("🤖 GPT 모델 선택", ("gpt-4o", "gpt-3.5-turbo"))
    if uploaded_file:
        rag_chain = initialize_components(model_option, uploaded_file)
        chat_history = StreamlitChatMessageHistory(key="chat_messages")

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input("질문을 입력하세요"):
            st.chat_message("human").write(prompt)
            with st.chat_message("ai"):
                with st.spinner("GPT가 답변 중입니다..."):
                    config = {"configurable": {"session_id": "user_session"}}
                    response = conversational_rag_chain.invoke({"input": prompt}, config)
                    st.write(response["answer"])
                    with st.expander("📄 참고 문서"):
                        for doc in response["context"]:
                            st.markdown(doc.metadata.get("source", "출처 없음"), help=doc.page_content)

# ✅ 실행 시작점
if __name__ == "__main__":
    main()
