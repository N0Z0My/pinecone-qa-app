import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import PineconeVectorStore

# Streamlit secret または環境変数から API キーとインデックス名を取得
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = st.secrets.get("PINECONE_INDEX") or os.getenv("PINECONE_INDEX")

# 必要な環境変数が設定されているか確認
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX]):
    st.error("必要な環境変数が設定されていません。OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX を確認してください。")
    st.stop()

# OpenAI embedding設定
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Pinecone初期化
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# 環境変数から指定されたインデックスを使用
index = pc.Index(PINECONE_INDEX)

# Pineconeベクトルストアの初期化
vectorstore = PineconeVectorStore(index, embeddings.embed_query, "text")

# ChatGPTモデルの初期化
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# QAチェーンの作成
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

def get_answer(query):
    result = qa({"query": query})
    return result["result"], result["source_documents"]

def main():
    st.title("AI Assistant - Ask Me Anything")

    query = st.text_input("あなたの質問を入力してください:")

    if st.button("質問する"):
        if query:
            with st.spinner("回答を生成中..."):
                answer, sources = get_answer(query)
            
            st.subheader("回答:")
            st.write(answer)

            st.subheader("参照元:")
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content)
                st.write(f"Metadata: {doc.metadata}")
                st.markdown("---")
        else:
            st.warning("質問を入力してください。")

if __name__ == "__main__":
    main()