from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
embeddings = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko", model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})

def load_retriever(vectorstore_path, embeddings=embeddings, k=5):
    vector_store = FAISS.load_local(
        folder_path=vectorstore_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    return retriever, vector_store

if __name__=="__main__":
    retriever,_ = load_retriever('db/season', k=3)

    retriever.invoke('여름 남자 머리 추천')