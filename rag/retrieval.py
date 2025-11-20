from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def load_retriever(vectorstore_path, embeddings, k=5):
    vector_store = FAISS.load_local(
        folder_path=vectorstore_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    return retriever, vector_store

def rerank(query: str, docs: list[Document], hf_reranker, k=5) -> list[Document]:
    sorted_docs = sorted([(docs[i], hf_reranker.score((docs[i].page_content, query))) for i in range(len(docs))], 
                        key=lambda x:x[1], 
                        reverse=True)

    return [doc for doc, _ in sorted_docs[:k]]
