import json
from dotenv import load_dotenv
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
# embeddings = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko", model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})

def create_vectorstore(json_path, embeddings, keyword, vectorstore_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        tmp = json.load(f)

    embedding_dim = len(embeddings.embed_query("HI AYN"))

    index = faiss.IndexFlatL2(embedding_dim)

    documents = [
        Document(
            page_content=row['content'], 
            metadata={
                'title': row['title'],
                'link': row['link'],
                'keyword': row['keyword'],
            }
        )
        for row in tmp
    ]

    # text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size=400,
    # chunk_overlap=100,
    # length_function=len,
    # is_separator_regex=False,
    # )

    # documents = text_splitter.split_documents(documents)
    # print(len(documents))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # tmp_docs = [doc for doc in documents if doc.metadata.get('keyword')=='쿨톤' or doc.metadata.get('keyword')=='웜톤']
    # tmp_docs = [doc for doc in documents if doc.metadata.get('keyword')==keyword]

    vector_store.add_documents(documents)
    vector_store.save_local(vectorstore_path)

if __name__=="__main__":
    create_vectorstore('summary_cat_hair_face.json', embeddings, '얼굴형별 헤어스타일', 'summary_cat_face')


