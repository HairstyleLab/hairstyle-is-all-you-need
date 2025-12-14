import os
from datetime import datetime
import faiss
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import json
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class QACache:
    
    def __init__(self, json_path, embeddings, vectorstore_path, similarity_threshold=0.85, batch_size=100):
        self.json_path = json_path
        self.embeddings = embeddings
        self.vectorstore_path = vectorstore_path 
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size  
        self.store_qa = []  
        
        self.vectorstore = self.create_qa_vectorstore(self.json_path, self.embeddings, self.vectorstore_path)
    
    # 질문에 대한 유사도 검색
    def get_answer(self, question):
    # 1. 먼저 아직 flush 안 된 store_qa에서 확인
        for qa_doc in self.store_qa:
            if qa_doc.page_content == question:
                return qa_doc

        # 2. vectorstore에서 검색
        results = self.vectorstore.similarity_search_with_score(question, k=1)

        if not results:
            return None

        doc, distance = results[0]
        similarity = 1 / (1 + distance)

        if similarity >= self.similarity_threshold:
            return doc
        else:
            return None

    def get_answer_with_filter(self, query_text, rule_filters=None):
        """
        룰베이스 필터링 + 유사도 검색 조합

        Args:
            query_text: hairstyle_keywords와 haircolor_keywords를 조합한 텍스트
            rule_filters: 정확히 일치해야 하는 필터 딕셔너리
                         {'gender': '남성', 'face_shape': 'Oval', ...}

        Returns:
            캐시된 답변 문자열 또는 None
        """
        # 1. 먼저 store_qa(아직 flush 안 된 데이터)에서 확인
        for qa_doc in self.store_qa:
            # 룰베이스 필터링 체크
            if rule_filters:
                match = all(
                    qa_doc.metadata.get(key) == value
                    for key, value in rule_filters.items()
                    if value is not None
                )
                if not match:
                    continue

            # 유사도 체크 (page_content가 query_text와 유사한지)
            # 여기서는 정확히 일치하는지만 확인 (나중에 벡터스토어에서 유사도 검색)
            if qa_doc.page_content == query_text:
                return qa_doc

        # 2. vectorstore에서 검색 (메타데이터 필터 적용)
        if rule_filters:
            # 메타데이터 필터를 vectorstore 검색에 적용
            # None이 아닌 값만 필터에 포함
            metadata_filter = {
                key: value
                for key, value in rule_filters.items()
                if value is not None
            }

            if metadata_filter:
                results = self.vectorstore.similarity_search_with_score(
                    query_text,
                    k=1,
                    filter=metadata_filter
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query_text, k=1)
        else:
            results = self.vectorstore.similarity_search_with_score(query_text, k=1)

        if not results:
            return None

        doc, distance = results[0]
        similarity = 1 / (1 + distance)

        if similarity >= self.similarity_threshold:
            return doc
        else:
            return None

    #벡터스토어 생성
    def create_qa_vectorstore(self, json_path, embeddings, vectorstore_path):

        # 기존 벡터 스토어가 있으면 로드 (index.faiss 파일이 있는지 확인)
        index_file = os.path.join(vectorstore_path, "index.faiss")
        index_file2 = os.path.join(vectorstore_path, "index.pkl")
        if os.path.exists(index_file) and os.path.exists(index_file2):
            vector_store = FAISS.load_local(
                folder_path=vectorstore_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        
        #아니면 새로 생성
        documents = []
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        documents = [
            Document(
                page_content=row['Qwestion'],  
                metadata={
                    'answer': row['Answer'] 
                }
            )
            for row in qa_data
        ]
        
        # 3. 벡터 스토어 생성
        embedding_dim = len(embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        if documents:
            vector_store.add_documents(documents)
        vector_store.save_local(vectorstore_path)
        
        return vector_store
    
    def add_qa(self, question, answer, metadata_dict=None):
        """
        QA 쌍 추가

        Args:
            question: 질문 텍스트 (hairstyle_keywords + haircolor_keywords)
            answer: 답변 텍스트
            metadata_dict: 룰베이스 필터링을 위한 메타데이터
                          {'gender': '남성', 'face_shape': 'Oval', ...}
        """
        metadata = {
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }

        # 추가 메타데이터가 있으면 병합
        if metadata_dict:
            metadata.update(metadata_dict)

        qa = Document(
            page_content=question,
            metadata=metadata
        )

        self.store_qa.append(qa)
        if len(self.store_qa) >= self.batch_size:
            self.flush()
    
    def flush(self):
        self.vectorstore.add_documents(self.store_qa)
        self.vectorstore.save_local(self.vectorstore_path)
        self.store_qa = []
    
    def verify_saved(self, question):
        """저장된 질문이 벡터스토어 또는 store_qa에 있는지 확인"""
        # 먼저 store_qa(아직 flush되지 않은 데이터)에서 확인
        for qa_doc in self.store_qa:
            if qa_doc.page_content == question:
                return True

        # 벡터스토어에서 확인
        results = self.vectorstore.similarity_search_with_score(question, k=1)
        if results:
            doc, distance = results[0]
            similarity = 1 / (1 + distance)
            return similarity > 0.9  # 거의 동일하면 저장된 것
        return False
    
    def get_cache_size(self):
        return self.vectorstore.index.ntotal
    
    def __del__(self):
        if self.store_qa:
            self.flush()
            

    
    
     
     

    
        
