"""
캐시 매니저 싱글톤 - 전역에서 캐시 접근 가능
"""
from langchain_openai import OpenAIEmbeddings
from rag.qa_cache import QACache
import traceback

class CacheManager:
    _instance = None
    _qa_cache = None
    _last_tool_params = None  # 마지막 툴 호출 파라미터
    _last_cache_hit = False   # 마지막 캐시 히트 여부

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._qa_cache is None:
            self._initialize_cache()

    def _initialize_cache(self):
        """QA 캐시 초기화"""
        try:
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
            self._qa_cache = QACache(
                json_path="rag/qa.json",
                embeddings=embeddings,
                vectorstore_path="rag/qa_vectorstore",
                similarity_threshold=0.3,
                batch_size=1
            )
            print("[CacheManager] QA Cache 초기화 완료")
        except Exception as e:
            print(f"[CacheManager] QA Cache 초기화 실패: {e}")
            self._qa_cache = None

    def search_cache(self, **kwargs):
        """
        캐시에서 답변 검색
        - hairstyle_keywords, haircolor_keywords: 유사도 검색
        - 나머지 필드: 룰베이스 정확 매칭

        Args:
            **kwargs: gender, face_shape, hairlength_keywords, season,
                      hairstyle_keywords, haircolor_keywords, personal_color 등

        Returns:
            캐시된 답변 문자열 또는 None
        """
        if not self._qa_cache:
            return None

        try:
            # 유사도 검색에 사용할 키워드들
            similarity_keys = ['hairstyle_keywords', 'haircolor_keywords']
            # 룰베이스 필터링에 사용할 필드들
            rule_keys = ['gender', 'face_shape', 'hairlength_keywords', 'season', 'personal_color']

            # 1. 유사도 검색을 위한 query_text 생성
            query_parts = []
            for key in similarity_keys:
                if key in kwargs and kwargs[key] is not None:
                    query_parts.append(str(kwargs[key]))

            if not query_parts:
                # 유사도 검색에 사용할 키워드가 없으면 검색 불가
                return None

            query_text = " ".join(query_parts)

            # 2. 룰베이스 필터 생성
            rule_filters = {}
            for key in rule_keys:
                if key in kwargs and kwargs[key] is not None:
                    rule_filters[key] = kwargs[key]

            # 3. 필터링 + 유사도 검색 수행
            cached_doc = self._qa_cache.get_answer_with_filter(query_text, rule_filters)

            if cached_doc:
                print(f"[CACHE HIT] 캐시에서 답변 반환")
                print(f"  - Query: {query_text[:50]}...")
                print(f"  - Filters: {rule_filters}")
                self._last_cache_hit = True
                self._last_tool_params = kwargs
                return cached_doc.metadata['answer']
            else:
                print(f"[CACHE MISS] 캐시에 없음, 새로 추론")
                print(f"  - Query: {query_text[:50]}...")
                print(f"  - Filters: {rule_filters}")
                self._last_cache_hit = False
                self._last_tool_params = kwargs
                return None

        except Exception as e:
            print(f"[CacheManager] Cache 검색 실패: {e}")
            traceback.print_exc()
            return None

    def store_cache(self, answer, **kwargs):
        """
        캐시에 답변 저장
        - hairstyle_keywords, haircolor_keywords: query text로 저장
        - 나머지 필드: 메타데이터로 저장

        Args:
            answer: 저장할 답변 문자열
            **kwargs: gender, face_shape, hairlength_keywords, season,
                      hairstyle_keywords, haircolor_keywords, personal_color 등
        """
        if not self._qa_cache:
            return

        try:
            # 유사도 검색에 사용할 키워드들
            similarity_keys = ['hairstyle_keywords', 'haircolor_keywords']
            # 룰베이스 필터링에 사용할 필드들
            rule_keys = ['gender', 'face_shape', 'hairlength_keywords', 'season', 'personal_color']

            # 1. query_text 생성 (hairstyle + haircolor keywords)
            query_parts = []
            for key in similarity_keys:
                if key in kwargs and kwargs[key] is not None:
                    query_parts.append(str(kwargs[key]))

            if not query_parts:
                # 유사도 검색에 사용할 키워드가 없으면 저장 불가
                return

            query_text = " ".join(query_parts)

            # 2. 메타데이터 생성 (룰베이스 필드들)
            metadata = {}
            for key in rule_keys:
                if key in kwargs and kwargs[key] is not None:
                    metadata[key] = kwargs[key]

            # 3. 최종 답변(문자열) 저장
            if isinstance(answer, str):
                self._qa_cache.add_qa(query_text, answer, metadata)
                print("Cache Size:", self._qa_cache.get_cache_size())
                print(f"[CACHE STORE] Cache 저장 완료")
                print(f"  - Query: {query_text[:50]}...")
                print(f"  - Metadata: {metadata}")

        except Exception as e:
            print(f"[CacheManager] Cache 저장 실패: {e}")
            traceback.print_exc()

    def get_cache_size(self):
        """캐시 크기 반환"""
        if self._qa_cache:
            return self._qa_cache.get_cache_size()
        return 0

    def get_last_tool_params(self):
        """마지막 툴 호출 파라미터 반환"""
        return self._last_tool_params

    def was_last_cache_hit(self):
        """마지막 조회가 캐시 히트였는지 반환"""
        return self._last_cache_hit

    def reset_state(self):
        """캐시 상태 리셋"""
        self._last_tool_params = None
        self._last_cache_hit = False


# 싱글톤 인스턴스 생성
cache_manager = CacheManager()
