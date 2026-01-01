import random
import pandas as pd
from typing import List, Dict
from llm_providers import get_llm_provider

LLM_PROVIDER = "ollama"
PROVIDER_MODELS = {
    'ollama': 'qwen3:8b',
    'openai': 'gpt-4o-mini'
}

try:
    llm_client = get_llm_provider(
        LLM_PROVIDER,
        model=PROVIDER_MODELS.get(LLM_PROVIDER)
    )
    print(f"LLM Provider 초기화 완료: {LLM_PROVIDER}")
except Exception as e:
    print(f"LLM Provider 초기화 실패: {e}")
    print("OpenAI로 폴백합니다...")
    LLM_PROVIDER = "openai"
    llm_client = get_llm_provider("openai")

def generate_queries_for_document(
    doc_id: str,
    content: str,
    metadata: Dict,
    doc_type: str,
    num_queries: int = 3
) -> List[str]:

    details = metadata.get('details', metadata.get('keyword', ''))
    gender = metadata.get('gender', '')

    prompts = {
        'hairstyle_feature': 
            f"""
            다음은 헤어스타일에 대한 설명입니다:

            {content}

            이 문서를 검색하기 위한 자연스러운 검색 쿼리를 1개 생성하세요.

            요구사항:
            - 오직 문서 내용을 바탕으로 해당 헤어스타일의 특징 키워드만을 추출
            - 각 쿼리는 다양한 키워드들을 같이 포함할 수 있음
            - 추출 가능한 특징들만 추출하되 최대 3개의 특징 까지만 추출
            - 헤어스타일명은 절대로 추출하지 말고 오직 느낌이나 성격, 묘사에 대한 키워드만 추출할 것

            예시:
            ["시원한, 이마 라인이 드러나는"]
            ["차분한"] 
            ["볼륨 있는, 단정해 보이는, 고풍스러운"]

            반드시 JSON 형식으로 반환하세요:
            {{"queries": ["쿼리"]}}
            """,
        'haircolor_feature': 
            f"""
            다음은 헤어 컬러에 대한 설명입니다:

            {content}

            이 문서를 검색하기 위한 자연스러운 검색 쿼리를 1개 생성하세요.

            요구사항:
            - 오직 문서 내용을 바탕으로 해당 헤어스타일의 특징 키워드만을 추출
            - 각 쿼리는 다양한 키워드들을 같이 포함할 수 있음
            - 추출 가능한 특징들만 추출하되 최대 3개의 특징 까지만 추출
            - 헤어컬러명은 절대로 추출하지 말고 오직 느낌이나 성격, 묘사에 대한 키워드만 추출할 것

            예시:
            ["따뜻한, 밝은"] 
            ["퇴폐적인, 븕은 계열, 어두운"]
            ["사람들이 많이 하는, 무난한"]

            반드시 JSON 형식으로 반환하세요:
            {{"queries": ["쿼리"]}}
            """
    }
    
    prompt = prompts.get(doc_type, prompts['hairstyle_feature'])

    try:
        queries = llm_client.generate_queries(prompt, num_queries)

        if queries and len(queries) > 0:
            return queries
        else:
            raise ValueError("쿼리 생성 실패")
    
    except Exception as e:
        print(f"Error generating queries for {doc_id}: {e}")
        fallback_query = details if details else metadata.get('title', 'unknown')
        return [fallback_query] * num_queries


def create_retrieval_dataset(
    grouped_docs: Dict,
    queries_per_doc: int = 3,
    max_docs_per_type: int = None,
    shuffle: bool = True
) -> pd.DataFrame:

    qa_dataset = []
    qid_counter = 0

    for doc_type, docs in grouped_docs.items():
        print(f"\n{'='*60}")
        print(f"처리 중: {doc_type} (총 {len(docs)}개 문서)")
        print(f"{'='*60}")

        if shuffle:
            docs_copy = docs.copy()
            random.shuffle(docs_copy)
            docs = docs_copy
            print(f"문서 순서 랜덤 셔플 완료")

        if max_docs_per_type:
            docs = docs[:max_docs_per_type]
            print(f"제한 적용: {max_docs_per_type}개 문서만 처리")

        for idx, doc in enumerate(docs, 1):
            doc_id = doc['doc_id']
            content = doc['contents']
            metadata = doc['metadata']

            print(f"  [{idx}/{len(docs)}] 문서 {doc_id}: 쿼리 생성 중...")

            queries = generate_queries_for_document(
                doc_id=doc_id,
                content=content,
                metadata=metadata,
                doc_type=doc_type,
                num_queries=queries_per_doc
            )

            print(f"생성된 쿼리: {queries}")

            for query in queries:
                metadata_filter = {}
                for key in ['title', 'category', 'gender', 'details']:
                    if key in metadata:
                        metadata_filter[key] = metadata[key]

                qa_dataset.append({
                    'qid': f'{doc_type[:2].upper()}_{qid_counter}',
                    'query': query,
                    'metadata_filter': metadata_filter,
                    'retrieval_gt': [doc_id],
                    'search_type': doc_type
                })
                qid_counter += 1

    df = pd.DataFrame(qa_dataset)
    print(f"\n{'='*60}")
    print(f"총 {len(df)}개 평가 항목 생성 완료")
    print(f"{'='*60}")

    return df

if "__main__" == __name__:
    qa_dataset = create_retrieval_dataset(
        grouped_docs=grouped_docs,
        queries_per_doc=3,
        max_docs_per_type=35
    )

    print(f"\n생성된 평가 데이터: {len(qa_dataset)}개")
    qa_dataset.to_parquet('qa_dataset_k1.parquet')