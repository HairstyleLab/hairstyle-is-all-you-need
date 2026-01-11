import random
import pandas as pd
from typing import List, Dict
from dataset_generation.llm_providers import get_llm_provider

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
            f"""당신은 헤어스타일 특징을 추출하는 전문가입니다.

아래 문서에서 헤어스타일의 느낌, 성격, 시각적 특징만을 추출하세요.

문서:
{content}

중요 규칙:
1. 헤어스타일 명칭(펌, 컷, 스타일 이름 등)은 절대 추출 금지
   예: "젤리펌", "쉐도우펌", "레이어드컷" 등은 사용 불가
2. 오직 느낌과 특징만 추출
   예: "자연스러운", "볼륨감 있는", "단정한", "시원한"
3. 최대 3개 특징까지만
4. 특징들은 쉼표로 구분

잘못된 예시:
{{"queries": ["젤리펌"]}} ❌
{{"queries": ["쉐도우펌"]}} ❌
{{"queries": ["레이어드컷"]}} ❌

올바른 예시:
{{"queries": ["자연스러운, 부드러운"]}} ✓
{{"queries": ["볼륨감 있는, 화사한, 여성스러운"]}} ✓
{{"queries": ["단정한, 깔끔한"]}} ✓

반드시 JSON 형식으로만 반환:
{{"queries": ["추출한 특징"]}}""",
        'haircolor_feature':
            f"""당신은 헤어 컬러 특징을 추출하는 전문가입니다.

아래 문서에서 헤어 컬러의 느낌, 성격, 분위기만을 추출하세요.

문서:
{content}

중요 규칙:
1. 컬러 명칭(애쉬, 베이지, 브라운 등)은 절대 추출 금지
2. 오직 느낌과 분위기만 추출
   예: "따뜻한", "차가운", "화사한", "차분한"
3. 최대 3개 특징까지만
4. 특징들은 쉼표로 구분

잘못된 예시:
{{"queries": ["애쉬 브라운"]}} ❌
{{"queries": ["베이지 컬러"]}} ❌

올바른 예시:
{{"queries": ["따뜻한, 밝은"]}} ✓
{{"queries": ["차가운, 세련된, 도시적인"]}} ✓
{{"queries": ["자연스러운, 무난한"]}} ✓

반드시 JSON 형식으로만 반환:
{{"queries": ["추출한 특징"]}}"""
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