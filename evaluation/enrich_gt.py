import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import os
import ast

try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except:
    device = 'cpu'

print(f"사용 디바이스: {device}")


def enrich_ground_truth(
    qa_dataset_path: str,
    corpus_path: str,
    output_path: str,
    similarity_margin: float = 0.005,
    embedding_model="dragonkue/snowflake-arctic-embed-l-v2.0-ko"
):

    print("="*70)
    print("GT 보강 프로세스 시작")
    print("="*70)

    print(f"[1] 데이터 로드")

    if qa_dataset_path.endswith('.parquet'):
        qa_dataset = pd.read_parquet(qa_dataset_path)
    else:
        qa_dataset = pd.read_csv(qa_dataset_path)
        qa_dataset['metadata_filter'] = qa_dataset['metadata_filter'].apply(ast.literal_eval)
        qa_dataset['retrieval_gt'] = qa_dataset['retrieval_gt'].apply(ast.literal_eval)

    print(f"QA 데이터셋: {len(qa_dataset)}개 쿼리")

    corpus_df = pd.read_parquet(corpus_path)
    print(f"Corpus: {len(corpus_df)}개 문서")

    # Doc ID to content mapping 생성
    doc_id_to_content = {row['doc_id']: row['contents'] for _, row in corpus_df.iterrows()}

    print(f"\n[2] Embedding 모델 초기화")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"모델: {embedding_model}")
    print(f"디바이스: {device}")

    print(f"\n[3] GT 보강 시작 (쿼리별 동적 마진: {similarity_margin})")

    enriched_dataset = []
    total_original_gt = 0
    total_enriched_gt = 0

    for idx, row in tqdm(qa_dataset.iterrows(), total=len(qa_dataset), desc="  처리 중"):
        query = row['query']
        original_gt = row['retrieval_gt']
        metadata_filter = row['metadata_filter']

        # title을 제외한 메타데이터로 필터링
        filter_for_search = {k: v for k, v in metadata_filter.items() if k != 'title'}

        # 메타데이터로 문서 필터링
        filtered_docs = []
        for _, corpus_row in corpus_df.iterrows():
            corpus_metadata = corpus_row['metadata']
            match = True
            for key, value in filter_for_search.items():
                if corpus_metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_docs.append(corpus_row)

        if len(filtered_docs) == 0:
            enriched_dataset.append(row.to_dict())
            total_original_gt += len(original_gt)
            total_enriched_gt += len(original_gt)
            continue

        # 쿼리 임베딩 계산
        query_embedding = embeddings.embed_query(query)

        # 1. 먼저 GT 문서들과의 유사도 계산
        gt_similarities = []
        for gt_doc_id in original_gt:
            if gt_doc_id in doc_id_to_content:
                doc_content = doc_id_to_content[gt_doc_id]
                doc_embedding = embeddings.embed_query(doc_content)
                similarity = np.dot(query_embedding, doc_embedding)
                gt_similarities.append(similarity)

        # 2. 이 쿼리의 GT 최소 유사도 기준으로 threshold 설정
        if len(gt_similarities) > 0:
            min_gt_similarity = min(gt_similarities)
            query_threshold = max(0.0, min_gt_similarity - similarity_margin)
        else:
            # GT가 없거나 계산 실패시 기본값
            query_threshold = 0.80

        # 3. 필터링된 모든 문서와 유사도 계산
        doc_similarities = []
        for doc_row in filtered_docs:
            doc_embedding = embeddings.embed_query(doc_row['contents'])
            similarity = np.dot(query_embedding, doc_embedding)
            doc_similarities.append({
                'doc_id': doc_row['doc_id'],
                'similarity': similarity
            })

        # 4. threshold 이상인 문서들을 GT에 추가
        enriched_gt = set(original_gt)
        for item in doc_similarities:
            if item['similarity'] >= query_threshold:
                enriched_gt.add(item['doc_id'])

        total_original_gt += len(original_gt)
        total_enriched_gt += len(enriched_gt)

        new_row = row.to_dict()
        new_row['retrieval_gt'] = list(enriched_gt)
        enriched_dataset.append(new_row)

    print(f"\n[5] 결과 저장")
    enriched_df = pd.DataFrame(enriched_dataset)

    parquet_path = output_path.replace('.csv', '.parquet') if output_path.endswith('.csv') else output_path
    enriched_df.to_parquet(parquet_path, index=False)
    print(f"  Parquet 저장: {parquet_path}")

    csv_path = output_path.replace('.parquet', '.csv') if output_path.endswith('.parquet') else output_path
    enriched_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV 저장: {csv_path}")

    print(f"\n{'='*70}")
    print(" GT 보강 결과")
    print(f"{'='*70}")
    print(f"  원본 GT 평균 개수: {total_original_gt / len(qa_dataset):.2f}")
    print(f"  보강 GT 평균 개수: {total_enriched_gt / len(qa_dataset):.2f}")
    print(f"  증가율: {((total_enriched_gt - total_original_gt) / total_original_gt * 100):.1f}%")
    print(f"{'='*70}")


def main():
    """메인 함수"""

    QA_HAIRSTYLE_PATH = "./output/qa_dataset_hairstyle.parquet"
    QA_HAIRCOLOR_PATH = "./output/qa_dataset_haircolor.parquet"
    CORPUS_PATH = "./output/corpus.parquet"

    OUTPUT_HAIRSTYLE = "./output/qa_dataset_hairstyle_enriched.parquet"
    OUTPUT_HAIRCOLOR = "./output/qa_dataset_haircolor_enriched.parquet"

    # GT 최소 유사도에서 이 값만큼 빼서 threshold 설정
    SIMILARITY_MARGIN = 0.005

    if os.path.exists(QA_HAIRSTYLE_PATH):
        print("\n" + "="*70)
        print(" HAIRSTYLE 데이터셋 GT 보강")
        print("="*70)
        enrich_ground_truth(
            qa_dataset_path=QA_HAIRSTYLE_PATH,
            corpus_path=CORPUS_PATH,
            output_path=OUTPUT_HAIRSTYLE,
            similarity_margin=SIMILARITY_MARGIN
        )

    if os.path.exists(QA_HAIRCOLOR_PATH):
        print("\n" + "="*70)
        print(" HAIRCOLOR 데이터셋 GT 보강")
        print("="*70)
        enrich_ground_truth(
            qa_dataset_path=QA_HAIRCOLOR_PATH,
            corpus_path=CORPUS_PATH,
            output_path=OUTPUT_HAIRCOLOR,
            similarity_margin=SIMILARITY_MARGIN
        )

    print("모든 GT 보강 완료!")


if __name__ == "__main__":
    main()
