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
    similarity_threshold: float = 0.85,
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

    print(f"\n[2] Embedding 모델 초기화")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"모델: {embedding_model}")
    print(f"디바이스: {device}")

    print(f"[3] GT 보강 시작 (유사도 임계값: {similarity_threshold})")

    enriched_dataset = []
    total_original_gt = 0
    total_enriched_gt = 0

    for idx, row in tqdm(qa_dataset.iterrows(), total=len(qa_dataset), desc="  처리 중"):
        query = row['query']
        original_gt = row['retrieval_gt']
        metadata_filter = row['metadata_filter']

        filter_for_search = {k: v for k, v in metadata_filter.items() if k != 'title'}

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

        query_embedding = embeddings.embed_query(query)

        doc_similarities = []
        for doc_row in filtered_docs:
            doc_embedding = embeddings.embed_query(doc_row['contents'])
            similarity = np.dot(query_embedding, doc_embedding)
            doc_similarities.append({
                'doc_id': doc_row['doc_id'],
                'similarity': similarity
            })

        enriched_gt = set(original_gt)
        for item in doc_similarities:
            if item['similarity'] >= similarity_threshold:
                enriched_gt.add(item['doc_id'])

        total_original_gt += len(original_gt)
        total_enriched_gt += len(enriched_gt)

        new_row = row.to_dict()
        new_row['retrieval_gt'] = list(enriched_gt)
        enriched_dataset.append(new_row)

    print(f"\n[4] 결과 저장")
    enriched_df = pd.DataFrame(enriched_dataset)

    if output_path.endswith('.parquet'):
        enriched_df.to_parquet(output_path, index=False)
    else:
        enriched_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"  저장 경로: {output_path}")

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

    SIMILARITY_THRESHOLD = 0.85

    if os.path.exists(QA_HAIRSTYLE_PATH):
        print("\n" + "="*70)
        print(" HAIRSTYLE 데이터셋 GT 보강")
        print("="*70)
        enrich_ground_truth(
            qa_dataset_path=QA_HAIRSTYLE_PATH,
            corpus_path=CORPUS_PATH,
            output_path=OUTPUT_HAIRSTYLE,
            similarity_threshold=SIMILARITY_THRESHOLD
        )

    if os.path.exists(QA_HAIRCOLOR_PATH):
        print("\n" + "="*70)
        print(" HAIRCOLOR 데이터셋 GT 보강")
        print("="*70)
        enrich_ground_truth(
            qa_dataset_path=QA_HAIRCOLOR_PATH,
            corpus_path=CORPUS_PATH,
            output_path=OUTPUT_HAIRCOLOR,
            similarity_threshold=SIMILARITY_THRESHOLD
        )

    print("모든 GT 보강 완료!")


if __name__ == "__main__":
    main()
