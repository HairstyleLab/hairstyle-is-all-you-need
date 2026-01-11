import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dataset_generation.corpus import extract_corpus_from_vectorstore
from dataset_generation.corpus_chunked import auto_detect_and_extract
from dataset_generation.metadata import group_by_metadata
from dataset_generation.generation import create_retrieval_dataset
from dataset_generation.check import verify_generated_dataset, quality_check


def main():

    DB_PATH = "./db/new_hf_1211"
    OUTPUT_DIR = "./output"

    DOCS_PER_CATEGORY = 150
    QUERIES_PER_DOC = 1
    AUTO_DETECT_CHUNKS = True

    SAVE_SEPARATE_FILES = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print(" RAG 평가 데이터셋 생성 시작")
    print(" (카테고리별 100개씩 별도 데이터셋)")
    print("="*70)

    print("\n[Step 1] VectorStore 로드 중...")
    try:
        embedding_model = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Embedding 디바이스: {device}")
        except:
            device = 'cpu'
            print(f"Embedding 디바이스: {device} (torch 없음)")

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"VectorStore 로드 완료: {DB_PATH}")
    except Exception as e:
        print(f"VectorStore 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[Step 2] Corpus 추출 중...")
    try:
        if AUTO_DETECT_CHUNKS:
            print("자동 감지 모드 활성화 (chunk 여부 자동 판단)")
            corpus_df = auto_detect_and_extract(vectorstore)
        else:
            corpus_df = extract_corpus_from_vectorstore(vectorstore)

        corpus_path = os.path.join(OUTPUT_DIR, 'corpus.parquet')
        corpus_df.to_parquet(corpus_path)
        print(f"Corpus 저장 완료: {corpus_path}")
        print(f"   총 {len(corpus_df)}개 문서")

        print("\n[Corpus 샘플]")
        if len(corpus_df) > 0:
            print(corpus_df.head(3)[['doc_id', 'metadata']])
            print(f"\n평균 문서 길이: {corpus_df['contents'].str.len().mean():.0f}자")
    except Exception as e:
        print(f"Corpus 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[Step 3] 문서 그룹화 및 샘플링 중...")
    try:
        grouped_docs = group_by_metadata(corpus_df, max_per_category=DOCS_PER_CATEGORY)

        total_docs = sum(len(docs) for docs in grouped_docs.values())
        print(f"\n총 선택된 문서 수: {total_docs}개")
    except Exception as e:
        print(f"문서 그룹화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n[Step 4] 평가 데이터셋 생성 중...")
    print(f"  - 문서당 쿼리 수: {QUERIES_PER_DOC}")
    print(f"  - 카테고리별 문서 수: {DOCS_PER_CATEGORY}")

    category_datasets = {}

    try:
        all_qa_dataset = create_retrieval_dataset(
            grouped_docs=grouped_docs,
            queries_per_doc=QUERIES_PER_DOC,
            max_docs_per_type=None,
            shuffle=True
        )

        if SAVE_SEPARATE_FILES:
            print("\n카테고리별 데이터셋 분리 중...")
            for category in ['hairstyle_feature', 'haircolor_feature', 'faceshape']:
                if category in grouped_docs:
                    category_df = all_qa_dataset[all_qa_dataset['search_type'] == category]
                    category_datasets[category] = category_df
                    print(f"  - {category}: {len(category_df)}개 평가 항목")

    except Exception as e:
        print(f"평가 데이터셋 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[Step 5] 품질 체크...")
    try:
        all_qa_dataset = quality_check(all_qa_dataset)
    except Exception as e:
        print(f"품질 체크 중 오류 발생: {e}")

    print("\n[Step 6] 데이터셋 저장 중...")
    saved_files = []

    try:
        all_qa_path = os.path.join(OUTPUT_DIR, 'qa_dataset_all.parquet')
        all_qa_dataset.to_parquet(all_qa_path, index=False)
        print(f"통합 평가 데이터셋 저장: {all_qa_path}")
        print(f"   총 {len(all_qa_dataset)}개 평가 항목")
        saved_files.append(all_qa_path)

        all_csv_path = os.path.join(OUTPUT_DIR, 'qa_dataset_all.csv')
        all_qa_dataset.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
        saved_files.append(all_csv_path)

        if SAVE_SEPARATE_FILES and category_datasets:
            print("\n카테고리별 파일 저장 중...")

            category_names = {
                'hairstyle_feature': 'hairstyle',
                'haircolor_feature': 'haircolor',
                'faceshape': 'face'
            }

            for category, df in category_datasets.items():
                short_name = category_names.get(category, category)

                cat_path = os.path.join(OUTPUT_DIR, f'qa_dataset_{short_name}.parquet')
                df.to_parquet(cat_path, index=False)
                print(f"{short_name}: {cat_path} ({len(df)}개 항목)")
                saved_files.append(cat_path)

                cat_csv_path = os.path.join(OUTPUT_DIR, f'qa_dataset_{short_name}.csv')
                df.to_csv(cat_csv_path, index=False, encoding='utf-8-sig')
                saved_files.append(cat_csv_path)

    except Exception as e:
        print(f"저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[Step 7] 샘플 검증")
    verify = input("샘플 검증을 진행하시겠습니까? (y/n): ")
    if verify.lower() == 'y':
        sample_size = int(input("검증할 샘플 수를 입력하세요 (기본: 10): ") or "10")
        verify_generated_dataset(all_qa_dataset, corpus_df, sample_size=sample_size)

    print("\n" + "="*70)
    print(" 평가 데이터셋 생성 완료!")
    print("="*70)
    print(f"\n생성된 파일:")
    print(f"  - {corpus_path}")
    for file_path in saved_files:
        print(f"  - {file_path}")

    print(f"\n카테고리별 데이터셋:")
    for category, short_name in category_names.items():
        if category in category_datasets:
            print(f"  - {short_name}: {DOCS_PER_CATEGORY}개 문서 x {QUERIES_PER_DOC}개 쿼리 = {len(category_datasets[category])}개 평가 항목")


if __name__ == "__main__":
    main()
