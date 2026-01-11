import pandas as pd


def verify_generated_dataset(qa_df: pd.DataFrame, corpus_df: pd.DataFrame, sample_size: int = 10):

    print("\n" + "="*70)
    print(" 데이터셋 샘플 검증")
    print("="*70 + "\n")

    actual_size = min(sample_size, len(qa_df))
    samples = qa_df.sample(n=actual_size, random_state=42)

    invalid_count = 0

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n{'='*70}")
        print(f"샘플 {i}/{actual_size}")
        print(f"{'='*70}")
        print(f"QID: {row['qid']}")
        print(f"검색 타입: {row['search_type']}")
        print(f"쿼리: {row['query']}")
        print(f"메타데이터 필터: {row['metadata_filter']}")

        gt_doc_id = row['retrieval_gt'][0]
        gt_docs = corpus_df[corpus_df['doc_id'] == gt_doc_id]

        if len(gt_docs) == 0:
            print(f"GT 문서를 찾을 수 없습니다: {gt_doc_id}")
            continue

        gt_doc = gt_docs.iloc[0]
        print(f"GT 문서 ID: {gt_doc['doc_id']}")
        print(f"GT 문서 메타데이터: {gt_doc['metadata']}")
        print(f"GT 문서 내용 (처음 300자):")
        print(f"{gt_doc['contents'][:300]}...")

        valid = input(f"\n이 쿼리가 이 문서를 찾기에 적합한가요? (y/n/s=skip): ")
        if valid.lower() == 's':
            print("건너뜀")
            continue
        elif valid.lower() != 'y':
            print("부적합한 쿼리로 표시됨")
            invalid_count += 1

    print("\n" + "="*70)
    print(f"검증 완료: {actual_size}개 샘플 중 {invalid_count}개 부적합")
    print("="*70)


def quality_check(qa_df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "="*70)
    print(" 데이터셋 품질 체크")
    print("="*70 + "\n")

    print("1. 검색 타입별 분포:")
    type_counts = qa_df['search_type'].value_counts()
    for search_type, count in type_counts.items():
        print(f"{search_type}: {count}개")

    qa_df_copy = qa_df.copy()
    qa_df_copy['query_length'] = qa_df_copy['query'].str.len()
    print("2. 쿼리 길이 통계:")
    stats = qa_df_copy['query_length'].describe()
    print(f"평균: {stats['mean']:.1f}자")
    print(f"최소: {stats['min']:.0f}자")
    print(f"최대: {stats['max']:.0f}자")
    print(f"중앙값: {stats['50%']:.1f}자")

    duplicates = qa_df['query'].duplicated().sum()
    print(f"3. 중복 쿼리: {duplicates}개")
    if duplicates > 0:
        print("중복 쿼리 예시:")
        dup_queries = qa_df[qa_df['query'].duplicated(keep=False)]['query'].unique()[:5]
        for query in dup_queries:
            count = (qa_df['query'] == query).sum()
            print(f"      \"{query}\" - {count}번 중복")

    empty = qa_df['query'].str.strip().str.len() == 0
    empty_count = empty.sum()
    print(f"4.빈 쿼리: {empty_count}개")
    if empty_count > 0:
        print("빈 쿼리가 발견되었습니다!")

    empty_gt = qa_df['retrieval_gt'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
    empty_gt_count = empty_gt.sum()
    print(f"5. GT 없는 항목: {empty_gt_count}개")
    if empty_gt_count > 0:
        print("GT가 없는 항목이 발견되었습니다!")

    print(f"\n{'='*70}")
    print("품질 체크 요약:")
    issues = []
    if duplicates > 0:
        issues.append(f"중복 쿼리 {duplicates}개")
    if empty_count > 0:
        issues.append(f"빈 쿼리 {empty_count}개")
    if empty_gt_count > 0:
        issues.append(f"GT 없음 {empty_gt_count}개")

    if issues:
        print(f"발견된 이슈: {', '.join(issues)}")
    else:
        print("품질 이슈 없음")
    print(f"{'='*70}")

    return qa_df


if "__main__" == __name__:
    verify_generated_dataset(qa_dataset, corpus_df, sample_size=10)
    qa_dataset = quality_check(qa_dataset)