import pandas as pd
from typing import Dict
import random


def group_by_metadata(corpus_df: pd.DataFrame, max_per_category: int = 100) -> Dict:
    if len(corpus_df) == 0:
        print("Corpus가 비어있습니다!")
        return {}

    print("\n메타데이터 구조 분석 중...")
    sample_metadata = corpus_df.iloc[0]['metadata']
    print(f"샘플 메타데이터 키: {list(sample_metadata.keys())}")
    print(f"샘플 메타데이터 값: {sample_metadata}")

    type_key = 'category'
    print(f"→ 타입 식별 키로 '{type_key}' 사용")

    all_categories = set()
    for _, row in corpus_df.iterrows():
        metadata = row['metadata']
        category = metadata.get(type_key, 'unknown')
        all_categories.add(category)

    print(f"\n발견된 카테고리: {all_categories}")

    target_categories = {
        'hairstyle': 'hairstyle_feature',
        'haircolor': 'haircolor_feature'
    }

    print(f"\n평가 대상 카테고리: {list(target_categories.keys())}")

    groups = {}
    for _, row in corpus_df.iterrows():
        metadata = row['metadata']
        category = metadata.get(type_key, 'unknown')

        if category in target_categories:
            mapped_name = target_categories[category]

            if mapped_name not in groups:
                groups[mapped_name] = []
            groups[mapped_name].append(row)

    print(f"\n카테고리별 샘플링 (각 {max_per_category}개):")
    for category_name, docs in groups.items():
        total = len(docs)
        print(f"  - {category_name}: {total}개 → ", end="")

        if total > max_per_category:
            sampled_docs = random.sample(docs, max_per_category)
            groups[category_name] = sampled_docs
            print(f"{max_per_category}개 샘플링")
        else:
            print(f"{total}개 전체 사용")

    print("\n최종 그룹화 결과:")
    for doc_type, docs in groups.items():
        print(f"  - {doc_type}: {len(docs)}개 문서")

    return groups

if "__main__" == __name__:
    grouped_docs = group_by_metadata(corpus_df)
    print(f"Hairstyle docs: {len(grouped_docs['hairstyle_feature'])}")
    print(f"Haircolor docs: {len(grouped_docs['haircolor_feature'])}")
    print(f"Faceshape docs: {len(grouped_docs['faceshape'])}")