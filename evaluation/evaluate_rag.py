import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGEvaluator:

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        embedding_model: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        device: str = "cpu"
    ):
        self.corpus_df = corpus_df
        self.embedding_model = embedding_model
        self.device = device

        print(f"임베딩 모델 로드 중: {embedding_model} (device: {device})")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_vectorstore(
        self,
        chunk_size: int,
        chunk_overlap: int,
        category_filter: str = None
    ) -> FAISS:

        if category_filter:
            filtered_df = self.corpus_df[
                self.corpus_df['metadata'].apply(lambda x: x.get('category') == category_filter)
            ]
            print(f"  카테고리 '{category_filter}' 필터링: {len(filtered_df)}개 문서")
        else:
            filtered_df = self.corpus_df

        documents = []
        for _, row in filtered_df.iterrows():
            doc = Document(
                page_content=row['contents'],
                metadata={
                    **row['metadata'],
                    'doc_id': row['doc_id']
                }
            )
            documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"  총 {len(chunks)}개 chunk 생성")

        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        return vectorstore

    def evaluate_retrieval(
        self,
        vectorstore: FAISS,
        qa_dataset: pd.DataFrame,
        top_k: int
    ) -> Dict[str, float]:

        recalls = []
        precisions = []

        for _, row in tqdm(qa_dataset.iterrows(), total=len(qa_dataset), desc=f"  평가 중 (top-k={top_k})"):
            query = row['query']
            gt_doc_ids = row['retrieval_gt']  # List[str]
            metadata_filter = row.get('metadata_filter', {})
            filter_for_search = {k: v for k, v in metadata_filter.items() if k != 'title'}

            fetch_k = len(vectorstore.docstore._dict)

            top_k_docs = vectorstore.similarity_search(
                query=query,
                k=top_k,
                fetch_k=fetch_k,
                filter=filter_for_search
            )

            retrieved_doc_ids = [doc.metadata.get('doc_id') for doc in top_k_docs]

            gt_set = set(gt_doc_ids)
            retrieved_set = set(retrieved_doc_ids)

            recall = len(gt_set & retrieved_set) / len(gt_set) if len(gt_set) > 0 else 0
            recalls.append(recall)

            precision = len(gt_set & retrieved_set) / len(retrieved_set) if len(retrieved_set) > 0 else 0
            precisions.append(precision)

        avg_recall = np.mean(recalls)
        avg_precision = np.mean(precisions)

        return {
            'recall': avg_recall,
            'precision': avg_precision,
            'num_queries': len(qa_dataset)
        }

    def grid_search(
        self,
        qa_dataset: pd.DataFrame,
        chunk_sizes: List[int] = [100, 200, 300, 500],
        chunk_overlaps: List[int] = [0, 50, 100],
        top_ks: List[int] = [1, 3, 5, 10],
        category_filter: str = None
    ) -> pd.DataFrame:

        results = []

        total_experiments = len(chunk_sizes) * len(chunk_overlaps) * len(top_ks)
        print(f"\n총 {total_experiments}개 실험 시작")
        print(f"Chunk sizes: {chunk_sizes}")
        print(f"Chunk overlaps: {chunk_overlaps}")
        print(f"Top-k: {top_ks}")
        if category_filter:
            print(f"Category: {category_filter}")

        experiment_num = 0

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                if chunk_overlap >= chunk_size:
                    continue

                print(f"\n{'='*70}")
                print(f"VectorStore 생성: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
                print(f"{'='*70}")

                vectorstore = self.create_vectorstore(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    category_filter=category_filter
                )

                for top_k in top_ks:
                    experiment_num += 1
                    print(f"\n[{experiment_num}/{total_experiments}] Evaluation: top_k={top_k}")

                    metrics = self.evaluate_retrieval(
                        vectorstore=vectorstore,
                        qa_dataset=qa_dataset,
                        top_k=top_k
                    )

                    result = {
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap,
                        'top_k': top_k,
                        'recall': metrics['recall'],
                        'precision': metrics['precision'],
                        'f1': 2 * metrics['recall'] * metrics['precision'] / (metrics['recall'] + metrics['precision']) if (metrics['recall'] + metrics['precision']) > 0 else 0,
                        'num_queries': metrics['num_queries']
                    }

                    results.append(result)

                    print(f"  → Recall: {metrics['recall']:.4f}")
                    print(f"  → Precision: {metrics['precision']:.4f}")
                    print(f"  → F1: {result['f1']:.4f}")

        results_df = pd.DataFrame(results)
        return results_df


def main():
    CORPUS_PATH = "./output/corpus.parquet"
    QA_HAIRSTYLE_PATH = "./output/qa_dataset_hairstyle.parquet"
    QA_HAIRCOLOR_PATH = "./output/qa_dataset_haircolor.parquet"
    OUTPUT_DIR = "./evaluation_results"

    CHUNK_SIZES = [100, 200, 300, 500]
    CHUNK_OVERLAPS = [0, 50, 100]
    TOP_KS = [1, 3, 5, 10]

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print(" RAG 시스템 평가 시작")
    print("="*70)

    print(f"\n[1] Corpus 로드: {CORPUS_PATH}")
    corpus_df = pd.read_parquet(CORPUS_PATH)
    print(f"  총 {len(corpus_df)}개 문서")

    print(f"\n[2] Evaluator 초기화")
    evaluator = RAGEvaluator(
        corpus_df=corpus_df,
        device=DEVICE
    )

    print(f"\n{'='*70}")
    print(" Hairstyle 카테고리 평가")
    print(f"{'='*70}")

    qa_hairstyle = pd.read_parquet(QA_HAIRSTYLE_PATH)
    print(f"평가 데이터셋: {len(qa_hairstyle)}개 쿼리")

    results_hairstyle = evaluator.grid_search(
        qa_dataset=qa_hairstyle,
        chunk_sizes=CHUNK_SIZES,
        chunk_overlaps=CHUNK_OVERLAPS,
        top_ks=TOP_KS,
        category_filter='hairstyle'
    )

    hairstyle_result_path = os.path.join(OUTPUT_DIR, 'results_hairstyle.csv')
    results_hairstyle.to_csv(hairstyle_result_path, index=False, encoding='utf-8-sig')
    print(f"\nHairstyle 결과 저장: {hairstyle_result_path}")

    print(f"\n{'='*70}")
    print(" Haircolor 카테고리 평가")
    print(f"{'='*70}")

    qa_haircolor = pd.read_parquet(QA_HAIRCOLOR_PATH)
    print(f"평가 데이터셋: {len(qa_haircolor)}개 쿼리")

    results_haircolor = evaluator.grid_search(
        qa_dataset=qa_haircolor,
        chunk_sizes=CHUNK_SIZES,
        chunk_overlaps=CHUNK_OVERLAPS,
        top_ks=TOP_KS,
        category_filter='haircolor'
    )

    haircolor_result_path = os.path.join(OUTPUT_DIR, 'results_haircolor.csv')
    results_haircolor.to_csv(haircolor_result_path, index=False, encoding='utf-8-sig')
    print(f"\nHaircolor 결과 저장: {haircolor_result_path}")

    print(f"\n{'='*70}")
    print(" 최적 설정")
    print(f"{'='*70}")

    best_hairstyle = results_hairstyle.sort_values('f1', ascending=False).iloc[0]
    print(f"\n[Hairstyle]")
    print(f"  Best F1 Score: {best_hairstyle['f1']:.4f}")
    print(f"  - chunk_size: {best_hairstyle['chunk_size']}")
    print(f"  - chunk_overlap: {best_hairstyle['chunk_overlap']}")
    print(f"  - top_k: {best_hairstyle['top_k']}")
    print(f"  - recall: {best_hairstyle['recall']:.4f}")
    print(f"  - precision: {best_hairstyle['precision']:.4f}")

    best_haircolor = results_haircolor.sort_values('f1', ascending=False).iloc[0]
    print(f"\n[Haircolor]")
    print(f"  Best F1 Score: {best_haircolor['f1']:.4f}")
    print(f"  - chunk_size: {best_haircolor['chunk_size']}")
    print(f"  - chunk_overlap: {best_haircolor['chunk_overlap']}")
    print(f"  - top_k: {best_haircolor['top_k']}")
    print(f"  - recall: {best_haircolor['recall']:.4f}")
    print(f"  - precision: {best_haircolor['precision']:.4f}")

    print(f"\n{'='*70}")
    print(" 전체 통계")
    print(f"{'='*70}")
    print(f"Hairstyle 평균 Recall: {results_hairstyle['recall'].mean():.4f}")
    print(f"Hairstyle 평균 Precision: {results_hairstyle['precision'].mean():.4f}")
    print(f"Haircolor 평균 Recall: {results_haircolor['recall'].mean():.4f}")
    print(f"Haircolor 평균 Precision: {results_haircolor['precision'].mean():.4f}")

    print(f"\n평가 완료!")
    print(f"결과 파일:")
    print(f"  - {hairstyle_result_path}")
    print(f"  - {haircolor_result_path}")


if __name__ == "__main__":
    main()
