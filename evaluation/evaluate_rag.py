import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
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
        reranker_model: Optional[str] = None,
        device: str = "cpu"
    ):
        self.corpus_df = corpus_df
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.device = device

        print(f"임베딩 모델 로드 중: {embedding_model} (device: {device})")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 리랭커 초기화
        self.reranker = None
        if reranker_model:
            print(f"리랭커 모델 로드 중: {reranker_model}")
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model, device=device)
                print(f"  리랭커 로드 완료")
            except ImportError:
                print("  경고: sentence-transformers 설치 필요 (pip install sentence-transformers)")
                self.reranker = None

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

    def _rerank_documents(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """리랭커로 문서 재정렬"""
        if not self.reranker or len(documents) == 0:
            return documents[:top_k]

        # 쿼리-문서 쌍 생성
        pairs = [[query, doc.page_content] for doc in documents]

        # 리랭커로 스코어 계산
        scores = self.reranker.predict(pairs)

        # 스코어로 정렬
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 반환
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

        return reranked_docs

    def _calculate_mrr(self, gt_doc_ids: List[str], retrieved_doc_ids: List[str]) -> float:
        """Mean Reciprocal Rank 계산"""
        for i, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in gt_doc_ids:
                return 1.0 / i
        return 0.0

    def _calculate_ndcg(self, gt_doc_ids: List[str], retrieved_doc_ids: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain 계산"""
        # DCG 계산
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids[:k], 1):
            if doc_id in gt_doc_ids:
                dcg += 1.0 / np.log2(i + 1)

        # IDCG 계산 (이상적인 경우)
        idcg = 0.0
        for i in range(1, min(len(gt_doc_ids), k) + 1):
            idcg += 1.0 / np.log2(i + 1)

        # NDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def evaluate_retrieval(
        self,
        vectorstore: FAISS,
        qa_dataset: pd.DataFrame,
        top_k: int,
        fetch_k_multiplier: int = 20
    ) -> Dict[str, float]:

        recalls = []
        precisions = []
        mrrs = []
        ndcgs = []

        # 리랭커 사용 시 20개 문서를 가져와서 재정렬
        initial_fetch_k = 20 if self.reranker else top_k

        desc = f"  평가 중 (top-k={top_k}, reranker={'ON' if self.reranker else 'OFF'})"
        for _, row in tqdm(qa_dataset.iterrows(), total=len(qa_dataset), desc=desc):
            query = row['query']
            gt_doc_ids = row['retrieval_gt']  # List[str]
            metadata_filter = row.get('metadata_filter', {})
            filter_for_search = {k: v for k, v in metadata_filter.items() if k != 'title'}

            max_fetch_k = len(vectorstore.docstore._dict)

            # 초기 검색
            initial_docs = vectorstore.similarity_search(
                query=query,
                k=min(initial_fetch_k, max_fetch_k),
                fetch_k=max_fetch_k,
                filter=filter_for_search
            )

            # 리랭커 적용 (있는 경우)
            if self.reranker:
                top_k_docs = self._rerank_documents(query, initial_docs, top_k)
            else:
                top_k_docs = initial_docs[:top_k]

            retrieved_doc_ids = [doc.metadata.get('doc_id') for doc in top_k_docs]

            gt_set = set(gt_doc_ids)
            retrieved_set = set(retrieved_doc_ids)

            # Recall & Precision
            recall = len(gt_set & retrieved_set) / len(gt_set) if len(gt_set) > 0 else 0
            recalls.append(recall)

            precision = len(gt_set & retrieved_set) / len(retrieved_set) if len(retrieved_set) > 0 else 0
            precisions.append(precision)

            # MRR
            mrr = self._calculate_mrr(gt_doc_ids, retrieved_doc_ids)
            mrrs.append(mrr)

            # NDCG@k
            ndcg = self._calculate_ndcg(gt_doc_ids, retrieved_doc_ids, top_k)
            ndcgs.append(ndcg)

        return {
            'recall': np.mean(recalls),
            'precision': np.mean(precisions),
            'mrr': np.mean(mrrs),
            'ndcg': np.mean(ndcgs),
            'num_queries': len(qa_dataset)
        }

    def grid_search(
        self,
        qa_dataset: pd.DataFrame,
        chunk_sizes: List[int] = [100, 200],
        chunk_overlaps: List[int] = [0, 50, 100],
        top_ks: List[int] = [1,2,3],
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
        if self.reranker:
            print(f"Reranker: {self.reranker_model}")

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
                        'mrr': metrics['mrr'],
                        'ndcg': metrics['ndcg'],
                        'f1': 2 * metrics['recall'] * metrics['precision'] / (metrics['recall'] + metrics['precision']) if (metrics['recall'] + metrics['precision']) > 0 else 0,
                        'num_queries': metrics['num_queries']
                    }

                    results.append(result)

                    print(f"  → Recall: {metrics['recall']:.4f}")
                    print(f"  → Precision: {metrics['precision']:.4f}")
                    print(f"  → MRR: {metrics['mrr']:.4f}")
                    print(f"  → NDCG@{top_k}: {metrics['ndcg']:.4f}")
                    print(f"  → F1: {result['f1']:.4f}")

        results_df = pd.DataFrame(results)
        return results_df


def main():
    CORPUS_PATH = "./output/corpus.parquet"
    QA_HAIRSTYLE_PATH = "./output/qa_dataset_hairstyle_enriched.parquet"
    QA_HAIRCOLOR_PATH = "./output/qa_dataset_haircolor_enriched.parquet"
    OUTPUT_DIR = "./evaluation_results"

    CHUNK_SIZES = [100, 200]
    CHUNK_OVERLAPS = [0, 50, 100]
    TOP_KS = [1, 2, 3]

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 임베딩 모델 후보군
    # EMBEDDING_MODELS = [
    #     "dragonkue/snowflake-arctic-embed-l-v2.0-ko",  # 현재 (베이스라인)
    #     "jhgan/ko-sroberta-multitask",  # 1순위: 한국어 특화
    #     "intfloat/multilingual-e5-large",  # 2순위: 최고 성능
    #     "BM-K/KoSimCSE-roberta",  # 3순위: 경량 한국어
    #     "BAAI/bge-m3",  # 4순위: 최신 SOTA
    # ]

    EMBEDDING_MODELS = ["dragonkue/snowflake-arctic-embed-l-v2.0-ko"] 

    RERANKER_MODELS = [
        None,  # 리랭커 없음 (베이스라인)
        "Dongjin-kr/ko-reranker",  # 한국어 리랭커
        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 경량 크로스 인코더
        "BAAI/bge-reranker-base",  # BGE 리랭커
    ]

    print("="*70)
    print(" RAG 시스템 평가 시작 (임베딩 모델 비교)")
    print("="*70)
    print(f"총 {len(EMBEDDING_MODELS)}개 임베딩 모델 평가")
    print(f"리랭커: 없음 (베이스라인)")
    print("="*70)

    print(f"\n[1] Corpus 로드: {CORPUS_PATH}")
    corpus_df = pd.read_parquet(CORPUS_PATH)
    print(f"  총 {len(corpus_df)}개 문서")

    print(f"\n[2] QA 데이터셋 로드")
    qa_hairstyle = pd.read_parquet(QA_HAIRSTYLE_PATH)
    qa_haircolor = pd.read_parquet(QA_HAIRCOLOR_PATH)
    print(f"  Hairstyle: {len(qa_hairstyle)}개 쿼리")
    print(f"  Haircolor: {len(qa_haircolor)}개 쿼리")

    # 모델별 최고 성능 저장
    all_model_results = []

    # 각 임베딩 모델에 대해 평가 수행
    for model_idx, embedding_model in enumerate(EMBEDDING_MODELS, 1):
        print(f"\n{'#'*70}")
        print(f"# [{model_idx}/{len(EMBEDDING_MODELS)}] 임베딩 모델: {embedding_model}")
        print(f"{'#'*70}")

        # 모델명에서 안전한 파일명 생성
        model_name_safe = embedding_model.replace('/', '_').replace('.', '_')

        # 모델별 결과 폴더 생성
        model_output_dir = os.path.join(OUTPUT_DIR, model_name_safe)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"결과 폴더: {model_output_dir}")

        try:
            # Evaluator 초기화
            print(f"\n[Step 1] Evaluator 초기화")
            evaluator = RAGEvaluator(
                corpus_df=corpus_df,
                device=DEVICE,
                embedding_model=embedding_model,
                reranker_model="BAAI/bge-reranker-base"
            )

            # Hairstyle 평가
            print(f"\n[Step 2] Hairstyle 카테고리 평가")
            results_hairstyle = evaluator.grid_search(
                qa_dataset=qa_hairstyle,
                chunk_sizes=CHUNK_SIZES,
                chunk_overlaps=CHUNK_OVERLAPS,
                top_ks=TOP_KS,
                category_filter='hairstyle'
            )

            # 모델명 추가
            results_hairstyle['embedding_model'] = embedding_model

            # 결과 저장 (모델별 폴더에)
            hairstyle_result_path = os.path.join(model_output_dir, 'results_hairstyle_reranker_bge.csv')
            results_hairstyle.to_csv(hairstyle_result_path, index=False, encoding='utf-8-sig')
            print(f"\n  → 저장: {hairstyle_result_path}")

            # Haircolor 평가
            print(f"\n[Step 3] Haircolor 카테고리 평가")
            results_haircolor = evaluator.grid_search(
                qa_dataset=qa_haircolor,
                chunk_sizes=CHUNK_SIZES,
                chunk_overlaps=CHUNK_OVERLAPS,
                top_ks=TOP_KS,
                category_filter='haircolor'
            )

            # 모델명 추가
            results_haircolor['embedding_model'] = embedding_model

            # 결과 저장 (모델별 폴더에)
            haircolor_result_path = os.path.join(model_output_dir, 'results_haircolor_reranker_bge.csv')
            results_haircolor.to_csv(haircolor_result_path, index=False, encoding='utf-8-sig')
            print(f"  → 저장: {haircolor_result_path}")

            # 최고 성능 추출
            best_hairstyle = results_hairstyle.sort_values('f1', ascending=False).iloc[0]
            best_haircolor = results_haircolor.sort_values('f1', ascending=False).iloc[0]

            # 모델별 요약 저장
            model_summary = {
                'embedding_model': embedding_model,
                'hairstyle_best_f1': best_hairstyle['f1'],
                'hairstyle_best_recall': best_hairstyle['recall'],
                'hairstyle_best_precision': best_hairstyle['precision'],
                'hairstyle_best_mrr': best_hairstyle['mrr'],
                'hairstyle_best_ndcg': best_hairstyle['ndcg'],
                'hairstyle_avg_recall': results_hairstyle['recall'].mean(),
                'hairstyle_avg_mrr': results_hairstyle['mrr'].mean(),
                'hairstyle_avg_ndcg': results_hairstyle['ndcg'].mean(),
                'haircolor_best_f1': best_haircolor['f1'],
                'haircolor_best_recall': best_haircolor['recall'],
                'haircolor_best_precision': best_haircolor['precision'],
                'haircolor_best_mrr': best_haircolor['mrr'],
                'haircolor_best_ndcg': best_haircolor['ndcg'],
                'haircolor_avg_recall': results_haircolor['recall'].mean(),
                'haircolor_avg_mrr': results_haircolor['mrr'].mean(),
                'haircolor_avg_ndcg': results_haircolor['ndcg'].mean(),
            }
            all_model_results.append(model_summary)

            # 현재 모델 성능 출력
            print(f"\n[Step 4] {embedding_model} 성능 요약")
            print(f"  Hairstyle - Best F1: {best_hairstyle['f1']:.4f}, Avg MRR: {results_hairstyle['mrr'].mean():.4f}")
            print(f"  Haircolor - Best F1: {best_haircolor['f1']:.4f}, Avg MRR: {results_haircolor['mrr'].mean():.4f}")

        except Exception as e:
            print(f"\n  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 전체 모델 비교 결과 저장
    print(f"\n{'='*70}")
    print(" 전체 임베딩 모델 비교 결과")
    print(f"{'='*70}")

    comparison_df = pd.DataFrame(all_model_results)
    comparison_path = os.path.join(OUTPUT_DIR, 'embedding_models_comparison.csv')
    # comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n비교 결과 저장: {comparison_path}")

    # 최고 성능 모델 출력
    print(f"\n{'='*70}")
    print(" 최고 성능 모델")
    print(f"{'='*70}")

    best_hairstyle_model = comparison_df.sort_values('hairstyle_best_f1', ascending=False).iloc[0]
    print(f"\n[Hairstyle 카테고리]")
    print(f"  최고 모델: {best_hairstyle_model['embedding_model']}")
    print(f"  Best F1: {best_hairstyle_model['hairstyle_best_f1']:.4f}")
    print(f"  Best Recall: {best_hairstyle_model['hairstyle_best_recall']:.4f}")
    print(f"  Best MRR: {best_hairstyle_model['hairstyle_best_mrr']:.4f}")
    print(f"  Best NDCG: {best_hairstyle_model['hairstyle_best_ndcg']:.4f}")

    best_haircolor_model = comparison_df.sort_values('haircolor_best_f1', ascending=False).iloc[0]
    print(f"\n[Haircolor 카테고리]")
    print(f"  최고 모델: {best_haircolor_model['embedding_model']}")
    print(f"  Best F1: {best_haircolor_model['haircolor_best_f1']:.4f}")
    print(f"  Best Recall: {best_haircolor_model['haircolor_best_recall']:.4f}")
    print(f"  Best MRR: {best_haircolor_model['haircolor_best_mrr']:.4f}")
    print(f"  Best NDCG: {best_haircolor_model['haircolor_best_ndcg']:.4f}")

    print(f"\n{'='*70}")
    print(" 평가 완료!")
    print(f"{'='*70}")
    print(f"결과 구조:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── embedding_models_comparison.csv  (전체 비교)")
    for model in EMBEDDING_MODELS:
        model_safe = model.replace('/', '_').replace('.', '_')
        print(f"  ├── {model_safe}/")
        print(f"  │   ├── results_hairstyle.csv")
        print(f"  │   └── results_haircolor.csv")


if __name__ == "__main__":
    main()
