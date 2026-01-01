import pandas as pd
from langchain_community.vectorstores import FAISS


def extract_corpus_from_chunked_vectorstore(
    vectorstore: FAISS,
    source_key: str = 'source',
    chunk_index_key: str = 'chunk_index',
) -> pd.DataFrame:

    print(f"원본 문서 재구성 중 (source_key: '{source_key}')...")

    all_chunks = []
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if doc is not None:
            all_chunks.append({
                'chunk_id': doc_id,
                'contents': doc.page_content,
                'metadata': doc.metadata
            })

    chunks_df = pd.DataFrame(all_chunks)
    print(f"총 {len(chunks_df)}개 chunk 추출됨")

    if source_key not in chunks_df.iloc[0]['metadata']:
        print(f"경고: 메타데이터에 '{source_key}' 필드가 없습니다.")
        print(f"사용 가능한 메타데이터 키: {list(chunks_df.iloc[0]['metadata'].keys())}")

        possible_keys = ['source', 'title', 'link', 'document_id', 'doc_id']
        for key in possible_keys:
            if key in chunks_df.iloc[0]['metadata']:
                source_key = key
                print(f"'{source_key}'를 대신 사용합니다.")
                break
        else:
            print("원본 문서를 식별할 수 있는 메타데이터 키를 찾을 수 없습니다.")
            print("Chunk된 DB를 그대로 사용합니다 (품질이 떨어질 수 있음)")
            return extract_corpus_without_reconstruction(chunks_df)

    documents = {}

    for _, row in chunks_df.iterrows():
        metadata = row['metadata']
        source_id = metadata.get(source_key, 'unknown')

        if source_id not in documents:
            documents[source_id] = {
                'chunks': [],
                'metadata': metadata.copy()
            }

        chunk_info = {
            'content': row['contents'],
            'index': metadata.get(chunk_index_key, 999999)
        }
        documents[source_id]['chunks'].append(chunk_info)

    corpus_data = []
    for doc_id, doc_info in documents.items():

        sorted_chunks = sorted(doc_info['chunks'], key=lambda x: x['index'])
        combined_content = '\n'.join([chunk['content'] for chunk in sorted_chunks])

        corpus_data.append({
            'doc_id': doc_id,
            'contents': combined_content,
            'metadata': doc_info['metadata'],
            'num_chunks': len(sorted_chunks)
        })

    corpus_df = pd.DataFrame(corpus_data)

    print(f"\n재구성 완료:")
    print(f"  - 원본 문서 수: {len(corpus_df)}")
    print(f"  - 평균 chunk 수: {corpus_df['num_chunks'].mean():.1f}")
    print(f"  - 평균 문서 길이: {corpus_df['contents'].str.len().mean():.0f}자")

    corpus_df = corpus_df.drop(columns=['num_chunks'])

    return corpus_df


def extract_corpus_without_reconstruction(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 문서 재구성 없이 chunk를 그대로 사용
    (품질이 떨어질 수 있음)
    """
    print("경고: chunk를 독립적인 문서로 처리합니다. 평가 품질이 떨어질 수 있습니다.")

    corpus_data = []
    for _, row in chunks_df.iterrows():
        corpus_data.append({
            'doc_id': row['chunk_id'],
            'contents': row['contents'],
            'metadata': row['metadata']
        })

    return pd.DataFrame(corpus_data)


def auto_detect_and_extract(vectorstore: FAISS) -> pd.DataFrame:
    sample_doc_id = list(vectorstore.index_to_docstore_id.values())[0]
    sample_doc = vectorstore.docstore.search(sample_doc_id)

    print("DB 구조 자동 감지 중...")
    print(f"샘플 문서 길이: {len(sample_doc.page_content)}자")
    print(f"샘플 메타데이터 키: {list(sample_doc.metadata.keys())}")

    avg_length = len(sample_doc.page_content)
    has_chunk_info = any(key in sample_doc.metadata for key in ['chunk', 'chunk_index', 'page'])

    if avg_length < 300 or has_chunk_info:
        print("Chunk된 DB로 감지됨")

        for key in ['source', 'title', 'link', 'document_id', 'keyword']:
            if key in sample_doc.metadata:
                print(f"원본 문서 식별 키: '{key}'")
                return extract_corpus_from_chunked_vectorstore(vectorstore, source_key=key)

        print("원본 문서 식별 키를 찾을 수 없음. chunk를 독립 문서로 처리합니다.")

        all_chunks = []
        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(doc_id)
            if doc is not None:
                all_chunks.append({
                    'chunk_id': doc_id,
                    'contents': doc.page_content,
                    'metadata': doc.metadata
                })
        return extract_corpus_without_reconstruction(pd.DataFrame(all_chunks))
    else:
        from corpus import extract_corpus_from_vectorstore
        return extract_corpus_from_vectorstore(vectorstore)


if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.load_local(
        "./db/new_hf_1211",
        embeddings,
        allow_dangerous_deserialization=True
    )

    corpus_df = auto_detect_and_extract(vectorstore)

    print(f"\n최종 추출된 문서 수: {len(corpus_df)}")
    print("\n샘플:")
    print(corpus_df.head(2))
