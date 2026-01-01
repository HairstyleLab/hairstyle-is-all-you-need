import pandas as pd
from langchain_community.vectorstores import FAISS


def extract_corpus_from_vectorstore(vectorstore: FAISS) -> pd.DataFrame:
    corpus_data = []

    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)

        if doc is not None:
            corpus_data.append({
                'doc_id': doc_id,
                'contents': doc.page_content,
                'metadata': doc.metadata
            })

    corpus_df = pd.DataFrame(corpus_data)
    print(f"추출된 문서 수: {len(corpus_df)}")

    return corpus_df

if "__main__" == __name__:
    corpus_df = extract_corpus_from_vectorstore()
    corpus_df.to_parquet('corpus.parquet')
    print(f"Total documents: {len(corpus_df)}")