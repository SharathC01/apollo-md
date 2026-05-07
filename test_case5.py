from src.retrieve import embed_chunks, retrieve
from src.ingest_enhanced import load_all_papers_enhanced

print('Loading chunks...')
chunks = load_all_papers_enhanced('data/raw')
print(f'Total chunks: {len(chunks)}')

print('Embedding...')
chunks, embeddings = embed_chunks(chunks)

query = 'latent class analysis sepsis phenotype clustering'
results = retrieve(query, chunks, embeddings, top_k=5)

for r in results:
    print(f'Score: {r["score"]:.3f} | File: {r["source_file"]} | Page: {r["page"]}')
    print(f'Text: {r["text"][:200]}')
    print()