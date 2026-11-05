from sentence_transformers import SentenceTransformer

print("Loading sentence embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "A viral video shows a minister resigning.",
    "A fake resignation video was circulated online."
]

embeddings = model.encode(
    sentences,
    convert_to_numpy=True
)

print("\nEmbedding test successful!")
print("Embedding shape:", embeddings.shape)
print("Vector dimension:", len(embeddings[0]))