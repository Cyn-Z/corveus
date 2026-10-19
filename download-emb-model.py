from sentence_transformers import SentenceTransformer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
output_dir = "./models/all-MiniLM-L6-v2"

# Download the model and its required tokenizer/configuration files.
model = SentenceTransformer(model_id)

# Save a self-contained local copy for offline use.
model.save(output_dir)

print(f"Model saved to: {output_dir}")