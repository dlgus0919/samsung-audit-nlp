import pickle
with open("data/processed/chunks_meta.pkl", "rb") as f:
    chunks = pickle.load(f)

for chunk in chunks:
    text = chunk["text"].replace(" ", "")
    if "부채" in text and "비율" in text:
        print(f"FOUND IN: {chunk['year']} {chunk['section']}")
        print(chunk["text"][:100])
