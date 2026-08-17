from src.rag.qa_pipeline import RAGPipeline
pipeline = RAGPipeline("data/processed/sections.csv")

contexts = pipeline.get_contexts("시간이 지남에 따른 부채비율 변화를 알려주세요.")
for i, ctx in enumerate(contexts):
    print(f"[{i+1}] {ctx['year']} {ctx['section']} : {ctx['text'][:20]}")
    if "부채비율" in ctx["text"].replace(" ", ""):
        print(" -> **부채비율 FOUND!**")
