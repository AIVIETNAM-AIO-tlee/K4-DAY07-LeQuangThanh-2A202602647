import os
import time
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import RecursiveChunker
from src.agent import KnowledgeBaseAgent


CACHE_FILE = Path(".embeddings_cache.json")


def load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict[str, list[float]]):
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


class CachedGeminiEmbedder:
    def __init__(self, model_name: str = "gemini-embedding-001"):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.cache = load_cache()

    def get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for i, t in enumerate(texts):
            h = self.get_hash(t)
            if h in self.cache:
                results[i] = self.cache[h]
            else:
                missing_indices.append(i)
                missing_texts.append(t)

        if missing_texts:
            print(f"Embedding {len(missing_texts)} uncached texts via Gemini (already cached: {len(self.cache)})...")
            batch_size = 20
            idx_ptr = 0
            while idx_ptr < len(missing_texts):
                batch = missing_texts[idx_ptr : idx_ptr + batch_size]
                batch_indices = missing_indices[idx_ptr : idx_ptr + batch_size]
                try:
                    res = self.client.models.embed_content(model=self.model_name, contents=batch)
                    for orig_idx, emb in zip(batch_indices, res.embeddings):
                        vec = [float(v) for v in emb.values]
                        results[orig_idx] = vec
                        self.cache[self.get_hash(texts[orig_idx])] = vec
                    save_cache(self.cache)
                    idx_ptr += len(batch)
                    print(f"  Progress: {idx_ptr}/{len(missing_texts)} (Total cached: {len(self.cache)})")
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Rate limit or error: {e}. Sleeping 25s...")
                    time.sleep(25.0)

        return results

    def __call__(self, text: str) -> list[float]:
        h = self.get_hash(text)
        if h in self.cache:
            return self.cache[h]
        for attempt in range(5):
            try:
                res = self.client.models.embed_content(model=self.model_name, contents=text)
                vec = [float(v) for v in res.embeddings[0].values]
                self.cache[h] = vec
                save_cache(self.cache)
                return vec
            except Exception:
                time.sleep(20)
        raise RuntimeError("Failed to embed query after retries")


def parse_markdown_with_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            content = parts[2].strip()
            meta = {}
            for line in fm_text.splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
            return meta, content
    return {}, text


def run_benchmark():
    embedder = CachedGeminiEmbedder()
    chunker = RecursiveChunker(chunk_size=500)
    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)

    corpus_dir = Path("data/shopee-ecommerce")
    target_files = ["77251.md", "77250.md", "77265.md"]
    md_files = [corpus_dir / f for f in target_files if (corpus_dir / f).exists()]

    all_docs: list[Document] = []
    print(f"Reading {len(md_files)} policy files...")
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        meta, content = parse_markdown_with_frontmatter(text)
        meta.setdefault("source", str(md_file))
        meta.setdefault("doc_id", md_file.stem)

        chunks = chunker.chunk(content)
        for i, c in enumerate(chunks):
            all_docs.append(
                Document(
                    id=f"{md_file.stem}#{i}",
                    content=c,
                    metadata={**meta, "chunk_index": i},
                )
            )

    print(f"Total chunks to store: {len(all_docs)}")
    embedder.embed_batch([d.content for d in all_docs])
    store.add_documents(all_docs)
    print(f"Stored {store.get_collection_size()} chunks in EmbeddingStore.\n")

    queries = [
        {
            "id": 1,
            "query": "Người mua có thể gửi yêu cầu trả hàng và hoàn tiền trong vòng bao nhiêu ngày kể từ khi nhận hàng?",
            "filter": None,
        },
        {
            "id": 2,
            "query": "Điều kiện để Người mua được yêu cầu trả hàng do không còn nhu cầu là gì?",
            "filter": None,
        },
        {
            "id": 3,
            "query": "Quy định về việc khiếu nại và đổi trả đối với Người mua (buyer) là gì?",
            "filter": {"audience": "buyer"},
        },
        {
            "id": 4,
            "query": "Người bán có được phép tự tổ chức vận chuyển hàng hóa trên sàn Shopee không?",
            "filter": None,
        },
        {
            "id": 5,
            "query": "Shopee hỗ trợ giải quyết tranh chấp, khiếu nại giữa Người Mua và Người Bán thông qua cơ chế nào?",
            "filter": None,
        },
    ]

    def simple_agent_llm(prompt: str) -> str:
        lines = [
            line.strip()
            for line in prompt.splitlines()
            if line.strip()
            and not line.startswith("Context:")
            and not line.startswith("Question:")
            and not line.startswith("Answer:")
            and not line.startswith("Answer the question")
            and not line.startswith("[")
        ]
        return f"{' '.join(lines)[:180]}..."

    agent = KnowledgeBaseAgent(store=store, llm_fn=simple_agent_llm)

    output_lines = []
    output_lines.append("=== KẾT QUẢ BENCHMARK RETRIEVAL ===")
    output_lines.append(f"Số file: {len(md_files)} | Tổng số chunks: {len(all_docs)}\n")

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        filt = q["filter"]

        results = store.search_with_filter(query_text, top_k=3, metadata_filter=filt)
        agent_ans = agent.answer(query_text, top_k=3)

        top1 = results[0] if results else None
        print(f"--- Query {qid}: {query_text} ---")
        if filt:
            print(f"Filter: {filt}")
        if top1:
            print(f"Top-1 Chunk ID: {top1['id']} (Score: {top1['score']:.4f})")
            print(f"Preview: {top1['content'][:150].replace(chr(10), ' ')}...")
        print(f"Agent answer: {agent_ans[:120]}...\n")

        output_lines.append(f"Query {qid}: {query_text}")
        if filt:
            output_lines.append(f"Filter: {filt}")
        if top1:
            output_lines.append(f"Top-1 ID: {top1['id']} | Score: {top1['score']:.4f}")
            output_lines.append(f"Top-1 Content: {top1['content'][:250].replace(chr(10), ' ')}")
        output_lines.append(f"Agent answer: {agent_ans}")
        output_lines.append("-" * 50)

    Path("ket_qua_benchmark.txt").write_text("\n".join(output_lines), encoding="utf-8")
    print("Done! Results saved to ket_qua_benchmark.txt")


if __name__ == "__main__":
    run_benchmark()
