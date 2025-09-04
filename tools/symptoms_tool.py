#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_symptoms.py - Công cụ tìm triệu chứng theo pipeline RAG + Gemini (Phiên bản cải thiện)

Cải thiện chính:
- Chuẩn hóa đơn giản: Chỉ dùng ViTokenizer để chuẩn hóa câu hỏi trong JSON.
- Quản lý lỗi và phụ thuộc tốt hơn: Kiểm tra phụ thuộc, khởi tạo mô hình rõ ràng.
- Hiệu suất: Lưu trữ embedding vào file, sử dụng IndexHNSW cho FAISS.
- Tìm kiếm: Kết hợp semantic và fuzzy search với trọng số, ngưỡng động.
- Trích xuất triệu chứng: Mở rộng regex, thêm few-shot prompt cho LLM.
- Đầu ra: Trả về JSON với extracted_symptoms và validation.
- Bảo trì: Sử dụng config dict, docstring chi tiết.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cập nhật: pipeline RAG cho PDF (mỗi trang = 1 dòng).
- _load_pdf(): mỗi trang -> 1 row (page_id, content, content_norm)
- _init_embedding_index(): build embeddings cho mỗi trang, cache + FAISS
- retrieval: semantic + fuzzy, merge, rerank (cosine) và trả top-K
- search_symptoms(): lấy top-2 sau rerank -> gửi LLM để synthesize
"""

import os
import re
import json
import logging
from typing import List, Dict, Any
from rapidfuzz import process, fuzz
import pandas as pd
from sentence_transformers import util, SentenceTransformer
import faiss
import pickle
from pyvi import ViTokenizer
from cleantext import clean
import fitz  # PyMuPDF để đọc PDF
import numpy as np
from sklearn.mixture import GaussianMixture
from itertools import groupby
import regex
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CONFIG
CONFIG = {
    "pdf_path": "data/data_chuong_164 166.pdf",
    "embed_cache_path": "data/embeddings_cache.pkl",
    "faiss_index_path": "data/faiss_index.index",
    "top_k": 5,
    "fuzzy_threshold": 60,
    "semantic_weight": 0.7,
    "fuzzy_weight": 0.3,
    "merge_threshold": 0.5,
    "embed_model_name": "bkai-foundation-models/vietnamese-bi-encoder",
    #"embed_model_name": "sentence-transformers/all-mpnet-base-v2",
    "embed_fallback": "all-MiniLM-L6-v2",
    "gemini_model": os.getenv("GOOGLE_GENAI_MODEL", "gemini-1.5-flash"),
    "top_k_for_llm": 3
}

# ====== Stopwords ======
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS_BASE = set(stopwords.words("vietnamese"))
except:
    STOPWORDS_BASE = set()

STOPWORDS = STOPWORDS_BASE.union({
    "là", "và", "có", "bị", "ở", "trong", "khi", "các", "những",
    "tôi", "bạn", "thì", "một", "này", "đang", "được", "với", "cho",
    "của", "tại", "trên", "dưới", "gì", "nào", "ai", "đó", "đây",
    "kia", "mà", "như", "lại", "còn", "đã", "chỉ", "mỗi", "để",
    "từ", "ra", "vào", "lên", "xuống", "nếu", "vì", "bởi", "do",
    "nên", "thế", "nhưng", "hay", "hoặc", "chưa", "rằng", "nữa",
    "luôn", "vẫn", "đều", "rất", "quá", "hết", "cùng", "theo",
    "về", "bằng", "ngoài", "giữa", "trước", "sau", "kể", "từng",
    "chẳng", "chứ", "mới", "đi", "làm", "nói", "nghe", "thấy",
    "bên", "nơi", "chỗ", "hơn", "ít", "nhiều", "vài", "tất", "cả",
    "mấy", "ai", "cái", "con", "người", "việc", "nào", "đâu",
    "thôi", "đấy", "ấy", "vậy", "thế", "nào", "bao", "giờ", "khiến",
    "bệnh", "triệu chứng", "cảm thấy", "thường", "thỉnh thoảng", "đột ngột",
    "mạnh", "yếu", "nhẹ", "nặng", "liên tục", "gián đoạn", "tăng", "giảm"
})

# ====== Kiểm tra phụ thuộc và API key ======
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not set")
    HAS_LLM = False
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        HAS_LLM = True
    except ImportError:
        logger.warning("langchain_google_genai not installed. LLM disabled.")
        HAS_LLM = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDER = True
except ImportError:
    logger.warning("SentenceTransformer not available. Falling back to fuzzy search only.")
    HAS_EMBEDDER = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    logger.warning("FAISS not available. Semantic search disabled.")
    HAS_FAISS = False

# Globals
_df: pd.DataFrame = None
_corpus: List[str] = []
_corpus_norm: List[str] = []
_corpus_for_embedding: List[str] = []
_meta: List[Dict[str, Any]] = []
_index: Any = None
_embed_model: Any = None
_gemini_model: Any = None


# ====== STEP 1: Load PDF với DEMO pipeline ======

import regex
import numpy as np
from itertools import groupby
from pyvi import ViTokenizer  # nếu không có thì comment lại

# Trích xuất ký tự từ 1 trang PDF
def extract_chars(page):
    """
    Trích xuất ký tự từ 1 trang PDF.
    Trả về list dict: {char, x, y, x1, y1, font_size, block, line}
    """
    try:
        chars = page.get_text("chars")
        out = []
        for c in chars:
            x0, y0, x1, y1, ch, bno, lno, *_ = c
            font_size = max(1.0, (y1 - y0))
            out.append({
                "char": ch, "x": x0, "y": y0, "x1": x1, "y1": y1,
                "font_size": font_size, "block": int(bno), "line": int(lno)
            })
        return sorted(out, key=lambda r: (r['block'], r['line'], round(r['y'],3), r['x']))
    except Exception:
        pass

    # fallback dùng page.get_text("dict")
    try:
        pg = page.get_text("dict")
    except Exception:
        return []

    out = []
    for b_index, block in enumerate(pg.get("blocks", [])):
        for l_index, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                bbox = span.get("bbox", None)
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = bbox
                else:
                    x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0
                width = max(1e-6, x1 - x0)
                n_chars = len(text)
                avg_cw = width / max(1, n_chars)
                cur_x = x0
                font_size = max(1.0, span.get("size", (y1-y0) if y1>y0 else 10.0))
                for ch in text:
                    ch_w = avg_cw
                    out.append({
                        "char": ch,
                        "x": cur_x, "y": y0, "x1": cur_x+ch_w, "y1": y1,
                        "font_size": font_size,
                        "block": int(b_index), "line": int(l_index)
                    })
                    cur_x += ch_w
    return sorted(out, key=lambda r: (r['block'], r['line'], round(r['y'],3), r['x']))

# Gom ký tự thành từng dòng (runs)
def group_runs(chars):
    runs = []
    for (b,l), group in groupby(chars, key=lambda r: (r['block'], r['line'])):
        run = sorted(list(group), key=lambda r: r['x'])
        runs.append(run)
    return runs

# Phát hiện vị trí cần chèn khoảng trắng
def detect_inserts_for_run(run, min_norm_threshold=0.8):
    if len(run) < 2:
        return []
    xs = [g['x'] for g in run]
    sizes = [g['font_size'] for g in run]
    deltas = [(xs[i+1] - xs[i]) / max(sizes[i], 1e-6) for i in range(len(xs)-1)]
    arr = np.array(deltas).reshape(-1,1)

    if np.all(arr <= min_norm_threshold):
        return []

    inserts = []
    try:
        if len(arr) >= 4:
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=2, random_state=0, n_init=3,
                                 covariance_type="full").fit(arr)
            labs = gm.predict(arr)
            means = [arr[labs==k].mean() if np.any(labs==k) else 0.0 for k in (0,1)]
            big_label = int(np.argmax(means))
            inserts = [i for i,label in enumerate(labs)
                       if label == big_label and arr[i][0] > min_norm_threshold]
        else:
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            thr = med + 3.0 * (mad if mad > 0 else 1e-3)
            inserts = [i for i,v in enumerate(deltas) if v > thr and v > min_norm_threshold]
    except Exception:
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        thr = med + 3.0 * (mad if mad > 0 else 1e-3)
        inserts = [i for i,v in enumerate(deltas) if v > thr and v > min_norm_threshold]
    return inserts

# Xây dựng text từ run
def build_text_for_run(run, inserts):
    txt = []
    for i,g in enumerate(run):
        txt.append(g['char'])
        if i in inserts:
            txt.append(" ")
    return "".join(txt)

# Xử lý 1 trang PDF
def fix_page_text(page, hyphen_merge=True):
    chars = extract_chars(page)
    runs = group_runs(chars)

    processed_runs = []
    for run in runs:
        inserts = detect_inserts_for_run(run)
        txt = build_text_for_run(run, inserts)
        txt = txt.replace('\n',' ').strip()
        processed_runs.append(txt)

    # xử lý nối từ bị gạch nối
    out_parts = []
    for i,part in enumerate(processed_runs):
        if i>0 and hyphen_merge and out_parts:
            prev = out_parts[-1]
            if prev.endswith('-') and len(part)>0 and part[0].islower():
                out_parts[-1] = prev[:-1] + part
                continue
        out_parts.append(part)

    page_line = " ".join([p for p in out_parts if p])
    page_line = regex.sub(r'\s+', ' ', page_line).strip()
    return page_line

def _load_pdf() -> pd.DataFrame:
    """
    Load PDF: mỗi trang -> 1 dòng text
    Trả về DataFrame với content để embedding/truy vấn.
    """
    logger.debug("[LOAD_PDF] Loading PDF dataset (1 page = 1 row)...")
    global _df
    if "_df" in globals() and _df is not None:
        return _df

    path = CONFIG.get("pdf_path", "data/dataset.pdf")
    if not os.path.exists(path):
        logger.error(f"PDF not found at {path}")
        return pd.DataFrame(columns=["page_id", "content", "content_norm"])

    import fitz
    doc = fitz.open(path)

    rows = []
    for i, page in enumerate(doc, start=1):
        page_line = fix_page_text(page)
        if not page_line:
            continue
        try:
            norm_text = ViTokenizer.tokenize(page_line.lower())
        except Exception:
            norm_text = page_line.lower()

        rows.append({
            "page_id": i,
            "content": page_line,
            "content_norm": norm_text
        })

    _df = pd.DataFrame(rows)
    logger.info(f"[LOAD_PDF] Loaded {len(_df)} pages from PDF")
    return _df

# ------------------- Init embeddings & FAISS (cache) -------------------
def _init_embedding_index():
    """
    Build embeddings for each page (one vector per page), cache embeddings + faiss index.
    Cache keys: last_modified timestamp of pdf file.
    """
    global _corpus, _meta, _index, _embed_model, _corpus_norm, _corpus_for_embedding
    pdf_path = CONFIG["pdf_path"]
    cache_path = CONFIG["embed_cache_path"]
    faiss_path = CONFIG["faiss_index_path"]

    df = _load_pdf()
    if df.empty:
        logger.warning("No pages to index.")
        return

    last_modified = os.path.getmtime(pdf_path)

    # Try to load cache
    if os.path.exists(cache_path) and os.path.exists(faiss_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("last_modified") == last_modified:
                _corpus = cached["corpus"]
                _corpus_norm = cached["corpus_norm"]
                _corpus_for_embedding = cached["corpus_for_embedding"]
                _meta = cached["meta"]
                # load faiss
                try:
                    _index = faiss.read_index(faiss_path)
                    try:
                        _index.hnsw.efSearch = 64
                    except Exception:
                        pass
                    logger.info("Loaded embedding cache + faiss index.")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load faiss index: {e}")
            else:
                logger.info("PDF changed on disk, rebuilding embedding cache.")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")

    # Build corpus and meta from df
    _corpus = []
    _corpus_norm = []
    _meta = []
    for i, row in df.iterrows():
        page_id = int(row["page_id"])
        content = str(row["content"])
        content_norm = str(row["content_norm"])
        _corpus.append(content)
        _corpus_norm.append(content_norm)
        _meta.append({
            "page_id": page_id,
            "content": content,
            "content_norm": content_norm
        })

    _corpus_for_embedding = _corpus_norm.copy()

    # Init embed model if possible
    if HAS_EMBEDDER and _embed_model is None:
        try:
            _embed_model = SentenceTransformer(CONFIG["embed_model_name"])
        except Exception as e:
            logger.warning("Failed to load embed model %s: %s. Falling back to %s",
                           CONFIG["embed_model_name"], e, CONFIG["embed_fallback"])
            _embed_model = SentenceTransformer(CONFIG["embed_fallback"])

    # Build FAISS index if possible
    if HAS_EMBEDDER and HAS_FAISS and _embed_model:
        try:
            logger.info("Encoding %d pages for FAISS index...", len(_corpus_for_embedding))
            corpus_emb = _embed_model.encode(_corpus_for_embedding, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(corpus_emb)
            dim = corpus_emb.shape[1]
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 64
            index.add(corpus_emb)
            _index = index

            # cache
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "last_modified": last_modified,
                    "corpus": _corpus,
                    "corpus_norm": _corpus_norm,
                    "corpus_for_embedding": _corpus_for_embedding,
                    "meta": _meta
                }, f)
            faiss.write_index(_index, faiss_path)
            logger.info("Built and cached FAISS index.")
            return
        except Exception as e:
            logger.warning("Failed to build faiss index: %s", e)

    # If cannot build embedding index, still keep corpus/meta for fuzzy fallback
    logger.info("Embedding/FAISS not available; falling back to fuzzy-only retrieval.")

# ====== STEP 3: Khởi tạo mô hình ======
def _init_models():
    global _embed_model, _gemini_model
    if HAS_EMBEDDER and _embed_model is None:
        try:
            _embed_model = SentenceTransformer(CONFIG["embed_model_name"])
        except Exception as e:
            logger.warning(f"Failed to load {CONFIG['embed_model_name']}: {e}. Falling back to {CONFIG['embed_fallback']}")
            _embed_model = SentenceTransformer(CONFIG["embed_fallback"])
    if HAS_LLM and _gemini_model is None:
        try:
            _gemini_model = ChatGoogleGenerativeAI(model=CONFIG["gemini_model"], google_api_key=os.getenv("GOOGLE_API_KEY"))
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            _gemini_model = None

# ====== STEP 4: Normalize Text ======
def _normalize_text(text: str) -> str:
    logger.debug(f"[NORMALIZE] Raw text: {text}")
    try:
        text = ViTokenizer.tokenize(text)
        words = text.split()
    except Exception as e:
        logger.warning(f"ViTokenizer failed: {e}, falling back to basic split")
        words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    text = " ".join(words).strip()
    text = re.sub(r"\s+", " ", text).strip()
    logger.debug(f"[NORMALIZE] Normalized text: {text}")
    return text

# ====== STEP 5: Extract Symptoms ======
def _extract_symptoms(query: str) -> List[str]:
    logger.debug(f"[SYMPTOM_EXTRACT] Input query: {query}")
    # 1) Thử LLM (nếu có)
    if HAS_LLM and _gemini_model:
        try:
            prompt = f"""
            Bạn là trợ lý y khoa. Liệt kê các triệu chứng chính trong câu dưới đây.
            Trả về một JSON array (ví dụ: ["đau đầu", "sốt cao"]) duy nhất, KHÔNG giải thích thêm.
            Câu: "{query}"
            """
            resp = _gemini_model.invoke(prompt)
            text = getattr(resp, "content", "") or str(resp or "")
            text = text.strip()
            if text:
                # 1a. thử load nguyên văn
                try:
                    symptoms = json.loads(text)
                except Exception:
                    # 1b. thử tìm JSON array bên trong (ví dụ LLM trả kèm giải thích)
                    m = re.search(r'(\[.*?\])', text, flags=re.S)
                    if m:
                        try:
                            symptoms = json.loads(m.group(1))
                        except Exception:
                            symptoms = None
                    else:
                        # 1c. fallback parse từng dòng (list bullet)
                        lines = []
                        for ln in text.splitlines():
                            ln2 = ln.strip().lstrip("-•*0123456789. \t")
                            if len(ln2) > 0 and len(ln2) < 100:
                                lines.append(ln2)
                        symptoms = lines if lines else None

                if isinstance(symptoms, list) and symptoms:
                    # normalize each symptom and return
                    return [_normalize_text(str(s)) for s in symptoms if str(s).strip()]
        except Exception as e:
            logger.warning(f"LLM symptom extraction failed: {e}")

    # 2) Fallback: regex mở rộng (nắm các triệu chứng y khoa hay gặp)
    patterns = [
        r"(đau [^.,;]+)", r"(ngứa [^.,;]+)", r"(sốt [^.,;]+)", r"(phát ban [^.,;]+)",
        r"(khó thở[^.,;]*)", r"(nôn|ói mửa|buồn nôn[^.,;]*)", r"(vàng da[^.,;]*)",
        r"(tiêu chảy|phân lỏng[^.,;]*)", r"(mệt mỏi|suy nhược[^.,;]*)", r"(ho [^.,;]+)",
        r"(chảy máu[^.,;]+)", r"(sưng [^.,;]+)", r"(mờ mắt[^.,;]*)", r"(rụng tóc[^.,;]*)",
        r"(khó nuốt[^.,;]*)", r"(cổ trướng[^.,;]*)", r"(lách to[^.,;]*)", r"(lách[^.,;]*)",
        r"(bệnh lý não[^.,;]*)", r"(chán ăn[^.,;]*)", r"(khó chịu[^.,;]*)"
    ]
    found = []
    for pat in patterns:
        matches = re.findall(pat, query, flags=re.IGNORECASE)
        found.extend(matches)
    # loại trùng, chuẩn hóa
    normalized = list(dict.fromkeys([_normalize_text(f) for f in found if f and f.strip()]))
    logger.debug(f"[SYMPTOM_EXTRACT] Extracted symptoms: {normalized}")
    return normalized




# ------------------- Semantic search -------------------
def _semantic_search(norm_query: str, k: int = 5):
    logger.debug("Semantic search for: %s", norm_query)
    global _index, _embed_model, _meta
    if _index is None or _embed_model is None or not HAS_FAISS:
        return []
    try:
        q_emb = _embed_model.encode([norm_query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        D, I = _index.search(q_emb, k)
        hits = []
        if len(I) > 0:
            for dist, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1:
                    continue
                score = max(0.0, 1.0 - 0.5 * float(dist))
                meta = _meta[idx] if idx < len(_meta) else {"page_id": idx, "content": _corpus[idx]}
                hits.append({
                    "score": score,
                    "meta": meta,
                    "text": meta.get("content_norm", meta.get("content", "")),
                    "corpus_id": idx
                })
        return hits
    except Exception as e:
        logger.warning("Semantic search failed: %s", e)
        return []

# ------------------- Fuzzy search -------------------
def _fuzzy_search(query: str, threshold: int = None, top_n: int = 5):
    if threshold is None:
        threshold = CONFIG["fuzzy_threshold"]
    global _corpus_norm, _corpus, _meta
    if not _corpus_norm:
        df = _load_pdf()
        _corpus_norm = [str(r["content_norm"]) for _, r in df.iterrows()]

    try:
        q_norm = ViTokenizer.tokenize(query.lower()).replace("_", " ")
    except Exception:
        q_norm = query.lower()

    hits = process.extract(q_norm, _corpus_norm, scorer=fuzz.token_sort_ratio, score_cutoff=threshold, limit=top_n)
    results = []
    for match_text, score, idx in hits:
        meta = _meta[idx] if idx < len(_meta) else {"page_id": idx, "content": _corpus[idx]}
        results.append({
            "score": float(score) / 100.0,
            "meta": meta,
            "text": meta.get("content", _corpus[idx]),
            "corpus_id": idx,
            "matched_text": match_text
        })
    return results

# ------------------- Merge semantic + fuzzy -------------------
def _merge_results(sem_hits, fuzzy_hits):
    merged = {}
    for hit in (sem_hits or []):
        idx = hit["corpus_id"]
        score = float(hit.get("score", 0.0)) * CONFIG["semantic_weight"]
        merged[idx] = {
            "idx": idx,
            "page_id": hit["meta"].get("page_id"),
            "content": hit["meta"].get("content"),
            "content_norm": hit["meta"].get("content_norm"),
            "similarity": score,
            "matched_text": hit.get("text")
        }
    for hit in (fuzzy_hits or []):
        idx = hit["corpus_id"]
        score = float(hit.get("score", 0.0)) * CONFIG["fuzzy_weight"]
        if idx in merged:
            merged[idx]["similarity"] = max(merged[idx]["similarity"], score)
        else:
            merged[idx] = {
                "idx": idx,
                "page_id": hit["meta"].get("page_id"),
                "content": hit["meta"].get("content"),
                "content_norm": hit["meta"].get("content_norm"),
                "similarity": score,
                "matched_text": hit.get("matched_text", hit.get("text"))
            }
    results = sorted(merged.values(), key=lambda x: x["similarity"], reverse=True)
    return results

# ------------------- Rerank using embedder cosine similarity -------------------
def _rerank_results(query: str, merged_results: List[Dict], top_k: int = 5) -> List[Dict]:
    if not merged_results or _embed_model is None:
        return merged_results[:top_k]
    try:
        q_emb = _embed_model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        corpus_texts = [r["content_norm"] if r.get("content_norm") else r["content"] for r in merged_results]
        corpus_emb = _embed_model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(q_emb, corpus_emb)[0].cpu().tolist()
        for r, sc in zip(merged_results, scores):
            r["similarity"] = float(sc)
        reranked = sorted(merged_results, key=lambda x: x["similarity"], reverse=True)
        return reranked[:top_k]
    except Exception as e:
        logger.warning("Rerank failed: %s", e)
        return merged_results[:top_k]

# ------------------- Format output for PDF -------------------
def format_output_pdf(query: str, norm_query: str, symptoms: List[str], results: List[Dict]) -> Dict:
    out = {
        "original_query": query,
        "normalized_query": norm_query,
        "extracted_symptoms": symptoms,
        "pdf_results": []
    }
    for r in results:
        snippet = (r.get("content") or "")[:400]  # first 400 chars
        out["pdf_results"].append({
            "page_id": r.get("page_id"),
            "corpus_idx": r.get("idx"),
            "similarity": float(r.get("similarity", 0.0)),
            "snippet": snippet,
            "full_text": r.get("content")
        })
    return out

def synthesize_answer_from_pages(query: str, top_pages: List[Dict]) -> str:
    logger.debug("[SYNTHESIZE] Query: %s", query)
    logger.debug("[SYNTHESIZE] Top pages: %s", [p.get("page_id") for p in top_pages])
    logger.debug("[SYNTHESIZE] Top pages content: %s", [{ "page_id": p.get("page_id"), "snippet": (p.get("content") or "")[:200]} for p in top_pages])
    if not HAS_LLM or _gemini_model is None:
        logger.warning("[SYNTHESIZE] LLM not available")
        return "Xin lỗi, hiện tại không thể sinh câu trả lời vì LLM chưa được cấu hình."

    pages_text = [f"Page {p.get('page_id')}: {(p.get('content') or '')[:1200]}" for p in top_pages]
    prompt = f"""
    Bạn là trợ lý thông minh. Người dùng hỏi: "{query}"
    Triệu chứng được trích xuất: {', '.join(top_pages[0].get('symptoms', [])) if top_pages and 'symptoms' in top_pages[0] else 'Không rõ'}
    Hệ thống đã tìm được những nội dung liên quan trong file PDF:
    {chr(10).join(pages_text) if pages_text else 'Không tìm thấy nội dung liên quan.'}
    Yêu cầu:
    1) Tổng hợp thông tin chính từ các trang trên (nếu có) hoặc dựa trên triệu chứng được trích xuất.
    2) Nếu không có nội dung PDF phù hợp, đưa ra gợi ý bệnh dựa trên triệu chứng và giải thích ngắn gọn.
    3) Trả kết quả bằng tiếng Việt, ngắn gọn, rõ ràng.
    4) Kết thúc bằng 1-2 câu khuyến cáo (ví dụ: kiểm tra thêm, tham khảo chuyên gia).
    5) Chỉ sử dụng thông tin từ PDF hoặc triệu chứng; không suy diễn ngoài dữ liệu.
    Trả về một JSON object với các khóa:
    - "answer": (string) nội dung trả lời tóm tắt,
    - "sources": [ {"page_id": ..., "excerpt": "..."} ] (liệt kê 3 nguồn nếu có, hoặc rỗng),
    - "advice": (string)
    Không trả lời thêm ngoài JSON này.
    """
    logger.debug("[SYNTHESIZE] Prompt sent to LLM: %s", prompt)
    try:
        resp = _gemini_model.invoke(prompt)
        text = getattr(resp, "content", "") or str(resp or "")
        logger.debug("[SYNTHESIZE] LLM response: %s", text)
        # Thử parse JSON để kiểm tra
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            logger.error("[SYNTHESIZE] JSON parsing failed: %s", str(e))
            return json.dumps({
                "answer": "Không thể xác định bệnh từ dữ liệu hiện tại.",
                "sources": [],
                "advice": "Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn."
            }, ensure_ascii=False)
    except Exception as e:
        logger.error("[SYNTHESIZE] Failed: %s", str(e))
        return json.dumps({
            "answer": "Không thể xác định bệnh từ dữ liệu hiện tại.",
            "sources": [],
            "advice": "Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn."
        }, ensure_ascii=False)

# ------------------- Main retrieval wrapper -------------------
def retrieve_and_rerank(query: str, top_k: int = 5):
    norm_query = _normalize_text(query)
    sem = _semantic_search(norm_query, k=top_k)
    fuzzy = _fuzzy_search(query, threshold=CONFIG["fuzzy_threshold"], top_n=top_k)
    merged = _merge_results(sem, fuzzy)
    reranked = _rerank_results(norm_query, merged, top_k=top_k)
    return norm_query, reranked

# ------------------- Tool entrypoint -------------------
from pydantic import BaseModel
from typing import Literal
class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'completed'
    message: str
    data: dict = None

from langchain_core.tools import tool
@tool
def search_symptoms(user_query: str) -> Dict:
    """
    Tìm kiếm và tổng hợp thông tin từ file PDF theo câu hỏi của người dùng.

    Mô tả tổng quan
    ---------------
    Hàm này nhận một câu hỏi (user_query), thực hiện pipeline RAG (Retrieval-Augmented Generation)
    trên một file PDF đã cấu hình trong `CONFIG["pdf_path"]` theo quy tắc: **mỗi trang PDF được coi là một
    document (một dòng/corpus)**. Quy trình gồm:
      1. Khởi tạo mô hình embedding và index (nếu chưa có): `_init_models()`, `_init_embedding_index()`.
      2. Chuẩn hoá và (tuỳ chọn) trích xuất các từ khoá/triệu chứng từ câu hỏi: `_normalize_text()`, `_extract_symptoms()`.
      3. Truy vấn tìm kiếm ngữ nghĩa (semantic search) trên FAISS + fallback fuzzy matching.
      4. Gộp kết quả semantic + fuzzy, rồi rerank bằng cosine similarity embedding.
      5. Lấy **top N** kết quả đã rerank (mặc định top 2) và gửi vào LLM để tổng hợp câu trả lời.
      6. Trả về cấu trúc dữ liệu (Pydantic `ResponseFormat` -> dict) chứa dữ liệu tìm kiếm và phần tổng hợp.

    Tham số
    --------
    user_query : str
        Câu hỏi / truy vấn của người dùng (tiếng Việt). Ví dụ: "Bệnh nhân đau ngực và khó thở, cần làm gì?"

    Giá trị trả về
    -------------
    Trả về một dict theo Pydantic `ResponseFormat` với các trường chính:
      - status: 'input_required' | 'completed' | 'error'
      - message: thông điệp tóm tắt (string)
      - data: dict hoặc None, nội dung gồm:
          * original_query: (str) - câu truy vấn gốc.
          * normalized_query: (str) - câu truy vấn đã normalize/tokenize.
          * extracted_symptoms: (List[str]) - danh sách thuật ngữ/triệu chứng được cố gắng trích xuất.
          * pdf_results: (List[dict]) - danh sách các trang phù hợp (mỗi phần tử có):
                - page_id: (int) số trang trong PDF
                - corpus_idx: (int) chỉ số trong corpus
                - similarity: (float) điểm tương đồng sau merge/rerank
                - snippet: (str) đoạn cắt ngắn (first ~400 chars) để hiển thị nhanh
                - full_text: (str) nội dung đầy đủ của trang
          * synthesized_answer / synthesized_answer_raw: kết quả LLM (nếu có). `synthesized_answer`
            có thể là JSON parsed nếu LLM trả đúng định dạng, còn `synthesized_answer_raw` giữ text thô.

    Side effects (tác động phụ)
    ---------------------------
      - Gọi `_init_embedding_index()` có thể tạo/ghi các file cache:
          * CONFIG["embed_cache_path"] (pickle) chứa text + meta
          * CONFIG["faiss_index_path"] (FAISS index)
      - Tải model embedding (SentenceTransformer) và FAISS index vào bộ nhớ.
      - Nếu LLM được cấu hình (GOOGLE_API_KEY + ChatGoogleGenerativeAI), sẽ thực hiện request tới API.
      - Ghi log quá trình (logger).

    Yêu cầu (Dependencies & Config)
    ------------------------------
      - Các package nên có: PyMuPDF (fitz), sentence-transformers, faiss, rapidfuzz, pyvi, clean-text, langchain_core...
      - Biến môi trường: `GOOGLE_API_KEY` (nếu muốn dùng Gemini/Google GenAI).
      - Cấu hình chính trong mã (CONFIG):
          * "pdf_path" : đường dẫn tới file PDF
          * "embed_cache_path", "faiss_index_path"
          * "embed_model_name", "embed_fallback"
          * "top_k_for_llm" (số trang gửi cho LLM; mặc định 2)

    Xử lý lỗi & hành vi khi thiếu resource
    ---------------------------------------
      - Nếu FAISS hoặc embedding không khả dụng, hàm sẽ fallback về fuzzy text matching.
      - Nếu LLM không được cấu hình, `synthesized_answer` sẽ chứa thông báo rằng LLM không khả dụng,
        nhưng phần `pdf_results` vẫn trả bình thường.
      - Hàm cố gắng bọc một số lỗi bằng try/except; nếu có lỗi lớn (ví dụ file PDF không tồn tại),
        return status='error' và message mô tả lỗi.

    Ví dụ (tóm tắt)
    ---------------
    >>> res = search_symptoms("Bệnh nhân sốt cao 39 độ, đau họng, ho khan")
    >>> # res['data'] sẽ chứa pdf_results (các trang liên quan) và synthesized_answer_raw nếu LLM có.

    Ghi chú bổ sung
    ----------------
      - Vì hàm được trang trí bởi `@tool` (langchain_core.tools.tool), docstring này cũng được dùng
        như description cho tool; nên giữ mô tả rõ ràng, ngắn gọn ở các phần tiêu đề khi cần.
      - Nếu muốn thay đổi hành vi (ví dụ: tăng số trang gửi cho LLM), chỉnh `CONFIG["top_k_for_llm"]`.
    """

    # 1. init
    _init_models()
    _init_embedding_index()

    # 2. extract symptoms
    norm_query, _ = (None, None)
    try:
        norm_query = _normalize_text(user_query)
    except Exception:
        norm_query = user_query.lower()

    symptoms = _extract_symptoms(user_query)

    # 3. retrieve + rerank
    _, reranked = retrieve_and_rerank(user_query, top_k=CONFIG.get("top_k", 5))

    # 4. format output (pdf)
    output = format_output_pdf(user_query, norm_query, symptoms, reranked[:CONFIG.get("top_k",5)])

    # 5. synthesize using top-N (we send top 2)
    top_for_llm = reranked[: CONFIG.get("top_k_for_llm", 2)]
    if top_for_llm and HAS_LLM and _gemini_model:
        try:
            answer_text = synthesize_answer_from_pages(user_query, top_for_llm)
            # try to parse JSON; if not JSON, keep raw text
            try:
                parsed = json.loads(answer_text)
                output["synthesized_answer"] = parsed
            except Exception:
                output["synthesized_answer_raw"] = answer_text
        except Exception as e:
            logger.warning("Synthesis failed: %s.", e)
            output["synthesized_answer"] = None
    else:
        output["synthesized_answer"] = "LLM không khả dụng hoặc không có kết quả để tổng hợp."

    synthesized = output.get("synthesized_answer") or output.get("synthesized_answer_raw") or "Hoàn thành"
    message = json.dumps(synthesized, ensure_ascii=False) if isinstance(synthesized, dict) else str(synthesized)
    return ResponseFormat(
        status="completed",
        message=message,
        data=output
    ).dict()