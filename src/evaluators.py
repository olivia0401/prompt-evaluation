"""
Scoring functions for sentence- and keyword-extraction tasks.

Public surface:
  - score_sentence(prediction, ground_truth, task, embedder) -> SentenceScore
  - score_keywords(predicted, ground_truth)                  -> KeywordScore
  - parse_keywords(raw_output)                               -> (terms, errors)
  - normalize_keyword(kw)                                    -> stemmed lowercase form
  - EmbeddingClient                                          -> OpenAI embeddings + disk cache

Length ranges and word_count formula must stay in sync with prompts.txt.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Lazy heavy imports — keep import-time cheap.

# ---------- Length rules (from prompts.txt) ----------

LENGTH_RANGES = {
    "concept_relevant":  (10, 20),
    "position_relevant": (15, 25),  # adjusted 2026-05-18: was (20, 30), matched to GT length distribution
    "emotion_relevant":  (15, 25),
    "function_relevant": (15, 25),
    "benefit_relevant":  (15, 25),
    "category_relevant": (15, 20),
    "feature_relevant":  (15, 25),
    "context_relevant":  (15, 25),
}


# ---------- Sentence task scoring ----------

@dataclass
class SentenceScore:
    cosine: Optional[float]
    rouge_l: float
    word_count: int
    length_compliant: bool
    in_range: tuple[int, int]


def score_sentence(
    prediction: str,
    ground_truth: str,
    task: str,
    embedder: "EmbeddingClient",
) -> SentenceScore:
    """
    Score one sentence-task output against its ground truth.

    - cosine is None for empty predictions (do not coerce to 0; scoring vs
      empty is undefined). Callers should filter or mark these as parse_fail.
    """
    from src.utils import word_count

    pred = (prediction or "").strip()
    gt = (ground_truth or "").strip()

    lo, hi = LENGTH_RANGES.get(task, (1, 10**6))

    if not pred:
        return SentenceScore(None, 0.0, 0, False, (lo, hi))

    # Cosine via injected embedder (testable; allows fallback / offline mode)
    cosine = _cosine(embedder.embed(pred), embedder.embed(gt))

    # ROUGE-L
    rouge_l = _rouge_l(gt, pred)

    wc = word_count(pred)
    return SentenceScore(
        cosine=cosine,
        rouge_l=rouge_l,
        word_count=wc,
        length_compliant=(lo <= wc <= hi),
        in_range=(lo, hi),
    )


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


_ROUGE_SCORER = None
_ROUGE_LOCK = threading.Lock()


def _rouge_l(reference: str, hypothesis: str) -> float:
    global _ROUGE_SCORER
    with _ROUGE_LOCK:
        if _ROUGE_SCORER is None:
            from rouge_score import rouge_scorer
            _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(_ROUGE_SCORER.score(reference, hypothesis)["rougeL"].fmeasure)


# ---------- Keyword task scoring ----------

@dataclass
class KeywordScore:
    precision: float
    recall: float
    f1: float
    pred_count: int          # after normalize / dedupe
    gt_count: int
    true_positives: int
    pred_normalized: list[str]
    gt_normalized: list[str]


_STEMMER = None
_STEMMER_LOCK = threading.Lock()


def _stemmer():
    global _STEMMER
    with _STEMMER_LOCK:
        if _STEMMER is None:
            from nltk.stem import PorterStemmer
            _STEMMER = PorterStemmer()
    return _STEMMER


def normalize_keyword(kw: str) -> str:
    """
    Normalization rule:
      1. lowercase
      2. strip surrounding whitespace / punctuation
      3. collapse hyphens (so "non-dairy" matches "nondairy")
      4. Porter stem

    Returns empty string if nothing usable remains.
    """
    if not isinstance(kw, str):
        return ""
    s = kw.strip().lower()
    # Strip surrounding/internal punctuation except letters, digits, hyphens
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = s.replace("-", "")
    if not s:
        return ""
    return _stemmer().stem(s)


def score_keywords(predicted: list[str], ground_truth: list[str]) -> KeywordScore:
    """
    Compute P / R / F1 over normalized sets.

    Duplicate predictions are de-duped (a model emitting "food" and "foods"
    both stem to "food" — counts once).
    """
    pred_norm = [normalize_keyword(k) for k in (predicted or [])]
    gt_norm   = [normalize_keyword(k) for k in (ground_truth or [])]
    pred_set = {k for k in pred_norm if k}
    gt_set   = {k for k in gt_norm if k}

    tp = len(pred_set & gt_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return KeywordScore(
        precision=p,
        recall=r,
        f1=f1,
        pred_count=len(pred_set),
        gt_count=len(gt_set),
        true_positives=tp,
        pred_normalized=sorted(pred_set),
        gt_normalized=sorted(gt_set),
    )


# ---------- Keyword output parsing ----------

_NUMBERED = re.compile(r"^\s*(\d{1,2})[.)\]:\-]\s*(.+?)\s*$")
_BULLET = re.compile(r"^\s*[\-*•]\s*(.+?)\s*$")
_STEP2_MARKER = re.compile(r"step\s*2|final\s*(list|10|ten|terms?)", re.IGNORECASE)


def parse_keywords(raw_output: str) -> tuple[list[str], list[str]]:
    """
    Parse a keyword-extraction response. Targets the Step 1 / Step 2 format
    in prompts.txt: a longlist followed by a numbered list of exactly 10 terms.

    Strategy:
      1. If "Step 2" / "Final" marker is present, only look after it.
      2. Collect all numbered lines (in document order) with valid single-word
         or hyphenated terms; multi-word terms are skipped and flagged.
      3. If no numbered lines, fall back to bullet lines.
      4. Without a Step 2 marker, if >10 terms were found assume Step 1 was
         also numbered, take the last 10.
      5. Flag (don't fail) when final count != 10.

    Tolerates markdown emphasis, trailing punctuation, surrounding quotes,
    and gaps in numbering.

    Returns (terms, errors). `errors` is informational.
    """
    errors: list[str] = []
    if not isinstance(raw_output, str) or not raw_output.strip():
        return [], ["empty output"]

    lines = raw_output.splitlines()

    # 1. Find Step 2 marker; restrict to lines after it if found
    step2_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _STEP2_MARKER.search(line):
            step2_idx = i
            break
    target_lines = lines[step2_idx + 1:] if step2_idx is not None else lines

    # 2. Collect valid numbered terms in document order
    terms: list[str] = []
    for line in target_lines:
        m = _NUMBERED.match(line)
        if not m:
            continue
        term = _clean_term(m.group(2))
        if not term:
            continue
        if " " in term and "-" not in term:
            errors.append(f"multi-word term ignored: '{term}'")
            continue
        terms.append(term)

    # 3. Bullet fallback
    if not terms:
        for line in target_lines:
            m = _BULLET.match(line)
            if not m:
                continue
            term = _clean_term(m.group(1))
            if not term:
                continue
            if " " in term and "-" not in term:
                errors.append(f"multi-word bullet ignored: '{term}'")
                continue
            terms.append(term)
        if terms:
            errors.append("no numbered list; using bullets")

    if not terms:
        return [], errors + ["no numbered list or bullets found"]

    # 4. Without Step 2 marker, if too many, take the last 10 (Step 2 after Step 1)
    if step2_idx is None and len(terms) > 10:
        errors.append(f"truncated from {len(terms)} to last 10 (Step 1 likely numbered)")
        terms = terms[-10:]

    # 5. Count flag
    if len(terms) != 10:
        errors.append(f"expected 10 terms, got {len(terms)}")

    return terms, errors


_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")


def _clean_term(s: str) -> str:
    s = s.strip()
    # Strip markdown emphasis
    s = re.sub(r"^\*+|\*+$", "", s).strip()
    s = re.sub(r"^_+|_+$", "", s).strip()
    # Strip surrounding quotes
    s = s.strip("\"'`")
    # Strip trailing punctuation
    s = _TRAILING_PUNCT.sub("", s)
    # Some models add parentheticals: "adventure (broad concept)" -> "adventure"
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    return s


# ---------- Embedding client ----------

class EmbeddingClient:
    """
    Embeddings with disk cache so we never re-embed the same string.

    Primary backend: OpenAI text-embedding-3-large (3072 dim).
    Fallback: sentence-transformers/all-mpnet-base-v2 (768 dim, local).
    When no OpenAI key is provided the client auto-selects the local fallback
    (use_fallback defaults to None = auto) so scoring never hard-fails on a
    missing key.

    Cache is keyed by (model, text) -> list[float]. File format is JSONL,
    one entry per line, append-only and resilient to interruption.

    Usage:
        from src import config
        emb = EmbeddingClient(api_key=config.OPENAI_API_KEY)
        v = emb.embed("hello world")
        emb.save_cache()  # flush any pending writes
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        cache_path: Optional[Path] = None,
        use_fallback: Optional[bool] = None,
    ):
        self.api_key = api_key
        self.model = model
        # use_fallback tri-state:
        #   None  -> auto: use the local model only when no OpenAI key is available,
        #            so scoring degrades gracefully instead of crashing.
        #   True  -> force the local sentence-transformers backend.
        #   False -> force the OpenAI backend (raises later if the key is missing).
        if use_fallback is None:
            use_fallback = not bool(api_key)
            if use_fallback:
                print(
                    "[EmbeddingClient] OPENAI_API_KEY not set; falling back to local "
                    "sentence-transformers/all-mpnet-base-v2 (768 dim). Scores are still "
                    "comparable within a run, but not across OpenAI vs local backends.",
                    file=sys.stderr,
                )
        self.use_fallback = use_fallback
        self._client = None
        self._fallback = None

        if cache_path is None:
            try:
                from src import config
            except ImportError:
                import config
            cache_path = config.OUTPUTS_DIR / "embedding_cache.jsonl"
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        self._dirty = 0
        self._load_cache()

    # ---- public ----

    def embed(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            raise ValueError("cannot embed empty string")
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        vec = self._embed_remote(text) if not self.use_fallback else self._embed_local(text)
        self._cache[key] = vec
        self._append_to_cache_file(key, vec)
        return vec

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Batch convenience. Cache-aware; only sends uncached texts."""
        out: list[Optional[list[float]]] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            t_clean = (t or "").strip()
            if not t_clean:
                raise ValueError(f"empty text at index {i}")
            k = self._cache_key(t_clean)
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                to_fetch.append((i, t_clean))

        # Send uncached in batches
        for chunk_start in range(0, len(to_fetch), batch_size):
            chunk = to_fetch[chunk_start:chunk_start + batch_size]
            texts_chunk = [t for _, t in chunk]
            vecs = self._embed_remote_batch(texts_chunk) if not self.use_fallback \
                else [self._embed_local(t) for t in texts_chunk]
            for (idx, t), v in zip(chunk, vecs):
                k = self._cache_key(t)
                self._cache[k] = v
                self._append_to_cache_file(k, v)
                out[idx] = v
        return out  # type: ignore[return-value]

    def cache_stats(self) -> dict:
        return {
            "cached_items": len(self._cache),
            "cache_file": str(self.cache_path),
            "model": self.model,
            "backend": "fallback" if self.use_fallback else "openai",
        }

    # ---- internal: OpenAI ----

    def _ensure_openai(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _embed_remote(self, text: str) -> list[float]:
        c = self._ensure_openai()
        r = c.embeddings.create(model=self.model, input=text)
        return list(r.data[0].embedding)

    def _embed_remote_batch(self, texts: list[str]) -> list[list[float]]:
        c = self._ensure_openai()
        r = c.embeddings.create(model=self.model, input=texts)
        return [list(d.embedding) for d in r.data]

    # ---- internal: local fallback ----

    def _ensure_fallback(self):
        if self._fallback is None:
            from sentence_transformers import SentenceTransformer
            self._fallback = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        return self._fallback

    def _embed_local(self, text: str) -> list[float]:
        m = self._ensure_fallback()
        return [float(x) for x in m.encode(text)]

    # ---- cache ----

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.model}:{text}".encode("utf-8")).hexdigest()
        return h[:24]

    def _load_cache(self):
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self._cache[entry["k"]] = entry["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass

    def _append_to_cache_file(self, key: str, vec: list[float]):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": vec}) + "\n")
        except OSError:
            # Don't fail scoring just because cache write failed
            pass

    def save_cache(self):
        """No-op for compatibility — writes are append-on-each-embed."""
        return
