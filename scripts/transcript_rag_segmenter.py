"""Lightweight local RAG helpers for ASR transcript segmentation."""

from __future__ import annotations

import math
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass

try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover - optional local dependency
    jieba = None

try:
    import numpy as np  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:  # pragma: no cover - optional local dependency
    np = None
    TfidfVectorizer = None
    cosine_similarity = None


@dataclass(frozen=True)
class SentenceUnit:
    index: int
    text: str
    char_start: int
    char_end: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class WindowUnit:
    index: int
    start_sentence: int
    end_sentence: int
    text: str
    keywords: tuple[str, ...]
    centrality: float
    backend: str = "tfidf"


@dataclass(frozen=True)
class IndexUnit:
    unit_id: str
    level: str
    index: int
    start_sentence: int
    end_sentence: int
    text: str
    keywords: tuple[str, ...]
    tfidf_score: float = 0.0
    embedding_score: float = 0.0
    fused_score: float = 0.0


STOPWORDS = {
    "一个",
    "一些",
    "这个",
    "那个",
    "我们",
    "你们",
    "他们",
    "就是",
    "然后",
    "所以",
    "因为",
    "但是",
    "如果",
    "可以",
    "进行",
    "以及",
    "包括",
    "还是",
    "已经",
    "没有",
    "不是",
    "什么",
    "这里",
    "里面",
}


def split_sentences(text: str) -> list[SentenceUnit]:
    units: list[SentenceUnit] = []
    line_matches = [match for match in re.finditer(r"[^\n]+", text)]
    if len(line_matches) >= 8:
        for line_match in line_matches:
            line = line_match.group(0).strip()
            if not line:
                continue
            pieces = list(re.finditer(r".+?(?:[。！？!?]+[」』”’）)]*|$)", line))
            usable_pieces = [piece for piece in pieces if piece.group(0).strip()]
            if len(usable_pieces) <= 1:
                units.append(
                    SentenceUnit(
                        index=len(units),
                        text=re.sub(r"\s+", " ", line),
                        char_start=line_match.start(),
                        char_end=line_match.end(),
                        tokens=tuple(tokenize(line)),
                    )
                )
                continue
            for piece in usable_pieces:
                sentence = re.sub(r"\s+", " ", piece.group(0)).strip()
                units.append(
                    SentenceUnit(
                        index=len(units),
                        text=sentence,
                        char_start=line_match.start() + piece.start(),
                        char_end=line_match.start() + piece.end(),
                        tokens=tuple(tokenize(sentence)),
                    )
                )
        return units

    for match in re.finditer(r".+?(?:[。！？!?]+[」』”’）)]*|$)", text, flags=re.S):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip()
        if not sentence:
            continue
        units.append(
            SentenceUnit(
                index=len(units),
                text=sentence,
                char_start=match.start(),
                char_end=match.end(),
                tokens=tuple(tokenize(sentence)),
            )
        )
    return units


def tokenize(text: str) -> list[str]:
    if jieba is not None:
        raw = jieba.lcut(text)
    else:
        raw = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text)
    tokens = []
    for token in raw:
        token = token.strip().lower()
        if len(token) < 2 or token in STOPWORDS:
            continue
        if re.fullmatch(r"\W+", token):
            continue
        tokens.append(token)
    return tokens


def build_windows(
    sentences: list[SentenceUnit],
    *,
    window_sentences: int = 8,
    overlap_sentences: int = 2,
    top_keywords: int = 8,
    backend: str = "tfidf",
    embedding_model: str = "qwen3-embedding:0.6b",
    ollama_base_url: str = "http://127.0.0.1:11434",
    embedding_timeout: float = 120.0,
) -> list[WindowUnit]:
    if backend in {"ollama", "hybrid"}:
        try:
            return build_windows_ollama(
                sentences,
                window_sentences=window_sentences,
                overlap_sentences=overlap_sentences,
                top_keywords=top_keywords,
                embedding_model=embedding_model,
                ollama_base_url=ollama_base_url,
                embedding_timeout=embedding_timeout,
                blend_tfidf=backend == "hybrid",
            )
        except Exception as exc:
            if backend == "ollama":
                raise
            print(f"Ollama embedding 不可用，回退 TF-IDF：{exc}", flush=True)
    if TfidfVectorizer is not None and cosine_similarity is not None:
        return build_windows_sklearn(
            sentences,
            window_sentences=window_sentences,
            overlap_sentences=overlap_sentences,
            top_keywords=top_keywords,
        )
    return build_windows_pure_python(
        sentences,
        window_sentences=window_sentences,
        overlap_sentences=overlap_sentences,
        top_keywords=top_keywords,
    )


def build_windows_sklearn(
    sentences: list[SentenceUnit],
    *,
    window_sentences: int = 8,
    overlap_sentences: int = 2,
    top_keywords: int = 8,
) -> list[WindowUnit]:
    if not sentences:
        return []
    step = max(1, window_sentences - overlap_sentences)
    raw_windows: list[tuple[int, int, str]] = []
    tokenized_docs: list[str] = []
    for start in range(0, len(sentences), step):
        end = min(len(sentences), start + window_sentences)
        text = "".join(sentence.text for sentence in sentences[start:end])
        raw_windows.append((start, end, text))
        tokenized_docs.append(" ".join(token for sentence in sentences[start:end] for token in sentence.tokens))
        if end == len(sentences):
            break

    if not any(doc.strip() for doc in tokenized_docs):
        return build_windows_pure_python(
            sentences,
            window_sentences=window_sentences,
            overlap_sentences=overlap_sentences,
            top_keywords=top_keywords,
        )

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(tokenized_docs)
    centroid = np.asarray(matrix.mean(axis=0)) if np is not None else matrix.mean(axis=0)
    similarities = cosine_similarity(matrix, centroid).reshape(-1).tolist()
    feature_names = vectorizer.get_feature_names_out()
    result: list[WindowUnit] = []
    for idx, (start, end, text) in enumerate(raw_windows):
        row = matrix.getrow(idx)
        scored = sorted(
            ((feature_names[col], float(score)) for col, score in zip(row.indices, row.data)),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        result.append(
            WindowUnit(
                index=idx,
                start_sentence=start,
                end_sentence=end,
                text=text,
                keywords=tuple(token for token, _score in scored[:top_keywords]),
                centrality=float(similarities[idx]),
                backend="tfidf",
            )
        )
    return result


def build_windows_pure_python(
    sentences: list[SentenceUnit],
    *,
    window_sentences: int = 8,
    overlap_sentences: int = 2,
    top_keywords: int = 8,
) -> list[WindowUnit]:
    if not sentences:
        return []
    step = max(1, window_sentences - overlap_sentences)
    windows: list[tuple[int, int, str, Counter[str]]] = []
    doc_freq: Counter[str] = Counter()
    for start in range(0, len(sentences), step):
        end = min(len(sentences), start + window_sentences)
        if start >= end:
            break
        text = "".join(sentence.text for sentence in sentences[start:end])
        counts: Counter[str] = Counter()
        for sentence in sentences[start:end]:
            counts.update(sentence.tokens)
        for token in counts:
            doc_freq[token] += 1
        windows.append((start, end, text, counts))
        if end == len(sentences):
            break

    total = max(1, len(windows))
    vectors: list[dict[str, float]] = []
    for _start, _end, _text, counts in windows:
        vector = {}
        for token, count in counts.items():
            idf = math.log((1 + total) / (1 + doc_freq[token])) + 1.0
            vector[token] = float(count) * idf
        vectors.append(vector)

    centroid: Counter[str] = Counter()
    for vector in vectors:
        centroid.update(vector)
    centroid_vector = {token: value / total for token, value in centroid.items()}

    result: list[WindowUnit] = []
    for idx, ((start, end, text, counts), vector) in enumerate(zip(windows, vectors)):
        scored_keywords = sorted(
            vector.items(),
            key=lambda item: (item[1], counts[item[0]], item[0]),
            reverse=True,
        )
        result.append(
            WindowUnit(
                index=idx,
                start_sentence=start,
                end_sentence=end,
                text=text,
                keywords=tuple(token for token, _score in scored_keywords[:top_keywords]),
                centrality=cosine(vector, centroid_vector),
                backend="tfidf-pure",
            )
        )
    return result


def build_windows_ollama(
    sentences: list[SentenceUnit],
    *,
    window_sentences: int = 8,
    overlap_sentences: int = 2,
    top_keywords: int = 8,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout: float,
    blend_tfidf: bool,
) -> list[WindowUnit]:
    if np is None:
        raise RuntimeError("numpy is required for Ollama embedding backend")
    if not sentences:
        return []
    step = max(1, window_sentences - overlap_sentences)
    raw_windows: list[tuple[int, int, str, Counter[str]]] = []
    for start in range(0, len(sentences), step):
        end = min(len(sentences), start + window_sentences)
        text = "".join(sentence.text for sentence in sentences[start:end])
        counts: Counter[str] = Counter()
        for sentence in sentences[start:end]:
            counts.update(sentence.tokens)
        raw_windows.append((start, end, text, counts))
        if end == len(sentences):
            break
    embeddings = [
        ollama_embed(
            text,
            model=embedding_model,
            base_url=ollama_base_url,
            timeout=embedding_timeout,
        )
        for _start, _end, text, _counts in raw_windows
    ]
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    centroid = matrix.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    similarities = matrix @ centroid

    tfidf_scores: list[float] = []
    tfidf_keyword_map: list[tuple[str, ...]] = []
    if blend_tfidf:
        tfidf_windows = build_windows_sklearn(
            sentences,
            window_sentences=window_sentences,
            overlap_sentences=overlap_sentences,
            top_keywords=top_keywords,
        )
        tfidf_scores = [window.centrality for window in tfidf_windows]
        tfidf_keyword_map = [window.keywords for window in tfidf_windows]

    result: list[WindowUnit] = []
    for idx, (start, end, text, counts) in enumerate(raw_windows):
        keywords = tfidf_keyword_map[idx] if idx < len(tfidf_keyword_map) else top_count_keywords(counts, top_keywords)
        centrality = float(similarities[idx])
        if blend_tfidf and idx < len(tfidf_scores):
            centrality = 0.7 * centrality + 0.3 * float(tfidf_scores[idx])
        result.append(
            WindowUnit(
                index=idx,
                start_sentence=start,
                end_sentence=end,
                text=text,
                keywords=keywords,
                centrality=centrality,
                backend="hybrid" if blend_tfidf else "ollama",
            )
        )
    return result


def ollama_embed(text: str, *, model: str, base_url: str, timeout: float) -> list[float]:
    payload = json.dumps({"model": model, "prompt": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(f"Ollama returned no embedding for model {model}")
    return [float(value) for value in embedding]


def top_count_keywords(counts: Counter[str], limit: int) -> tuple[str, ...]:
    return tuple(token for token, _count in counts.most_common(limit))


def sentence_index_units(sentences: list[SentenceUnit]) -> list[IndexUnit]:
    return [
        IndexUnit(
            unit_id=f"S{sentence.index:04d}",
            level="sentence",
            index=sentence.index,
            start_sentence=sentence.index,
            end_sentence=sentence.index + 1,
            text=sentence.text,
            keywords=tuple(sentence.tokens[:8]),
        )
        for sentence in sentences
    ]


def paragraph_index_units(
    sentences: list[SentenceUnit],
    *,
    paragraph_sentences: int = 4,
) -> list[IndexUnit]:
    units: list[IndexUnit] = []
    for start in range(0, len(sentences), paragraph_sentences):
        end = min(len(sentences), start + paragraph_sentences)
        chunk = sentences[start:end]
        counts: Counter[str] = Counter(token for sentence in chunk for token in sentence.tokens)
        units.append(
            IndexUnit(
                unit_id=f"P{len(units):04d}",
                level="paragraph",
                index=len(units),
                start_sentence=start,
                end_sentence=end,
                text="".join(sentence.text for sentence in chunk),
                keywords=top_count_keywords(counts, 10),
            )
        )
    return units


def topic_index_units(
    sentences: list[SentenceUnit],
    *,
    topic_sentences: int = 12,
    overlap_sentences: int = 2,
) -> list[IndexUnit]:
    units: list[IndexUnit] = []
    step = max(1, topic_sentences - overlap_sentences)
    for start in range(0, len(sentences), step):
        end = min(len(sentences), start + topic_sentences)
        chunk = sentences[start:end]
        counts: Counter[str] = Counter(token for sentence in chunk for token in sentence.tokens)
        units.append(
            IndexUnit(
                unit_id=f"T{len(units):04d}",
                level="topic",
                index=len(units),
                start_sentence=start,
                end_sentence=end,
                text="".join(sentence.text for sentence in chunk),
                keywords=top_count_keywords(counts, 12),
            )
        )
        if end == len(sentences):
            break
    return units


def build_hierarchical_units(sentences: list[SentenceUnit]) -> list[IndexUnit]:
    return [
        *sentence_index_units(sentences),
        *paragraph_index_units(sentences),
        *topic_index_units(sentences),
    ]


def tokenized_unit_docs(units: list[IndexUnit]) -> list[str]:
    docs: list[str] = []
    for unit in units:
        tokens = tokenize(unit.text)
        docs.append(" ".join(tokens))
    return docs


def tfidf_centrality(units: list[IndexUnit]) -> dict[str, float]:
    if not units:
        return {}
    docs = tokenized_unit_docs(units)
    if not any(doc.strip() for doc in docs):
        return {}
    if TfidfVectorizer is not None and cosine_similarity is not None and np is not None:
        matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(docs)
        centroid = np.asarray(matrix.mean(axis=0))
        scores = cosine_similarity(matrix, centroid).reshape(-1).tolist()
        return {unit.unit_id: float(score) for unit, score in zip(units, scores)}
    counts = [Counter(doc.split()) for doc in docs]
    doc_freq: Counter[str] = Counter(token for item in counts for token in item)
    vectors: list[dict[str, float]] = []
    total = max(1, len(units))
    for count in counts:
        vectors.append(
            {
                token: value * (math.log((1 + total) / (1 + doc_freq[token])) + 1.0)
                for token, value in count.items()
            }
        )
    centroid: Counter[str] = Counter()
    for vector in vectors:
        centroid.update(vector)
    centroid_vector = {token: value / total for token, value in centroid.items()}
    return {unit.unit_id: cosine(vector, centroid_vector) for unit, vector in zip(units, vectors)}


def embedding_centrality(
    units: list[IndexUnit],
    *,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout: float,
) -> dict[str, float]:
    if not units:
        return {}
    if np is None:
        raise RuntimeError("numpy is required for embedding centrality")
    embeddings = [
        ollama_embed(
            unit.text,
            model=embedding_model,
            base_url=ollama_base_url,
            timeout=embedding_timeout,
        )
        for unit in units
    ]
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    centroid = matrix.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    scores = matrix @ centroid
    return {unit.unit_id: float(score) for unit, score in zip(units, scores)}


def rrf_rank(scores: dict[str, float], *, k: int = 60) -> dict[str, float]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {unit_id: 1.0 / (k + rank) for rank, (unit_id, _score) in enumerate(ranked, start=1)}


def score_hierarchical_units(
    units: list[IndexUnit],
    *,
    backend: str,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout: float,
) -> list[IndexUnit]:
    tfidf_scores = tfidf_centrality(units)
    embedding_scores: dict[str, float] = {}
    if backend in {"ollama", "hybrid"}:
        embedding_scores = embedding_centrality(
            units,
            embedding_model=embedding_model,
            ollama_base_url=ollama_base_url,
            embedding_timeout=embedding_timeout,
        )
    if backend == "tfidf":
        fused = rrf_rank(tfidf_scores)
    elif backend == "ollama":
        fused = rrf_rank(embedding_scores)
    else:
        tfidf_rrf = rrf_rank(tfidf_scores)
        embedding_rrf = rrf_rank(embedding_scores)
        fused = {
            unit.unit_id: tfidf_rrf.get(unit.unit_id, 0.0) + embedding_rrf.get(unit.unit_id, 0.0)
            for unit in units
        }
    return [
        IndexUnit(
            unit_id=unit.unit_id,
            level=unit.level,
            index=unit.index,
            start_sentence=unit.start_sentence,
            end_sentence=unit.end_sentence,
            text=unit.text,
            keywords=unit.keywords,
            tfidf_score=tfidf_scores.get(unit.unit_id, 0.0),
            embedding_score=embedding_scores.get(unit.unit_id, 0.0),
            fused_score=fused.get(unit.unit_id, 0.0),
        )
        for unit in units
    ]


def select_hierarchical_evidence(units: list[IndexUnit], *, max_units: int) -> list[IndexUnit]:
    level_budgets = {
        "sentence": max(4, max_units // 4),
        "paragraph": max(4, max_units // 3),
        "topic": max(4, max_units - (max_units // 4) - (max_units // 3)),
    }
    selected: list[IndexUnit] = []
    seen: set[str] = set()
    for level, budget in level_budgets.items():
        ranked = sorted(
            (unit for unit in units if unit.level == level),
            key=lambda item: (item.fused_score, item.embedding_score, item.tfidf_score),
            reverse=True,
        )
        for unit in ranked[:budget]:
            selected.append(unit)
            seen.add(unit.unit_id)
    if len(selected) < max_units:
        ranked_all = sorted(
            units,
            key=lambda item: (item.fused_score, item.embedding_score, item.tfidf_score),
            reverse=True,
        )
        for unit in ranked_all:
            if unit.unit_id in seen:
                continue
            selected.append(unit)
            seen.add(unit.unit_id)
            if len(selected) >= max_units:
                break
    return sorted(selected[:max_units], key=lambda item: (item.start_sentence, item.end_sentence, item.level))


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_rag_context(
    text: str,
    *,
    max_windows: int = 24,
    backend: str = "ollama",
    embedding_model: str = "qwen3-embedding:0.6b",
    ollama_base_url: str = "http://127.0.0.1:11434",
    embedding_timeout: float = 120.0,
) -> tuple[str, dict[str, int]]:
    sentences = split_sentences(text)
    units = score_hierarchical_units(
        build_hierarchical_units(sentences),
        backend=backend,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        embedding_timeout=embedding_timeout,
    )
    evidence = select_hierarchical_evidence(units, max_units=max_windows)

    lines = [
        "下面是面向长视频字幕的四层流水线候选证据：",
        "1. 中文断句；2. Paragraph / Topic 语义窗口；3. TF-IDF + Embedding 双索引；4. Qwen 结构分析。",
        "索引层级包含 Sentence、Paragraph、Topic。分数为 RRF 融合后的候选强度；证据只用于判断主题边界，最终输出必须覆盖完整原文。",
        "",
    ]
    for unit in evidence:
        preview = unit.text
        if len(preview) > 520:
            preview = preview[:520].rstrip() + "..."
        lines.extend(
            [
                (
                    f"[{unit.unit_id}] level={unit.level} sentences={unit.start_sentence}-{unit.end_sentence - 1} "
                    f"tfidf={unit.tfidf_score:.3f} embedding={unit.embedding_score:.3f} fused={unit.fused_score:.4f}"
                ),
                f"keywords: {', '.join(unit.keywords) or '无'}",
                preview,
                "",
            ]
        )
    level_counts = Counter(unit.level for unit in evidence)
    return "\n".join(lines).strip(), {
        "sentences": len(sentences),
        "windows": len(evidence),
        "embedding_backend": backend,
        "sentence_units": level_counts.get("sentence", 0),
        "paragraph_units": level_counts.get("paragraph", 0),
        "topic_units": level_counts.get("topic", 0),
    }
