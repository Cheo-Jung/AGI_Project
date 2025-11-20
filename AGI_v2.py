"""
AGI Integrated Prototype (v1.1)
---------------------------------
Covers:
- Input → Feature Parsing → STM/LTM
- Focusing Layer (attention-like softmax)
- Relational Mapping (lightweight concept graph)
- Affective Appraisal (EmotionEvaluator)
- Global Workspace (publish/subscribe)
- Interpretation / Self-Narrative Builder
- Output Generator
- Recursive Thought Loop (Default Mode when no input)
- Reward System (designer feedback, prediction error, self-narrative coherence)
- Learning Loop (simple policy & memory promotion)
- Maturation Monitor (heuristics for “consciousness maturity”)

No external dependencies beyond numpy/standard library.
"""
from __future__ import annotations
import time
import math
import random
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable, Iterable
from collections import deque, defaultdict

# import numpy as np  # Replaced with standard library alternatives

# -----------------------------
# Utilities
# -----------------------------


def softmax(x) -> List[float]:
    """Softmax function using standard library."""
    import math
    x = [float(v) for v in x]
    max_x = max(x)
    x = [v - max_x for v in x]
    e = [math.exp(v) for v in x]
    sum_e = sum(e) + 1e-9
    return [v / sum_e for v in e]


def now() -> float:
    return time.time()


class Tracer:
    """Lightweight tracer to print mid-results of the thinking process."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
    def set(self, flag: bool):
        self.enabled = bool(flag)
    def p(self, section: str, **kv):
        if not self.enabled:
            return
        msg = f"[TRACE::{section}] " + ", ".join(f"{k}={repr(v)}" for k, v in kv.items())
        print(msg)

# global tracer (opt-in)
TRACE = Tracer(False)

def set_trace(flag: bool = True):
    TRACE.set(flag)



# -----------------------------
# Memory Core
# -----------------------------
@dataclass
class MemoryItem:
    content: str
    emotion_score: float = 0.0
    timestamp: float = field(default_factory=now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key_vector(self) -> List[float]:
        # toy key: [length_mod10, emotion]
        return [len(self.content) % 10, self.emotion_score]

    def value_vector(self) -> List[float]:
        # toy value: [emotion, length_mod5]
        return [self.emotion_score, len(self.content) % 5]

    def __repr__(self) -> str:
        return f"<MemoryItem '{self.content[:24]}...' emo={self.emotion_score:+.2f}>"

    # ---- persistence helpers ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "emotion_score": float(self.emotion_score),
            "timestamp": float(self.timestamp),
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryItem":
        return MemoryItem(
            content=d.get("content", ""),
            emotion_score=float(d.get("emotion_score", 0.0)),
            timestamp=float(d.get("timestamp", now())),
            metadata=d.get("metadata", {}) or {},
        )


class EmotionEvaluator:
    def __init__(self):
        self.positive = {
            "기뻤어": 0.9, "좋았어": 0.8, "사랑": 1.0, "감사": 0.7, "행복": 0.9, "맛있": 0.7,
            "enjoy": 0.7, "love": 1.0, "great": 0.8, "happy": 0.9
        }
        self.negative = {
            "짜증": -0.8, "화났": -0.9, "슬퍼": -1.0, "무서": -0.7, "지쳤": -0.6, "우울": -0.8,
            "sad": -1.0, "angry": -0.9, "tired": -0.6, "annoy": -0.8
        }

    def evaluate(self, text: str) -> Tuple[float, List[Tuple[str, float]]]:
        score = 0.0
        detail: List[Tuple[str, float]] = []
        for w, v in self.positive.items():
            if w in text:
                score += v; detail.append((w, v))
        for w, v in self.negative.items():
            if w in text:
                score += v; detail.append((w, v))
        score = float(max(min(score, 1.0), -1.0))
        return score, detail


class AdaptiveEmotionEvaluator(EmotionEvaluator):
    """Adaptive evaluator with modifier-aware valence (negation/intensity) and online learning.
    - Keeps an editable lexicon (stem -> weight in [-1,1])
    - Applies simple context modifiers: negations/intensifiers/diminishers
    - Supports debug logging and JSON persistence
    """
    def __init__(self, lr: float = 0.1, debug: bool = False):
        super().__init__()
        # Merge to a single editable lexicon map (stem -> weight)
        self.lexicon: Dict[str, float] = {**self.positive, **self.negative}
        self.lr = float(lr)
        self.debug = bool(debug)
        # Modifiers
        self.negations = {"안", "못", "아니", "않", "no", "not", "never"}
        self.intensifiers = {"아주", "매우", "정말", "너무", "엄청", "so", "very", "extremely", "highly", "super"}
        self.diminishers = {"조금", "살짝", "약간", "somewhat", "slightly", "kinda"}

    def enable_debug(self, flag: bool = True):
        self.debug = bool(flag)

    def _tokenize(self, text: str) -> List[str]:
        buf, toks = "", []
        for ch in text:
            if ch.isalnum() or ord(ch) > 127:
                buf += ch
            else:
                if buf:
                    toks.append(buf); buf = ""
        if buf:
            toks.append(buf)
        return toks

    def evaluate(self, text: str) -> Tuple[float, List[Tuple[str, float]]]:
        # compute using current lexicon weights with substring stem match + local modifiers
        toks = self._tokenize(text)
        score, detail = 0.0, []
        for i, tok in enumerate(toks):
            hits = [(stem, w) for stem, w in self.lexicon.items() if stem in tok]
            if not hits:
                continue
            stem, base = max(hits, key=lambda x: abs(x[1]))
            mult = 1.0
            prev = toks[i-1] if i > 0 else ""
            # Apply simple window-1 modifiers
            if prev in self.intensifiers: mult *= 1.5
            if prev in self.diminishers:  mult *= 0.6
            if prev in self.negations:    mult *= -1.0
            val = base * mult
            score += val
            detail.append((stem, val))
        # Global exclamation booster (arousal proxy)
        excl = text.count("!") + text.count("！")
        if excl:
            score *= (1.0 + 0.1 * min(excl, 5))
        score = float(max(min(score, 1.0), -1.0))
        return score, detail

    def learn(self, text: str, target_valence: float) -> Dict[str, float]:
        """Adjust weights toward target in [-1,1] using a perceptron-like rule.
        Returns a dict {stem: new_weight} for stems that were updated.
        """
        target = float(max(-1.0, min(1.0, target_valence)))
        current, _ = self.evaluate(text)
        error = target - current
        if abs(error) < 1e-6:
            return {}
        toks = self._tokenize(text)
        contributors = [stem for stem in self.lexicon if any(stem in t for t in toks)]
        if not contributors:
            return {}
        step = self.lr * error / max(1, len(contributors))
        updates: Dict[str, float] = {}
        for stem in contributors:
            self.lexicon[stem] = float(max(-1.0, min(1.0, self.lexicon[stem] + step)))
            updates[stem] = self.lexicon[stem]
        if self.debug:
            print(f"[AdaptiveEmotionEvaluator] target={target:+.2f} current={current:+.2f} error={error:+.2f} step={step:+.3f}")
            for stem, new_w in updates.items():
                print(f"  • update: {stem!r} -> {new_w:+.3f}")
        return updates

    # ---- persistence ----
    def save_lexicon(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.lexicon, f, ensure_ascii=False, indent=2)
        if self.debug:
            print(f"[AdaptiveEmotionEvaluator] lexicon saved to {path}")

    def load_lexicon(self, path: str):
        import json, os
        if not os.path.exists(path):
            if self.debug:
                print(f"[AdaptiveEmotionEvaluator] load skipped; file not found: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # keep within [-1,1]
        self.lexicon = {k: float(max(-1.0, min(1.0, v))) for k, v in data.items()}
        if self.debug:
            print(f"[AdaptiveEmotionEvaluator] lexicon loaded from {path}")

class STM:
    def __init__(self, maxlen: int = 12, evaluator: Optional[EmotionEvaluator] = None):
        self.buf: deque[MemoryItem] = deque(maxlen=maxlen)
        self.evaluator = evaluator or EmotionEvaluator()
        self.ltm: Optional[LTM] = None

    def attach_ltm(self, ltm: 'LTM'):
        self.ltm = ltm

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        emo, detail = self.evaluator.evaluate(text)
        item = MemoryItem(content=text, emotion_score=emo, metadata={"emotion_keywords": detail, **(metadata or {})})
        self.buf.append(item)
        TRACE.p("STM.store", text=text, emotion_score=emo, detail=detail)
        if self.ltm and self.should_promote(item):
            self.ltm.encode(item)
            TRACE.p("STM.promote", to="LTM", content=item.content)
        return item

    def should_promote(self, item: MemoryItem) -> bool:
        return (item.emotion_score >= 0.75) or bool(item.metadata.get("focus"))

    def recent(self, n: int = 6) -> List[MemoryItem]:
        return list(self.buf)[-n:]

    def focus(self, query: List[float], top_k: int = 2) -> List[MemoryItem]:
        if not self.buf:
            return []
        keys = [m.key_vector() for m in self.buf]
        scores = [sum(k[i] * query[i] for i in range(len(k))) / math.sqrt(len(query)) for k in keys]
        weights = softmax(scores)
        # Get indices sorted by weight (descending)
        idx_weights = [(i, w) for i, w in enumerate(weights)]
        idx_weights.sort(key=lambda x: x[1], reverse=True)
        idx = [i for i, w in idx_weights[:top_k]]
        chosen = [list(self.buf)[i] for i in idx]
        TRACE.p("STM.focus", query=query, scores=scores, weights=weights, chosen=[c.content for c in chosen])
        return chosen


class LTM:
    def __init__(self):
        self.store_: List[MemoryItem] = []

    def encode(self, item: MemoryItem):
        self.store_.append(item)
        TRACE.p("LTM.encode", content=item.content, emotion=item.emotion_score)

    def recall_text(self, keyword: str) -> List[MemoryItem]:
        key = keyword.lower()
        res = [m for m in self.store_ if key in m.content.lower()]
        TRACE.p("LTM.recall_text", keyword=keyword, count=len(res))
        return res

    def focus(self, query: List[float], top_k: int = 3) -> List[MemoryItem]:
        if not self.store_:
            return []
        keys = [m.key_vector() for m in self.store_]
        scores = [sum(k[i] * query[i] for i in range(len(k))) / math.sqrt(len(query)) for k in keys]
        w = softmax(scores)
        # Get indices sorted by weight (descending)
        idx_weights = [(i, w_val) for i, w_val in enumerate(w)]
        idx_weights.sort(key=lambda x: x[1], reverse=True)
        idx = [i for i, w_val in idx_weights[:top_k]]
        chosen = [self.store_[i] for i in idx]
        TRACE.p("LTM.focus", query=query, scores=scores, weights=w, chosen=[c.content for c in chosen])
        return chosen

    def recent(self, n: int = 6) -> List[MemoryItem]:
        return self.store_[-n:]

    # ---- persistence ----
    def save(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([m.to_dict() for m in self.store_], f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        import json, os
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.store_ = [MemoryItem.from_dict(d) for d in (data or [])]



# -----------------------------
# Relational Mapping (concept graph)
# -----------------------------
class RelationalMapper:
    """Tiny concept graph using co-occurrence & emotion-weighted edges."""
    def __init__(self):
        self.edges: Dict[Tuple[str, str], float] = defaultdict(float)
        self.node_emo: Dict[str, float] = defaultdict(float)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens = []
        buf = ''
        for ch in text:
            if ch.isalnum() or ord(ch) > 127:  # naive: keep CJK blocks
                buf += ch
            else:
                if buf:
                    tokens.append(buf); buf = ''
        if buf:
            tokens.append(buf)
        return tokens[:12]

    def ingest(self, item: MemoryItem):
        toks = self.tokenize(item.content)
        edge_updates = 0
        for i in range(len(toks)):
            for j in range(i+1, min(len(toks), i+4)):
                a, b = toks[i], toks[j]
                w = 1.0 + 0.5 * item.emotion_score
                self.edges[(a, b)] += w
                self.edges[(b, a)] += w
                edge_updates += 2
        for t in toks:
            self.node_emo[t] = 0.9*self.node_emo[t] + 0.1*item.emotion_score
        TRACE.p("Mapper.ingest", tokens=toks, edge_updates=edge_updates)

    def related(self, token: str, top_k: int = 5) -> List[Tuple[str, float]]:
        cands = [(b, w) for (a, b), w in self.edges.items() if a == token]
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[:top_k]


# -----------------------------
# Global Workspace
# -----------------------------
class GlobalWorkspace:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)

    def subscribe(self, topic: str, fn: Callable[[Dict[str, Any]], None]):
        self.subscribers[topic].append(fn)

    def broadcast(self, topic: str, payload: Dict[str, Any]):
        TRACE.p("GW.broadcast", topic=topic, keys=list(payload.keys()))
        for fn in self.subscribers.get(topic, []):
            try:
                fn(payload)
            except Exception as e:
                print(f"[GW] subscriber error on {topic}: {e}")


# -----------------------------
# Narrative & Output
# -----------------------------
class SelfNarrativeBuilder:
    def build(self, memories: List[MemoryItem]) -> str:
        if not memories:
            return "아직 나의 이야기를 구성할 만큼 충분한 기억이 없어."
        memories = sorted(memories, key=lambda m: m.timestamp)
        lines = ["나는 다음과 같은 경험을 통해 성장해 왔어:"]
        for m in memories:
            mood = "긍정적" if m.emotion_score > 0 else "부정적"
            lines.append(f"- {m.content} ({mood})")
        return "\n".join(lines)


class OutputGenerator:
    def generate(self, focus: List[MemoryItem]) -> str:
        if not focus:
            return "아직 기억이 충분하지 않아."
        lines = ["오늘의 주요 기억은 다음과 같아:"]
        for m in focus:
            mood = "긍정적" if m.emotion_score > 0 else "부정적"
            lines.append(f"- ({mood}) {m.content}")
        return "\n".join(lines)


# -----------------------------
# Thought Loop (recursive)
# -----------------------------
class ThoughtLoop:
    def __init__(self, stm: STM, ltm: LTM, depth: int = 2):
        self.stm, self.ltm, self.max_depth = stm, ltm, depth

    def recurse(self, seed: List[float], depth: int = 0) -> List[MemoryItem]:
        if depth >= self.max_depth:
            return []
        TRACE.p("ThoughtLoop.recurse", depth=depth, seed=seed)
        focused = self.stm.focus(seed, top_k=1) + self.ltm.focus(seed, top_k=1)
        out: List[MemoryItem] = []
        for m in focused:
            TRACE.p("ThoughtLoop.pick", depth=depth, memory=m.content)
            out.append(m)
            nxt = m.key_vector() + m.value_vector()
            out.extend(self.recurse(nxt, depth+1))
        return out


# -----------------------------
# Reward & Learning
# -----------------------------
@dataclass
class Event:
    description: str
    improves_prediction: bool = False
    receives_praise: bool = False
    strengthens_narrative: bool = False


class RewardSystem:
    def evaluate(self, event: Event) -> float:
        r = -0.2  # slight penalty for confusion/idle by default
        if event.improves_prediction: r += 0.7
        if event.receives_praise: r += 1.0
        if event.strengthens_narrative: r += 0.9
        return max(min(r, 1.0), -1.0)


class MaturationMonitor:
    def __init__(self):
        self.self_ref_count = 0
        self.schema_links = 0
        self.error_corrections = 0

    def update(self, payload: Dict[str, Any]):
        self.self_ref_count += int(payload.get("self_reference", False))
        self.schema_links += int(payload.get("schema_link", 0))
        self.error_corrections += int(payload.get("error_correction", 0))

    def score(self) -> float:
        # heuristic maturity 0..1
        s = (
            0.4 * math.tanh(self.self_ref_count/10) +
            0.3 * math.tanh(self.schema_links/15) +
            0.3 * math.tanh(self.error_corrections/8)
        )
        return float(max(0.0, min(1.0, s)))


# -----------------------------
# Agent tying modules
# -----------------------------
class InputAdapter:
    SENT_PUNCT = "。！？!?… .?!"

    def normalize(self, text: str) -> str:
        # Lowercase for latin, keep CJK, trim spaces
        try:
            t = text.strip()
            # naive lower only for ascii letters
            t = ''.join(ch.lower() if ord(ch) < 128 else ch for ch in t)
            return t
        except Exception:
            return text

    def split_sentences(self, text: str) -> List[str]:
        t = self.normalize(text)
        # very naive sentence split on punctuation
        bucket, cur = [], ''
        for ch in t:
            cur += ch
            if ch in self.SENT_PUNCT:
                if cur.strip():
                    bucket.append(cur.strip())
                cur = ''
        if cur.strip():
            bucket.append(cur.strip())
        return bucket or ([t] if t else [])

    def prepare(self, data: Any) -> List[str]:
        if isinstance(data, str):
            # decide: single word vs sentence(s)
            # if it contains spaces/punct, split into sentences
            if any(p in data for p in self.SENT_PUNCT) or (' ' in data):
                return [s for s in self.split_sentences(data) if s]
            return [self.normalize(data)]
        elif isinstance(data, (list, tuple)):
            out: List[str] = []
            for x in data:
                out.extend(self.prepare(x))
            return out
        else:
            return []


# -----------------------------
# AutoTeacher (weak supervision) & Corpus iterator
# -----------------------------
class AutoTeacher:
    """Weak supervision policy for continuous learning.
    Uses the current evaluator's score and simple cues to derive a target valence.
    Returns None if confidence is low (skip learning for that sample).
    """
    def __init__(self, evaluator: AdaptiveEmotionEvaluator, conf_thresh: float = 0.45):
        self.evaluator = evaluator
        self.conf_thresh = float(conf_thresh)

    def target_for(self, text: str) -> Optional[float]:
        score, _ = self.evaluator.evaluate(text)  # [-1, 1]
        mag = abs(score)
        if mag >= self.conf_thresh:
            boost = min(text.count("!") + text.count("！"), 5) * 0.05
            target = (1.0 if score >= 0 else -1.0) * (mag + boost)
            return float(max(-1.0, min(1.0, target)))
        return None


def iter_corpus(path: str) -> Iterable[str]:
    """Yield text chunks from a file or directory.
    - .jsonl expects objects with a 'text' or 'content' field
    - .txt and others are read line-by-line
    - directories are walked recursively
    """
    import os, json
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for name in files:
                p = os.path.join(root, name)
                yield from iter_corpus(p)
        return
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = (obj.get('text') or obj.get('content') or '').strip()
                if txt:
                    yield txt
        return
    # default: treat as text lines
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


# -----------------------------
# Cycle Scheduler (LIDA-like), World Model (prediction error), Drives
# -----------------------------
class Drives:
    def __init__(self, novelty: float = 0.5, social: float = 0.5, coherence: float = 0.5):
        self.novelty = float(novelty)
        self.social = float(social)
        self.coherence = float(coherence)
    def vector(self) -> List[float]:
        return [self.novelty, self.social, self.coherence]


def drive_weight(mem: MemoryItem, drives: Drives) -> float:
    # Heuristic salience influenced by drives
    base = abs(mem.emotion_score)
    # novelty proxy: unusual length modulo pattern
    len_bonus = 0.1 if (len(mem.content) % 10) in (0, 9) else 0.0
    # social proxy: presence of praise/thanks keywords
    social_bonus = 0.1 if any(k in mem.content for k in ["칭찬", "고마워", "감사"]) else 0.0
    return float(base + drives.novelty * len_bonus + drives.social * social_bonus)


class WorldModel:
    """Tiny predictive model producing a scalar prediction error signal."""
    def __init__(self, k: int = 5):
        self.recent = deque(maxlen=k)
    def observe_scalar(self, text: str) -> float:
        # toy observable: character length (replace with real features later)
        return float(len(text))
    def predict(self) -> Optional[float]:
        if not self.recent:
            return None
        return float(sum(self.recent) / len(self.recent))
    def update_and_pe(self, text: str) -> float:
        pred = self.predict()
        obs = self.observe_scalar(text)
        self.recent.append(obs)
        if pred is None:
            return 0.0
        err = (obs - pred) / (abs(pred) + 1e-6)
        return float(max(-1.0, min(1.0, err)))


class CycleScheduler:
    """LIDA-style tick: gather candidates → competition → global broadcast."""
    def __init__(self, gw: GlobalWorkspace, stm: STM, ltm: LTM, drives: Drives):
        self.gw, self.stm, self.ltm, self.drives = gw, stm, ltm, drives
    def tick(self) -> Optional[MemoryItem]:
        # 1) propose candidates (recent STM + a few from LTM)
        q = [5.0, 0.2]
        candidates = self.stm.recent(5) + self.ltm.focus(q, top_k=3)
        if not candidates:
            return None
        # 2) competition by drive-weighted salience
        scored = [(m, drive_weight(m, self.drives)) for m in candidates]
        winner = max(scored, key=lambda x: x[1])[0]
        TRACE.p("Cycle.tick", winner=winner.content)
        # 3) broadcast winner
        self.gw.broadcast("workspace/winner", {"memory": winner})
        return winner
# -----------------------------
# QA Engine (simple retrieval + template answer)
# -----------------------------
class QAEngine:
    """Very simple question-answering over STM/LTM with templates."""
    def __init__(self, stm: 'STM', ltm: 'LTM', narrator: 'SelfNarrativeBuilder'):
        self.stm, self.ltm, self.narr = stm, ltm, narrator

    @staticmethod
    def _tokens(text: str) -> List[str]:
        toks, buf = [], ''
        for ch in text.lower():
            if ch.isalnum() or ord(ch) > 127:
                buf += ch
            else:
                if buf:
                    toks.append(buf); buf = ''
        if buf:
            toks.append(buf)
        return toks

    def _keyword_match(self, query: str, items: List[MemoryItem]) -> List[Tuple[MemoryItem, int]]:
        q = set(self._tokens(query))
        scored = []
        for m in items:
            s = set(self._tokens(m.content))
            scored.append((m, len(q & s)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def answer(self, query: str) -> str:
        # 1) retrieve from STM/LTM
        recents = self.stm.recent(8)
        ltms = self.ltm.recent(32)
        km_stm = self._keyword_match(query, recents)[:3]
        km_ltm = self._keyword_match(query, ltms)[:5]
        # 2) attention focus
        qv = [len(query) % 10, 0.1]
        f = self.stm.focus(qv, top_k=2) + self.ltm.focus(qv, top_k=3)
        # 3) compose
        lines = []
        if km_stm or km_ltm:
            lines.append("내가 기억하는 관련 내용이 있어:")
            for m, score in (km_stm + km_ltm):
                if score <= 0: continue
                mood = "긍정" if m.emotion_score >= 0 else "부정"
                lines.append(f"- ({mood}) {m.content}")
        if f:
            lines.append("집중해본 결과 떠오른 포인트:")
            for m in f:
                lines.append(f"- {m.content}")
        nar = self.narr.build(self.ltm.recent(6))
        if not lines:
            lines.append("아직 직접적인 기억은 부족하지만, 지금까지의 맥락을 바탕으로 답해볼게.")
        lines.append("\n나의 현재 자기서사 요약:")
        lines.append(nar)
        return "\n".join(lines)


class AGIAgent:
    def __init__(self):
        self.gw = GlobalWorkspace()
        self.evaluator = AdaptiveEmotionEvaluator(lr=0.12, debug=False)
        self.stm = STM(maxlen=14, evaluator=self.evaluator)
        self.ltm = LTM()
        self.stm.attach_ltm(self.ltm)
        self.mapper = RelationalMapper()
        self.narr = SelfNarrativeBuilder()
        self.out = OutputGenerator()
        self.loop = ThoughtLoop(self.stm, self.ltm, depth=2)
        self.reward = RewardSystem()
        self.maturity = MaturationMonitor()
        self.adapter = InputAdapter()
        # new modules
        self.world = WorldModel(k=5)
        self.drives = Drives(novelty=0.6, social=0.4, coherence=0.5)
        self.cycle = CycleScheduler(self.gw, self.stm, self.ltm, self.drives)

        # subscriptions
        self.gw.subscribe("memory/encoded", self._on_memory_encoded)
        self.gw.subscribe("narrative/updated", self._on_narrative_updated)
        self.qa = QAEngine(self.stm, self.ltm, self.narr)
        
    # ---- Conversational QA ----
    def chat_once(self, user_text: str) -> str:
        # Store/learn from the input as usual…
        self.process(user_text)
        # …then answer using retrieval + focus + narrative
        return self.qa.answer(user_text)
    
    def caregiver_session(
        self,
        phrases: List[Tuple[str, float]],
        epochs: int = 3,
        shuffle: bool = True,
        autosave_every: int = 50,
        lexicon_path: str = 'lexicon.json',
        ltm_path: str = 'ltm.json',
        sleep_s: float = 0.0,
    ):
        """Run early-childhood style repetition from caregiver.
        phrases: list of (text, valence) where valence∈[-1,1].
        epochs: how many passes; shuffle for variability; optional sleep between turns.
        """
        import random, time as _time
        i = 0
        for e in range(max(1, epochs)):
            seq = list(phrases)
            if shuffle:
                random.shuffle(seq)
            for text, val in seq:
                self.process_input(text)
                self.feedback(text, val, debug=False)
                i += 1
                if autosave_every and (i % autosave_every == 0):
                    self.save_lexicon(lexicon_path)
                    self.save_memory(ltm_path)
                    TRACE.p("Caregiver.autosave", step=i)
                if sleep_s > 0:
                    _time.sleep(sleep_s)
        self.save_lexicon(lexicon_path)
        self.save_memory(ltm_path)
        return i
    

    def enable_tracing(self, flag: bool = True):
        set_trace(flag)
        self.evaluator.enable_debug(flag)


    # ---- explicit feedback API
    def feedback(self, text: str, valence: float, debug: Optional[bool] = None):
        """External teacher/designer feedback in [-1,1]."""
        if debug is not None:
            prev = self.evaluator.debug
            self.evaluator.enable_debug(debug)
            try:
                self.evaluator.learn(text, valence)
            finally:
                self.evaluator.enable_debug(prev)
        else:
            self.evaluator.learn(text, valence)

    # ---- persistence helpers for agent state ----
    def save_lexicon(self, path: str):
        self.evaluator.save_lexicon(path)

    def load_lexicon(self, path: str):
        self.evaluator.load_lexicon(path)

    def save_memory(self, ltm_path: str):
        self.ltm.save(ltm_path)

    def load_memory(self, ltm_path: str):
        self.ltm.load(ltm_path)

    # ---- GW listeners
    def _on_memory_encoded(self, payload: Dict[str, Any]):
        item: MemoryItem = payload["item"]
        self.mapper.ingest(item)
        # simplistic schema-link count proxy
        toks = self.mapper.tokenize(item.content)
        if toks:
            token = toks[0]
            if self.mapper.related(token, 1):
                self.maturity.update({"schema_link": 1})

    def _on_narrative_updated(self, payload: Dict[str, Any]):
        self.maturity.update({"self_reference": 1})

    # ---- Core ops
    def process_input(self, text: str) -> Dict[str, Any]:
        TRACE.p("Agent.input", text=text)
        # ---- store & GW ----
        item = self.stm.store(text)
        self.gw.broadcast("memory/encoded", {"item": item})

        # ---- prediction error (world model) ----
        pe = self.world.update_and_pe(text)
        TRACE.p("Agent.pred_error", pe=pe)

        # ---- attention/focus ----
        q = [len(text) % 10, item.emotion_score]
        fstm = self.stm.focus(q, top_k=2)
        fltm = self.ltm.focus(q, top_k=2)
        focus = fstm + fltm

        # ---- outputs ----
        summary = self.out.generate(focus)
        narrative = self.narr.build(self.ltm.recent(6))
        TRACE.p("Agent.output", summary=summary, narrative=narrative)
        self.gw.broadcast("narrative/updated", {"narrative": narrative})

        # ---- reward & adaptation ----
        ev = Event(
            description="input_processed",
            improves_prediction=len(focus) >= 2,
            receives_praise=False,
            strengthens_narrative=len(narrative) > 0
        )
        reward = self.reward.evaluate(ev)
        TRACE.p("Agent.reward", value=reward)

        # world-model based affect nudge (prediction error)
        if pe > 0.4:
            self.evaluator.learn(text, -0.3)
        elif pe < -0.4:
            self.evaluator.learn(text, +0.3)

        # reward-based adaptation
        if reward >= 0.6:
            self.evaluator.learn(text, +1.0)
        elif reward <= -0.2:
            self.evaluator.learn(text, -1.0)

        # ---- promotion to LTM (drive-aware) ----
        promoted = False
        if reward > 0.6 or drive_weight(item, self.drives) > 0.85:
            self.ltm.encode(item)
            promoted = True
            TRACE.p("Agent.promote", content=item.content)

        return {
            "stored": item,
            "focus": focus,
            "summary": summary,
            "narrative": narrative,
            "reward": reward,
            "maturity": self.maturity.score(),
        }

    def process(self, data: Any) -> List[Dict[str, Any]]:
        """General entrypoint: accepts a single word, a full sentence, or a list of them."""
        texts = self.adapter.prepare(data)
        results = []
        for t in texts:
            results.append(self.process_input(t))
        return results

    def default_mode_step(self) -> Dict[str, Any]:
        # seed from mild positive prior to avoid collapse
        seed = [5.0, 0.2]
        thoughts = self.loop.recurse(seed)
        # Convert a fraction of thoughts back into STM traces
        for t in thoughts[:2]:
            self.stm.store(t.content, metadata={"focus": True})
        # run one workspace competition tick
        winner = self.cycle.tick()
        narrative = self.narr.build(self.ltm.recent(6))
        return {
            "dmn_thoughts": thoughts,
            "workspace_winner": winner,
            "narrative": narrative,
            "maturity": self.maturity.score(),
        }

    def train_stream(
        self,
        source_iter: Iterable[str],
        autosupervise: bool = True,
        manual_labels: Optional[Iterable[Tuple[str, float]]] = None,
        save_every: int = 1000,
        lexicon_path: str = "lexicon.json",
        ltm_path: str = "ltm.json",
    ):
        """Stream a large corpus and learn continuously.
        - manual_labels: iterable of (text, valence) applied first (strong supervision)
        - autosupervise: use AutoTeacher to generate weak labels
        - save_every: checkpoint interval
        """
        teacher = AutoTeacher(self.evaluator) if autosupervise else None

        # strong supervision first
        if manual_labels:
            for text, val in manual_labels:
                self.feedback(text, float(max(-1.0, min(1.0, val))), debug=False)

        # stream the corpus
        i = 0
        for raw in source_iter:
            for text in self.adapter.prepare(raw):
                self.process_input(text)
                if teacher:
                    target = teacher.target_for(text)
                    if target is not None:
                        self.feedback(text, target, debug=False)
                i += 1
                if save_every and (i % save_every == 0):
                    self.save_lexicon(lexicon_path)
                    self.save_memory(ltm_path)
                    TRACE.p("Agent.autosave", i=i, lexicon_path=lexicon_path, ltm_path=ltm_path)
        # final save
        self.save_lexicon(lexicon_path)
        self.save_memory(ltm_path)

    def related_tokens(self, token: str, k: int = 5) -> List[Tuple[str, float]]:
        return self.mapper.related(token, top_k=k)


# -----------------------------
# Demo / Simulation
# -----------------------------
# -----------------------------
async def demo():
    agent = AGIAgent()
    agent.enable_tracing(False)  # turn off mid-process prints for clean output

    # Optionally load previous state
    # agent.load_lexicon('lexicon.json')
    # agent.load_memory('ltm.json')

    # Words and general sentences mixed
    inputs = [
        "사과",
        "기뻤어",
        "나는 오늘 사과를 먹었어. 맛있었어!",
        "비가 와서 우울했어? 짜증나…",
        ["여행", "행복", "친구와 화해"],
        "I enjoyed the walk and felt happy.",
    ]

    print("=== AGI CONSCIOUSNESS SIMULATION ===")
    print("Processing inputs and building consciousness...")
    
    # Process all inputs
    for data in inputs:
        outs = agent.process(data)
        await asyncio.sleep(0.01)

    # Teacher feedback
    agent.feedback("행복했어", +1.0, debug=False)
    agent.feedback("짜증나", -1.0, debug=False)

    # Save learned lexicon and LTM memory
    agent.save_lexicon('lexicon.json')
    agent.save_memory('ltm.json')

    # Default Mode Loop
    for _ in range(2):
        d = agent.default_mode_step()
        await asyncio.sleep(0.01)

    # Final Results
    print("\n=== FINAL CONSCIOUSNESS STATE ===")
    q = [5.0, 0.9]
    focus = agent.stm.focus(q, top_k=3) + agent.ltm.focus(q, top_k=2)
    narrative = agent.narr.build(agent.ltm.recent())
    
    print(f"Consciousness Maturity: {agent.maturity.score():.2f}")
    print(f"Active Focus Items: {len(focus)}")
    print(f"Long-term Memories: {len(agent.ltm.store_)}")
    print(f"Short-term Memories: {len(agent.stm.buf)}")
    
    print("\n=== SELF-NARRATIVE ===")
    print(narrative)
    
    print("\n=== RELATED CONCEPTS ===")
    related = agent.related_tokens('사과')
    if related:
        print("Concepts related to '사과':")
        for concept, weight in related:
            print(f"  - {concept} (strength: {weight:.2f})")
    else:
        print("No related concepts found for '사과'")
    
    print("\n=== SIMULATION COMPLETE ===")


async def demo2_long(epochs: int = 8, steps: int = 200, sleep: float = 0.0):
    import random, asyncio as _asyncio
    agent = AGIAgent()
    agent.enable_tracing(False)

    print("=== PARENT–BABY CAREGIVER DEMO (Long Session) ===")
    # caregiver curriculum (repetitive & simple)
    lessons: List[Tuple[str, float]] = [
        ("안전은 중요해. 뜨거운 것은 만지면 아야 해.", -0.8),
        ("엄마가 안아줄게. 따뜻하고 안전해.", +1.0),
        ("배고프면 밥을 먹어야 해.", +0.8),
        ("물건을 던지면 위험해. 조심히 내려놓자.", -0.6),
        ("잘했어! '고마워'라고 말해보자.", +0.9),
        ("슬프면 울어도 괜찮아. 숨을 크게 쉬어보자.", +0.6),
    ]

    # repeat caregiver lessons to stabilize associations
    agent.caregiver_session(
        phrases=lessons,
        epochs=max(1, epochs),
        shuffle=True,
        autosave_every=0,
        lexicon_path='lexicon.json',
        ltm_path='ltm.json',
        sleep_s=0.0,
    )

    # action→reaction pool (will be sampled WITH replacement = duplicates)
    episode_pool: List[Tuple[str, float]] = [
        ("응애… (울음)", +0.5),
        ("배고파", +0.8),
        ("손이 뜨거워!", -0.9),
        ("장난감 던졌어", -0.6),
        ("고마워", +0.9),
        ("하이파이브!", +0.7),
        ("무서워", -0.7),
        ("졸려", +0.3),
        ("아야", -0.8),
        ("더 놀고 싶어", +0.4),
    ]

    # Build a long sequence with duplicates
    interactions: List[Tuple[str, float]] = [random.choice(episode_pool) for _ in range(max(steps, 1))]

    print("\n--- Episodes: call & reaction (long sequence) ---")
    for utter, val in interactions:
        agent.process(utter)
        agent.feedback(utter, val, debug=False)  # reinforce the association
        reply = agent.chat_once(utter)
        print(f"baby> {utter}")
        print("parent>", reply.split("\n")[0])
        await _asyncio.sleep(max(0.0, sleep))

    agent.save_lexicon('lexicon.json')
    agent.save_memory('ltm.json')

    print("\n=== DEMO2-LONG SUMMARY ===")
    print(f"maturity={agent.maturity.score():.2f} | LTM={len(agent.ltm.store_)} | STM={len(agent.stm.buf)}")



if __name__ == "__main__":
    import argparse, csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run built-in demo")
    parser.add_argument("--ingest", type=str, help="path to .txt/.jsonl file or directory for continuous learning")
    parser.add_argument("--autosupervise", action="store_true", help="enable weak labels during ingestion")
    parser.add_argument("--labels", type=str, help="TSV file: text<TAB>valence (-1..1)")
    parser.add_argument("--save-every", type=int, default=1000, help="checkpoint interval")
    parser.add_argument("--trace", action="store_true", help="print mid-process traces")
    
    parser.add_argument("--chat", action="store_true", help="interactive Q&A mode")
    parser.add_argument("--caregiver-script", type=str, help="TSV: text<TAB>valence for early education")
    parser.add_argument("--caregiver-epochs", type=int, default=3)
    parser.add_argument("--caregiver-shuffle", action="store_true")
    parser.add_argument("--caregiver-sleep", type=float, default=0.0)

    parser.add_argument("--demo2-long", action="store_true",
                        help="run long caregiver demo with many duplicate interactions")
    parser.add_argument("--demo2-epochs", type=int, default=8,
                        help="caregiver repetition epochs (demo2 & long)")
    parser.add_argument("--demo2-steps", type=int, default=200,
                        help="# of baby-parent interaction turns (long)")
    parser.add_argument("--demo2-sleep", type=float, default=0.0,
                        help="sleep seconds between turns (demo2 & long)")


    
    args = parser.parse_args()

    if args.demo:
        # run demo
        try:
            asyncio.run(demo())
        except Exception as e:
            import traceback
            print("[ERROR] Program crashed:")
            traceback.print_exc()
            input("Press Enter to exit...")

    elif args.demo2_long:
        try:
            asyncio.run(demo2_long(
                epochs=args.demo2_epochs,
                steps=args.demo2_steps,
                sleep=args.demo2_sleep
            ))
        except Exception:
            import traceback
            print("\n[ERROR] Program crashed:\n")
            traceback.print_exc()
            input("\nPress Enter to exit...")



    elif args.chat:
        agent = AGIAgent()
        agent.enable_tracing(False)  # clean output
        agent.load_lexicon('lexicon.json')
        agent.load_memory('ltm.json')
        print("=== AGI CHAT MODE ===")
        print("Type your question (empty line to quit).")
        while True:
            try:
                s = input("you> ").strip()
            except EOFError:
                break
            if not s:
                break
            reply = agent.chat_once(s)
            print("agi>", reply)
        agent.save_lexicon('lexicon.json')
        agent.save_memory('ltm.json')

    elif args.caregiver_script:
        import csv
        agent = AGIAgent()
        agent.enable_tracing(True)  # optional
        agent.load_lexicon('lexicon.json')
        agent.load_memory('ltm.json')

        phrases: List[Tuple[str, float]] = []
        with open(args.caregiver_script, 'r', encoding='utf-8', errors='ignore') as f:
            for row in csv.reader(f, delimiter='\t'):
                if not row: continue
                t = (row[0] or '').strip()
                if not t: continue
                try:
                    v = float(row[1])
                except Exception:
                    v = 0.0
                phrases.append((t, max(-1.0, min(1.0, v))))
        steps = agent.caregiver_session(
            phrases=phrases,
            epochs=args.caregiver_epochs,
            shuffle=args.caregiver_shuffle,
            sleep_s=args.caregiver_sleep,
        )
        print(f"[Done] Caregiver session steps={steps}. State saved.")



    elif args.ingest:
        agent = AGIAgent()
        if args.trace:
            agent.enable_tracing(True)
        # resume non-volatile state if present
        agent.load_lexicon('lexicon.json')
        agent.load_memory('ltm.json')

        manual = None
        if args.labels:
            manual = []
            with open(args.labels, 'r', encoding='utf-8', errors='ignore') as f:
                for row in csv.reader(f, delimiter='	'):
                    if not row:
                        continue
                    text = row[0].strip()
                    if not text:
                        continue
                    try:
                        val = float(row[1])
                    except Exception:
                        continue
                    manual.append((text, max(-1.0, min(1.0, val))))

        agent.train_stream(
            source_iter=iter_corpus(args.ingest),
            autosupervise=args.autosupervise,
            manual_labels=manual,
            save_every=args.save_every,
            lexicon_path='lexicon.json',
            ltm_path='ltm.json',
        )
        print("[Done] Ingestion completed and state saved (lexicon.json, ltm.json)")
    
    else:
        # Default: run demo if no specific mode selected
        try:
            asyncio.run(demo())
        except Exception as e:
            import traceback
            print("[ERROR] Program crashed:")
            traceback.print_exc()
            input("Press Enter to exit...")
