"""
AGI Integrated Prototype (v2.0 - Enhanced with LLM-inspired Architecture)
--------------------------------------------------------------------------
Enhanced version incorporating modern LLM architecture principles while maintaining
cognitive architecture foundations.

Covers:
- Input → Feature Parsing → STM/LTM
- Enhanced Text Embeddings (LLM-inspired: character n-grams, word features, positional encoding)
- Multi-Head Attention (for memory focus and retrieval)
- Focusing Layer (enhanced attention with semantic similarity)
- Relational Mapping (lightweight concept graph)
- Affective Appraisal (EmotionEvaluator)
- Global Workspace (publish/subscribe)
- Interpretation / Self-Narrative Builder
- Output Generator
- Recursive Thought Loop with Layer Normalization & Residual Connections
- Reward System (designer feedback, prediction error, self-narrative coherence)
- Learning Loop (simple policy & memory promotion)
- Maturation Monitor (heuristics for "consciousness maturity")

Key Enhancements (LLM-inspired):
- Rich text embeddings (64-dim feature vectors with character n-grams, word features)
- Multi-head attention mechanism (4 heads, 16-dim per head)
- Layer normalization for stable computations
- Residual connections in thought loops
- Semantic similarity search (cosine similarity)
- Recency weighting for temporal relevance
- Larger context windows (STM: 32, enhanced retrieval)

No external dependencies beyond standard library.
Backward compatible with v1.1 (falls back to simple features if embedder not used).
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


def layer_norm(x: List[float], eps: float = 1e-9) -> List[float]:
    """Layer normalization (standardize to mean=0, std=1)."""
    if not x:
        return x
    mean = sum(x) / len(x)
    variance = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(variance + eps)
    return [(v - mean) / std for v in x]


def dot_product(a: List[float], b: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(a[i] * b[i] for i in range(min(len(a), len(b))))


def cosine_similarity(a: List[float], b: List[float], eps: float = 1e-9) -> float:
    """Cosine similarity between two vectors."""
    dot = dot_product(a, b)
    norm_a = math.sqrt(sum(v * v for v in a) + eps)
    norm_b = math.sqrt(sum(v * v for v in b) + eps)
    return dot / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0


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
# Text Embedding (LLM-inspired)
# -----------------------------
class TextEmbedder:
    """
    Enhanced text embedding system inspired by modern LLMs.
    Creates rich feature vectors from text using:
    - Character n-grams (1-3 grams)
    - Word-level features (length, word count, unique words)
    - Character frequency features
    - Positional features
    - Semantic-like features (common word patterns)
    """
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        # Vocabulary for word-based features (grows dynamically)
        self.word_to_idx: Dict[str, int] = {}
        self.word_counts: Dict[str, int] = defaultdict(int)
        self.next_idx = 0
        # Common function words (for semantic features)
        self.function_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "from", "by", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "이", "가", "을", "를", "의", "와", "과", "은", "는", "도", "로", "으로"
        }
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words (handles both English and CJK)."""
        tokens = []
        buf = ''
        for ch in text:
            if ch.isalnum() or ord(ch) > 127:
                buf += ch.lower() if ord(ch) < 128 else ch
            else:
                if buf:
                    tokens.append(buf)
                    buf = ''
        if buf:
            tokens.append(buf)
        return tokens
    
    def _get_char_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract character n-grams."""
        ngrams = []
        text_lower = text.lower()
        for i in range(len(text_lower) - n + 1):
            ngrams.append(text_lower[i:i+n])
        return ngrams[:20]  # Limit to prevent explosion
    
    def embed(self, text: str, emotion_score: float = 0.0, 
              position: int = 0, max_position: int = 100) -> List[float]:
        """
        Create embedding vector for text.
        Returns a fixed-size feature vector combining multiple signals.
        """
        tokens = self._tokenize(text)
        vec = []
        
        # 1. Length features (4 dims)
        text_len = len(text)
        vec.append(math.log(text_len + 1) / 10.0)  # Normalized log length
        vec.append(min(text_len / 100.0, 1.0))  # Length ratio
        vec.append(len(tokens) / 50.0)  # Word count normalized
        vec.append(len(set(tokens)) / max(len(tokens), 1))  # Vocabulary diversity
        
        # 2. Character n-gram features (12 dims: 3 n-grams × 4 features)
        for n in [1, 2, 3]:
            ngrams = self._get_char_ngrams(text, n)
            if ngrams:
                vec.append(len(ngrams) / 20.0)  # Count
                vec.append(len(set(ngrams)) / max(len(ngrams), 1))  # Diversity
                # Common n-gram indicator (simplified)
                common = sum(1 for ng in ngrams if ng in ["the", "ing", "ion", "ed ", "ly "])
                vec.append(common / max(len(ngrams), 1))
                vec.append(sum(ord(c) for ng in ngrams[:5] for c in ng) % 100 / 100.0)  # Hash-like
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0])
        
        # 3. Word-level features (8 dims)
        if tokens:
            # Word length statistics
            word_lens = [len(t) for t in tokens]
            vec.append(sum(word_lens) / len(word_lens) / 10.0)  # Avg word length
            vec.append(max(word_lens) / 20.0)  # Max word length
            # Function word ratio
            func_count = sum(1 for t in tokens if t in self.function_words)
            vec.append(func_count / len(tokens))
            # Capitalization (for English)
            caps = sum(1 for c in text if c.isupper())
            vec.append(caps / max(text_len, 1))
        else:
            vec.extend([0.0, 0.0, 0.0, 0.0])
        
        # 4. Character frequency features (10 dims)
        chars = text.lower()
        common_chars = "aeiouaeiou가나다라마"  # vowels + common CJK
        for ch in common_chars[:10]:
            vec.append(chars.count(ch) / max(text_len, 1))
        
        # 5. Semantic-like features (word patterns) (8 dims)
        if tokens:
            # Update word vocabulary
            for token in tokens[:10]:  # Limit to prevent explosion
                if token not in self.word_to_idx:
                    self.word_to_idx[token] = self.next_idx
                    self.next_idx += 1
                self.word_counts[token] += 1
            
            # Word ID features (hash-like)
            word_ids = [self.word_to_idx.get(t, 0) % 1000 for t in tokens[:8]]
            vec.extend([(id % 100) / 100.0 for id in word_ids[:8]])
        else:
            vec.extend([0.0] * 8)
        
        # 6. Emotion and metadata features (4 dims)
        vec.append(emotion_score)  # Emotion score
        vec.append(abs(emotion_score))  # Emotion intensity
        vec.append(1.0 if emotion_score > 0 else -1.0 if emotion_score < 0 else 0.0)  # Polarity
        vec.append(emotion_score * emotion_score)  # Emotion squared (non-linearity)
        
        # 7. Positional encoding (4 dims) - inspired by transformer positional encoding
        pos_ratio = position / max(max_position, 1)
        vec.append(math.sin(2 * math.pi * pos_ratio))  # Sin encoding
        vec.append(math.cos(2 * math.pi * pos_ratio))  # Cos encoding
        vec.append(math.sin(4 * math.pi * pos_ratio))  # Higher frequency
        vec.append(math.cos(4 * math.pi * pos_ratio))
        
        # 8. Padding/truncation to fixed size
        target_dim = self.embedding_dim
        if len(vec) < target_dim:
            # Pad with zeros
            vec.extend([0.0] * (target_dim - len(vec)))
        elif len(vec) > target_dim:
            # Truncate (keep first target_dim)
            vec = vec[:target_dim]
        
        # Normalize the vector (L2 normalization like in transformers)
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-9
        vec = [v / norm for v in vec]
        
        return vec
    
    def embed_query(self, text: str) -> List[float]:
        """Create query embedding (similar to embed but for search queries)."""
        return self.embed(text, emotion_score=0.0, position=0, max_position=1)


# Global embedder instance (can be shared or per-agent)
_default_embedder = TextEmbedder(embedding_dim=64)


# -----------------------------
# Memory Core
# -----------------------------
@dataclass
class MemoryItem:
    content: str
    emotion_score: float = 0.0
    timestamp: float = field(default_factory=now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _embedding_cache: Optional[List[float]] = field(default=None, init=False, repr=False)
    _position: int = field(default=0, init=False, repr=False)
    
    def key_vector(self, embedder: Optional[TextEmbedder] = None) -> List[float]:
        """
        Enhanced key vector using rich embeddings (LLM-inspired).
        Falls back to simple features if embedder not provided.
        """
        if embedder is not None:
            return embedder.embed(self.content, self.emotion_score, 
                                 self._position, max_position=100)
        # Fallback: simple key vector (backward compatible)
        return [len(self.content) % 10, self.emotion_score]

    def value_vector(self, embedder: Optional[TextEmbedder] = None) -> List[float]:
        """
        Enhanced value vector using rich embeddings.
        Can be different from key vector for richer representations.
        """
        if embedder is not None:
            # Value vector can include additional context
            base = embedder.embed(self.content, self.emotion_score, 
                                 self._position, max_position=100)
            # Add timestamp recency as additional feature
            age = now() - self.timestamp
            recency = math.exp(-age / 3600.0)  # Exponential decay (1 hour half-life)
            # Append recency to value (will be truncated if needed by embedder)
            return base + [recency, math.log(age + 1) / 10.0]
        # Fallback: simple value vector
        return [self.emotion_score, len(self.content) % 5]
    
    def get_embedding(self, embedder: Optional[TextEmbedder] = None) -> List[float]:
        """Get or compute embedding with caching."""
        if embedder is None:
            return self.key_vector()
        if self._embedding_cache is None:
            self._embedding_cache = embedder.embed(
                self.content, self.emotion_score, self._position, max_position=100
            )
        return self._embedding_cache
    
    def set_position(self, pos: int):
        """Set position for positional encoding."""
        self._position = pos
        self._embedding_cache = None  # Invalidate cache

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


# -----------------------------
# Physiologically-Grounded Emotion System
# -----------------------------

@dataclass
class PhysiologicalState:
    """
    Models bodily/physiological state that gives rise to emotions.
    Based on James-Lange theory: emotions arise from physiological responses.
    Uses Circumplex Model of Affect: emotions as valence × arousal.
    """
    valence: float = 0.0  # Pleasure-displeasure dimension [-1, 1]
    arousal: float = 0.0  # Activation-deactivation dimension [0, 1]
    heart_rate_base: float = 70.0  # Baseline heart rate (BPM)
    heart_rate_current: float = 70.0  # Current heart rate
    stress_level: float = 0.0  # Stress/activation [0, 1]
    energy_level: float = 0.5  # Energy/tiredness [0, 1]
    timestamp: float = field(default_factory=now)
    
    def update_from_emotion(self, valence: float, arousal: float, decay: float = 0.95):
        """Update physiological state based on emotion (with decay)."""
        # Exponential decay toward baseline
        self.valence = self.valence * decay + valence * (1 - decay)
        self.arousal = self.arousal * decay + arousal * (1 - decay)
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        
        # Physiological responses
        # Heart rate increases with arousal and valence (excitement) or negative valence (stress)
        hr_change = (abs(self.valence) * 20 + self.arousal * 30) * (1.0 if self.valence < 0 else 0.8)
        self.heart_rate_current = self.heart_rate_base + hr_change
        self.heart_rate_current = max(60.0, min(150.0, self.heart_rate_current))
        
        # Stress increases with negative valence and high arousal
        self.stress_level = max(0.0, -self.valence * 0.7 + self.arousal * 0.3)
        
        # Energy decreases with negative valence and low arousal (depression/fatigue)
        self.energy_level = 0.5 + self.valence * 0.3 + self.arousal * 0.2
        self.energy_level = max(0.0, min(1.0, self.energy_level))
        self.timestamp = now()
    
    def to_emotion_score(self) -> float:
        """Convert physiological state to emotion score (for compatibility)."""
        # Emotion score combines valence and arousal
        # High arousal amplifies valence
        return self.valence * (0.7 + 0.3 * self.arousal)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "heart_rate": self.heart_rate_current,
            "stress": self.stress_level,
            "energy": self.energy_level,
        }


class CognitiveAppraisal:
    """
    Evaluates situations/contexts to determine emotional significance.
    Based on Cognitive Appraisal Theory: emotions arise from evaluating situations.
    No hard-coded keywords - uses context understanding.
    """
    def __init__(self, embedder: Optional[TextEmbedder] = None):
        self.embedder = embedder or _default_embedder
        # Learned associations: context embeddings -> (valence, arousal) responses
        # Starts empty, learns from experience
        self.context_associations: List[Tuple[List[float], Tuple[float, float]]] = []
        self.learning_rate = 0.15
    
    def appraise(self, text: str, previous_state: Optional[PhysiologicalState] = None) -> Tuple[float, float]:
        """
        Appraise a situation and predict (valence, arousal) response.
        Returns: (valence, arousal) tuple
        """
        # Get context embedding (no keywords!)
        context_emb = self.embedder.embed_query(text)
        
        # Extract features from text that indicate appraisal dimensions
        # These are universal indicators, not language-specific keywords
        text_lower = text.lower()
        text_len = len(text)
        
        # Appraisal dimensions (based on psychological appraisal theory):
        # 1. Goal relevance (does this matter to me?)
        relevance = self._assess_relevance(text, context_emb)
        
        # 2. Goal congruence (is this good/bad for my goals?)
        congruence = self._assess_congruence(text, context_emb)
        
        # 3. Coping potential (can I handle this?)
        coping = self._assess_coping_potential(text, context_emb)
        
        # 4. Novelty/unexpectedness
        novelty = self._assess_novelty(text)
        
        # Compute valence and arousal from appraisals
        # Valence: primarily from goal congruence
        valence = congruence * relevance
        
        # Arousal: from relevance, novelty, and (low) coping potential
        arousal = relevance * (0.5 + 0.3 * novelty + 0.2 * (1.0 - coping))
        
        # Check learned associations (similar contexts)
        if self.context_associations:
            similarities = [
                cosine_similarity(context_emb, ctx_emb) 
                for ctx_emb, _ in self.context_associations
            ]
            if max(similarities) > 0.6:  # Similar context found
                best_idx = similarities.index(max(similarities))
                learned_val, learned_aro = self.context_associations[best_idx][1]
                # Blend learned and computed
                blend = max(similarities) * 0.7
                valence = valence * (1 - blend) + learned_val * blend
                arousal = arousal * (1 - blend) + learned_aro * blend
        
        # Apply constraints
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        
        return valence, arousal
    
    def _assess_relevance(self, text: str, context_emb: List[float]) -> float:
        """Assess goal relevance (0-1): does this matter?"""
        # Indicators: personal pronouns, action verbs, length (more content = more relevant)
        indicators = []
        text_lower = text.lower()
        
        # Personal relevance indicators (universal patterns, not keywords)
        personal_refs = sum(1 for word in ['i', 'me', 'my', '나', '내', '저', '우리'] 
                           if word in text_lower)
        indicators.append(min(personal_refs / 3.0, 1.0))
        
        # Content richness (longer, more detailed = more relevant)
        indicators.append(min(len(text) / 50.0, 1.0))
        
        # Question marks = seeking relevance
        indicators.append(min(text.count('?') / 2.0, 1.0))
        
        # Exclamation marks = high relevance
        indicators.append(min((text.count('!') + text.count('！')) / 3.0, 1.0))
        
        return sum(indicators) / max(len(indicators), 1)
    
    def _assess_congruence(self, text: str, context_emb: List[float]) -> float:
        """Assess goal congruence (-1 to 1): is this good or bad?"""
        # Use embedding features (high-dimensional semantic signals)
        # Positive signals: embedding features that correlate with positive contexts
        # Negative signals: features that correlate with negative contexts
        
        # Extract features from embedding
        if len(context_emb) >= 4:
            # Use embedding dimensions as indicators (learned patterns)
            # Positive: higher values in certain dimensions
            positive_signal = (context_emb[0] + context_emb[2]) / 2.0 if len(context_emb) > 2 else 0.0
            negative_signal = -(context_emb[1] + context_emb[3]) / 2.0 if len(context_emb) > 3 else 0.0
            signal = positive_signal + negative_signal
        else:
            signal = 0.0
        
        # Text intensity indicators (universal, not language-specific)
        excl_count = text.count('!') + text.count('！')
        excl_boost = min(excl_count / 3.0, 1.0)
        
        # Start neutral, bias by signals
        congruence = signal * 0.7 + (excl_boost - 0.5) * 0.3
        
        return max(-1.0, min(1.0, congruence))
    
    def _assess_coping_potential(self, text: str, context_emb: List[float]) -> float:
        """Assess coping potential (0-1): can I handle this?"""
        # Higher coping = lower arousal
        # Indicators: questions (uncertainty), length (complexity)
        text_lower = text.lower()
        
        # Uncertainty indicators (questions, uncertainty words - universal patterns)
        uncertainty = min((text.count('?') + text.count('maybe') + text.count('perhaps') + 
                          text.count('아마') + text.count('혹시')) / 4.0, 1.0)
        
        # Complexity (longer = harder to cope with)
        complexity = min(len(text) / 100.0, 1.0)
        
        # Lower coping = higher uncertainty + complexity
        coping = 1.0 - (uncertainty * 0.6 + complexity * 0.4)
        return max(0.0, min(1.0, coping))
    
    def _assess_novelty(self, text: str) -> float:
        """Assess novelty/unexpectedness (0-1)."""
        # Novelty indicators: questions, exclamations, length
        excl = text.count('!') + text.count('！')
        quest = text.count('?')
        length_novelty = min(len(text) / 80.0, 1.0)
        
        novelty = (min(excl / 2.0, 1.0) * 0.4 + min(quest / 2.0, 1.0) * 0.3 + length_novelty * 0.3)
        return max(0.0, min(1.0, novelty))
    
    def learn(self, text: str, target_valence: float, target_arousal: float):
        """Learn association between context and physiological response."""
        context_emb = self.embedder.embed_query(text)
        
        # Find similar contexts and update them
        updated = False
        for i, (ctx_emb, (val, aro)) in enumerate(self.context_associations):
            sim = cosine_similarity(context_emb, ctx_emb)
            if sim > 0.8:  # Very similar context
                # Update existing association
                new_val = val * (1 - self.learning_rate) + target_valence * self.learning_rate
                new_aro = aro * (1 - self.learning_rate) + target_arousal * self.learning_rate
                self.context_associations[i] = (ctx_emb, (new_val, new_aro))
                updated = True
                break
        
        # Add new association if not updated
        if not updated:
            self.context_associations.append((context_emb, (target_valence, target_arousal)))
            # Limit size (keep most recent/relevant)
            if len(self.context_associations) > 100:
                self.context_associations = self.context_associations[-100:]


class PureCognitiveEmotionEvaluator:
    """
    Pure cognitive emotion evaluator - consciousness without physiological simulation.
    Based on Buddhist philosophy: consciousness can arise from cognitive processes alone,
    without requiring the six sensory gates or bodily states.
    
    - No hard-coded keywords
    - Pure cognitive appraisal (goal relevance, congruence, coping, novelty)
    - Emotions emerge directly from cognitive evaluation
    - No physiological simulation (no heart rate, stress, etc.)
    - Learns associations through experience
    """
    def __init__(self, embedder: Optional[TextEmbedder] = None, 
                 learning_rate: float = 0.15, debug: bool = False):
        self.embedder = embedder or _default_embedder
        self.appraisal = CognitiveAppraisal(embedder=self.embedder)
        self.learning_rate = learning_rate
        self.debug = debug
        # Emotion history for temporal dynamics
        self.emotion_history: deque = deque(maxlen=20)
        # Current cognitive state (valence, arousal) - no physiology
        self.current_valence: float = 0.0
        self.current_arousal: float = 0.0
    
    def enable_debug(self, flag: bool = True):
        self.debug = bool(flag)
    
    def evaluate(self, text: str) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Evaluate emotion from text using pure cognitive appraisal.
        No physiological simulation - consciousness emerges from cognition alone.
        Returns: (emotion_score, detail_list) compatible with old interface
        """
        # Pure cognitive appraisal (no physiological state needed)
        valence, arousal = self.appraisal.appraise(text, previous_state=None)
        
        # Update cognitive state with decay (temporal continuity)
        decay = 0.92
        self.current_valence = self.current_valence * decay + valence * (1 - decay)
        self.current_arousal = self.current_arousal * decay + arousal * (1 - decay)
        self.current_valence = max(-1.0, min(1.0, self.current_valence))
        self.current_arousal = max(0.0, min(1.0, self.current_arousal))
        
        # Emotion emerges directly from cognitive state
        emotion_score = self.current_valence * (0.7 + 0.3 * self.current_arousal)
        
        # Store in history
        self.emotion_history.append({
            "valence": valence,
            "arousal": arousal,
            "emotion_score": emotion_score,
            "text": text[:50],
            "timestamp": now()
        })
        
        # Create detail list (for compatibility)
        detail = [
            ("valence", self.current_valence),
            ("arousal", self.current_arousal),
            ("cognitive_state", emotion_score),
        ]
        
        if self.debug:
            print(f"[CognitiveEmotion] text='{text[:40]}...' -> valence={valence:+.2f}, "
                  f"arousal={arousal:.2f}, score={emotion_score:+.2f}")
        
        return emotion_score, detail
    
    def learn(self, text: str, target_valence: float) -> Dict[str, float]:
        """
        Learn from feedback (compatible with old interface).
        target_valence: desired emotion score in [-1, 1]
        """
        # Convert emotion score to (valence, arousal)
        target_arousal = 0.5 + abs(target_valence) * 0.3
        
        # Learn the association (pure cognitive)
        self.appraisal.learn(text, target_valence, target_arousal)
        
        if self.debug:
            print(f"[CognitiveEmotion.learn] learned: '{text[:40]}...' -> "
                  f"valence={target_valence:+.2f}, arousal={target_arousal:.2f}")
        
        return {"valence": target_valence, "arousal": target_arousal}
    
    def get_emotion_label(self) -> str:
        """Get emotion label from cognitive state."""
        v, a = self.current_valence, self.current_arousal
        # Map to emotion labels (Circumplex Model)
        if a > 0.7:
            if v > 0.5:
                return "excited"
            elif v < -0.5:
                return "angry"
            else:
                return "alert"
        elif a < 0.3:
            if v > 0.5:
                return "content"
            elif v < -0.5:
                return "sad"
            else:
                return "calm"
        else:
            if v > 0.5:
                return "happy"
            elif v < -0.5:
                return "upset"
            else:
                return "neutral"
    
    # ---- persistence (compatible interface) ----
    def save_lexicon(self, path: str):
        """Save learned associations (compatible with old interface)."""
        import json
        data = {
            "associations": [
                {"embedding": emb, "valence": val, "arousal": aro}
                for emb, (val, aro) in self.appraisal.context_associations
            ],
            "cognitive_state": {
                "valence": self.current_valence,
                "arousal": self.current_arousal,
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        if self.debug:
            print(f"[CognitiveEmotion] saved to {path}")
    
    def load_lexicon(self, path: str):
        """Load learned associations."""
        import json, os
        if not os.path.exists(path):
            if self.debug:
                print(f"[CognitiveEmotion] load skipped; file not found: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Restore associations
        if "associations" in data:
            self.appraisal.context_associations = [
                (item["embedding"], (item["valence"], item["arousal"]))
                for item in data["associations"]
            ]
        # Restore cognitive state if available
        if "cognitive_state" in data:
            cs = data["cognitive_state"]
            self.current_valence = cs.get("valence", 0.0)
            self.current_arousal = cs.get("arousal", 0.0)
        if self.debug:
            print(f"[CognitiveEmotion] loaded from {path}")


class PhysiologicallyGroundedEmotionEvaluator:
    """
    Emotion evaluator based on human physiology and consciousness.
    - No hard-coded keywords
    - Uses cognitive appraisal to evaluate situations
    - Models physiological responses (valence × arousal)
    - Emotions emerge from bodily states (James-Lange theory)
    - Uses Circumplex Model of Affect
    - Learns associations through experience
    """
    def __init__(self, embedder: Optional[TextEmbedder] = None, 
                 learning_rate: float = 0.15, debug: bool = False):
        self.embedder = embedder or _default_embedder
        self.appraisal = CognitiveAppraisal(embedder=self.embedder)
        self.physiology = PhysiologicalState()
        self.learning_rate = learning_rate
        self.debug = debug
        # Emotion history for temporal dynamics
        self.emotion_history: deque = deque(maxlen=20)
    
    def enable_debug(self, flag: bool = True):
        self.debug = bool(flag)
    
    def evaluate(self, text: str) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Evaluate emotion from text using physiological modeling.
        Returns: (emotion_score, detail_list) compatible with old interface
        """
        # Step 1: Cognitive appraisal (evaluate the situation)
        valence, arousal = self.appraisal.appraise(text, self.physiology)
        
        # Step 2: Update physiological state (bodily response)
        self.physiology.update_from_emotion(valence, arousal, decay=0.92)
        
        # Step 3: Emotion emerges from physiology (James-Lange theory)
        emotion_score = self.physiology.to_emotion_score()
        
        # Store in history
        self.emotion_history.append({
            "valence": valence,
            "arousal": arousal,
            "emotion_score": emotion_score,
            "text": text[:50],
            "timestamp": now()
        })
        
        # Create detail list (for compatibility)
        detail = [
            ("valence", valence),
            ("arousal", arousal),
            ("heart_rate", self.physiology.heart_rate_current),
            ("stress", self.physiology.stress_level),
        ]
        
        if self.debug:
            print(f"[PhysioEmotion] text='{text[:40]}...' -> valence={valence:+.2f}, "
                  f"arousal={arousal:.2f}, score={emotion_score:+.2f}, "
                  f"HR={self.physiology.heart_rate_current:.1f}")
        
        return emotion_score, detail
    
    def learn(self, text: str, target_valence: float) -> Dict[str, float]:
        """
        Learn from feedback (compatible with old interface).
        target_valence: desired emotion score in [-1, 1]
        """
        # Convert emotion score to (valence, arousal)
        # Assume moderate arousal for feedback (can be refined)
        target_arousal = 0.5 + abs(target_valence) * 0.3
        
        # Learn the association
        self.appraisal.learn(text, target_valence, target_arousal)
        
        if self.debug:
            print(f"[PhysioEmotion.learn] learned: '{text[:40]}...' -> "
                  f"valence={target_valence:+.2f}, arousal={target_arousal:.2f}")
        
        return {"valence": target_valence, "arousal": target_arousal}
    
    def get_physiological_state(self) -> PhysiologicalState:
        """Get current physiological state."""
        return self.physiology
    
    def get_emotion_label(self) -> str:
        """Get emotion label from Circumplex Model."""
        v, a = self.physiology.valence, self.physiology.arousal
        # Map to emotion labels (Circumplex Model)
        if a > 0.7:
            if v > 0.5:
                return "excited"
            elif v < -0.5:
                return "angry"
            else:
                return "alert"
        elif a < 0.3:
            if v > 0.5:
                return "content"
            elif v < -0.5:
                return "sad"
            else:
                return "calm"
        else:
            if v > 0.5:
                return "happy"
            elif v < -0.5:
                return "upset"
            else:
                return "neutral"
    
    # ---- persistence (compatible interface) ----
    def save_lexicon(self, path: str):
        """Save learned associations (compatible with old interface)."""
        import json
        data = {
            "associations": [
                {"embedding": emb, "valence": val, "arousal": aro}
                for emb, (val, aro) in self.appraisal.context_associations
            ],
            "physiology": self.physiology.to_dict(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        if self.debug:
            print(f"[PhysioEmotion] saved to {path}")
    
    def load_lexicon(self, path: str):
        """Load learned associations."""
        import json, os
        if not os.path.exists(path):
            if self.debug:
                print(f"[PhysioEmotion] load skipped; file not found: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Restore associations
        if "associations" in data:
            self.appraisal.context_associations = [
                (item["embedding"], (item["valence"], item["arousal"]))
                for item in data["associations"]
            ]
        # Restore physiology state if available
        if "physiology" in data:
            p = data["physiology"]
            self.physiology.valence = p.get("valence", 0.0)
            self.physiology.arousal = p.get("arousal", 0.0)
        if self.debug:
            print(f"[PhysioEmotion] loaded from {path}")


# Legacy classes kept for backward compatibility
class EmotionEvaluator:
    """
    DEPRECATED: Hard-coded keyword-based emotion evaluator.
    Use PhysiologicallyGroundedEmotionEvaluator instead.
    """
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

class MultiHeadAttention:
    """
    Simplified multi-head attention mechanism (LLM-inspired).
    Splits query/key/value into multiple heads and combines results.
    """
    def __init__(self, num_heads: int = 4, head_dim: int = 16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
    
    def _split_heads(self, vec: List[float], head_dim: int) -> List[List[float]]:
        """Split vector into multiple heads (simplified: just chunk the vector)."""
        heads = []
        for i in range(0, len(vec), head_dim):
            head = vec[i:i+head_dim]
            if len(head) < head_dim:
                head.extend([0.0] * (head_dim - len(head)))
            heads.append(head[:head_dim])
        # If we have fewer heads than num_heads, duplicate or pad
        while len(heads) < self.num_heads:
            heads.append(heads[-1] if heads else [0.0] * head_dim)
        return heads[:self.num_heads]
    
    def _combine_heads(self, heads: List[List[float]]) -> List[float]:
        """Combine heads back into single vector."""
        combined = []
        for head in heads:
            combined.extend(head)
        return combined
    
    def attend(self, query: List[float], keys: List[List[float]], 
               values: List[List[float]], top_k: Optional[int] = None) -> Tuple[List[float], List[float]]:
        """
        Multi-head attention computation.
        Returns: (attention_weights, attended_values)
        """
        if not keys or not values:
            return [], []
        
        # Split query into heads
        q_heads = self._split_heads(query, self.head_dim)
        
        all_scores = []
        all_weights = []
        
        # Compute attention for each head
        for head_idx in range(self.num_heads):
            q_head = q_heads[head_idx] if head_idx < len(q_heads) else [0.0] * self.head_dim
            
            # Compute scores for this head
            scores = []
            for k in keys:
                k_head = self._split_heads(k, self.head_dim)[head_idx] if head_idx < len(self._split_heads(k, self.head_dim)) else [0.0] * self.head_dim
                # Dot product attention (scaled)
                score = dot_product(q_head, k_head) / math.sqrt(self.head_dim)
                scores.append(score)
            
            # Softmax weights
            weights = softmax(scores)
            all_weights.append(weights)
            
            # Weighted sum of values for this head
            head_output = [0.0] * self.head_dim
            for i, v in enumerate(values):
                v_head = self._split_heads(v, self.head_dim)[head_idx] if head_idx < len(self._split_heads(v, self.head_dim)) else [0.0] * self.head_dim
                for j in range(min(len(v_head), self.head_dim)):
                    head_output[j] += weights[i] * v_head[j]
            
            all_scores.append(head_output)
        
        # Combine heads
        combined_output = self._combine_heads(all_scores)
        
        # Average weights across heads for interpretability
        avg_weights = []
        if all_weights:
            for i in range(len(all_weights[0])):
                avg_weights.append(sum(w[i] for w in all_weights) / len(all_weights))
        
        return avg_weights, combined_output


class STM:
    def __init__(self, maxlen: int = 32, evaluator: Optional[EmotionEvaluator] = None,
                 embedder: Optional[TextEmbedder] = None, use_multihead: bool = True):
        self.buf: deque[MemoryItem] = deque(maxlen=maxlen)
        self.evaluator = evaluator or EmotionEvaluator()
        self.embedder = embedder or _default_embedder
        self.ltm: Optional[LTM] = None
        self.use_multihead = use_multihead
        if use_multihead:
            self.attention = MultiHeadAttention(num_heads=4, head_dim=16)

    def attach_ltm(self, ltm: 'LTM'):
        self.ltm = ltm

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        emo, detail = self.evaluator.evaluate(text)
        item = MemoryItem(content=text, emotion_score=emo, metadata={"emotion_keywords": detail, **(metadata or {})})
        # Set position based on buffer size
        item.set_position(len(self.buf))
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

    def focus(self, query: List[float], top_k: int = 2, 
              use_embedder: bool = True) -> List[MemoryItem]:
        """
        Enhanced focus with multi-head attention and rich embeddings.
        """
        if not self.buf:
            return []
        
        items = list(self.buf)
        
        if use_embedder and self.embedder:
            # Use rich embeddings
            query_vec = query if len(query) >= self.embedder.embedding_dim else self.embedder.embed_query(" ".join(str(q) for q in query[:10]))
            keys = [m.key_vector(self.embedder) for m in items]
            values = [m.value_vector(self.embedder) for m in items]
            
            if self.use_multihead:
                # Multi-head attention
                weights, attended = self.attention.attend(query_vec, keys, values, top_k=top_k)
                # Select top_k based on attention weights
                if weights:
                    idx_weights = [(i, w) for i, w in enumerate(weights)]
                    idx_weights.sort(key=lambda x: x[1], reverse=True)
                    idx = [i for i, w in idx_weights[:top_k]]
                    chosen = [items[i] for i in idx]
                else:
                    chosen = items[:top_k]
            else:
                # Standard scaled dot-product attention
                scores = [cosine_similarity(query_vec, k) for k in keys]
                weights = softmax(scores)
                idx_weights = [(i, w) for i, w in enumerate(weights)]
                idx_weights.sort(key=lambda x: x[1], reverse=True)
                idx = [i for i, w in idx_weights[:top_k]]
                chosen = [items[i] for i in idx]
        else:
            # Fallback to original simple attention
            keys = [m.key_vector() for m in items]
            scores = [dot_product(k, query) / math.sqrt(len(query)) for k in keys]
            weights = softmax(scores)
            idx_weights = [(i, w) for i, w in enumerate(weights)]
            idx_weights.sort(key=lambda x: x[1], reverse=True)
            idx = [i for i, w in idx_weights[:top_k]]
            chosen = [items[i] for i in idx]
        
        TRACE.p("STM.focus", query_len=len(query), top_k=top_k, chosen_count=len(chosen))
        return chosen


class LTM:
    def __init__(self, embedder: Optional[TextEmbedder] = None, use_multihead: bool = True):
        self.store_: List[MemoryItem] = []
        self.embedder = embedder or _default_embedder
        self.use_multihead = use_multihead
        if use_multihead:
            self.attention = MultiHeadAttention(num_heads=4, head_dim=16)

    def encode(self, item: MemoryItem):
        # Set position for positional encoding
        item.set_position(len(self.store_))
        self.store_.append(item)
        TRACE.p("LTM.encode", content=item.content, emotion=item.emotion_score)

    def recall_text(self, keyword: str) -> List[MemoryItem]:
        """Enhanced recall with semantic similarity (if embedder available)."""
        key = keyword.lower()
        # Basic keyword matching
        res = [m for m in self.store_ if key in m.content.lower()]
        
        # If embedder available, also use semantic similarity
        if self.embedder and len(res) < 5:
            query_emb = self.embedder.embed_query(keyword)
            # Compute similarity scores for all memories
            similarities = []
            for m in self.store_:
                mem_emb = m.get_embedding(self.embedder)
                sim = cosine_similarity(query_emb, mem_emb)
                similarities.append((m, sim))
            # Sort by similarity and add top matches
            similarities.sort(key=lambda x: x[1], reverse=True)
            for m, sim in similarities[:5]:
                if m not in res and sim > 0.3:  # Threshold
                    res.append(m)
        
        TRACE.p("LTM.recall_text", keyword=keyword, count=len(res))
        return res

    def focus(self, query: List[float], top_k: int = 3, 
              use_embedder: bool = True) -> List[MemoryItem]:
        """
        Enhanced focus with multi-head attention and rich embeddings.
        Includes recency weighting for temporal relevance.
        """
        if not self.store_:
            return []
        
        if use_embedder and self.embedder:
            # Use rich embeddings
            query_vec = query if len(query) >= self.embedder.embedding_dim else self.embedder.embed_query(" ".join(str(q) for q in query[:10]))
            keys = [m.key_vector(self.embedder) for m in self.store_]
            values = [m.value_vector(self.embedder) for m in self.store_]
            
            if self.use_multihead:
                # Multi-head attention
                weights, attended = self.attention.attend(query_vec, keys, values, top_k=top_k)
                # Apply recency weighting (more recent = higher weight)
                current_time = now()
                recency_weights = []
                for i, m in enumerate(self.store_):
                    age = current_time - m.timestamp
                    recency = math.exp(-age / 86400.0)  # 1 day half-life
                    recency_weights.append(recency)
                
                # Combine attention weights with recency
                if weights:
                    combined_weights = [w * (1.0 + 0.2 * r) for w, r in zip(weights, recency_weights)]
                    combined_weights = softmax(combined_weights)
                    idx_weights = [(i, w) for i, w in enumerate(combined_weights)]
                    idx_weights.sort(key=lambda x: x[1], reverse=True)
                    idx = [i for i, w in idx_weights[:top_k]]
                    chosen = [self.store_[i] for i in idx]
                else:
                    chosen = self.store_[-top_k:]
            else:
                # Standard scaled dot-product attention with recency
                scores = [cosine_similarity(query_vec, k) for k in keys]
                current_time = now()
                for i, m in enumerate(self.store_):
                    age = current_time - m.timestamp
                    recency = math.exp(-age / 86400.0)
                    scores[i] *= (1.0 + 0.2 * recency)
                weights = softmax(scores)
                idx_weights = [(i, w) for i, w in enumerate(weights)]
                idx_weights.sort(key=lambda x: x[1], reverse=True)
                idx = [i for i, w in idx_weights[:top_k]]
                chosen = [self.store_[i] for i in idx]
        else:
            # Fallback to original simple attention
            keys = [m.key_vector() for m in self.store_]
            scores = [dot_product(k, query) / math.sqrt(len(query)) for k in keys]
            weights = softmax(scores)
            idx_weights = [(i, w_val) for i, w_val in enumerate(weights)]
            idx_weights.sort(key=lambda x: x[1], reverse=True)
            idx = [i for i, w_val in idx_weights[:top_k]]
            chosen = [self.store_[i] for i in idx]
        
        TRACE.p("LTM.focus", query_len=len(query), top_k=top_k, chosen_count=len(chosen))
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
        
        # Sort by timestamp and deduplicate by content
        memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)  # Most recent first
        seen = set()
        unique_memories = []
        for m in memories:
            if m.content not in seen:
                seen.add(m.content)
                unique_memories.append(m)
        
        # Limit to most recent 5 unique items for conciseness
        unique_memories = unique_memories[:5]
        
        if not unique_memories:
            return "기억이 아직 부족해요."
        
        lines = ["최근 기억:"]
        for m in unique_memories:
            mood = "긍정" if m.emotion_score > 0 else "부정"
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
    """
    Enhanced thought loop with layer normalization and residual connections (LLM-inspired).
    """
    def __init__(self, stm: STM, ltm: LTM, depth: int = 2, use_normalization: bool = True):
        self.stm, self.ltm, self.max_depth = stm, ltm, depth
        self.use_normalization = use_normalization
        self.embedder = getattr(stm, 'embedder', None) or _default_embedder

    def recurse(self, seed: List[float], depth: int = 0, 
                residual: Optional[List[float]] = None) -> List[MemoryItem]:
        """
        Recursive thought traversal with residual connections and normalization.
        """
        if depth >= self.max_depth:
            return []
        TRACE.p("ThoughtLoop.recurse", depth=depth, seed_len=len(seed))
        
        # Normalize seed (like layer normalization in transformers)
        if self.use_normalization and seed:
            seed = layer_norm(seed)
        
        # Focus/attention over memories
        focused = self.stm.focus(seed, top_k=1, use_embedder=True) + \
                  self.ltm.focus(seed, top_k=1, use_embedder=True)
        
        out: List[MemoryItem] = []
        for m in focused:
            TRACE.p("ThoughtLoop.pick", depth=depth, memory=m.content[:30])
            out.append(m)
            
            # Get embeddings
            if self.embedder:
                key_emb = m.key_vector(self.embedder)
                val_emb = m.value_vector(self.embedder)
                # Combine key and value (like in transformers)
                # Use residual connection: next = seed + transformed(memory)
                combined = [k + v for k, v in zip(key_emb[:len(val_emb)], val_emb[:len(key_emb)])]
                if len(combined) < len(seed):
                    combined.extend([0.0] * (len(seed) - len(combined)))
                elif len(combined) > len(seed):
                    combined = combined[:len(seed)]
                
                # Residual connection: add seed to combined
                nxt = [seed[i] + 0.5 * combined[i] for i in range(min(len(seed), len(combined)))]
                if residual:
                    # Additional residual from previous layer
                    nxt = [nxt[i] + 0.3 * residual[i] for i in range(min(len(nxt), len(residual)))]
                
                # Normalize
                if self.use_normalization:
                    nxt = layer_norm(nxt)
            else:
                # Fallback: simple concatenation
                key_vec = m.key_vector()
                val_vec = m.value_vector()
                nxt = seed + key_vec + val_vec
            
            # Recursive call with residual
            out.extend(self.recurse(nxt, depth+1, residual=seed))
        
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


def load_word_emotion_pairs(path: str) -> List[Tuple[str, float]]:
    """Load word-emotion pairs from a file.
    
    Supports formats:
    - TSV/CSV: word<TAB>emotion_score or word<COMMA>emotion_score
    - JSON: list of {"word": str, "emotion": float} objects
    - JSONL: one JSON object per line with "word" and "emotion" fields
    
    Args:
        path: Path to the file containing word-emotion pairs
        
    Returns:
        List of (word/phrase, emotion_score) tuples where emotion_score ∈ [-1.0, 1.0]
    """
    import json
    pairs = []
    
    if path.endswith('.json'):
        # Single JSON file with list of objects
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    word = item.get('word') or item.get('text') or item.get('phrase') or ''
                    emotion = item.get('emotion') or item.get('valence') or item.get('score') or 0.0
                    if word:
                        pairs.append((str(word).strip(), float(emotion)))
            elif isinstance(data, dict):
                # Dictionary format: {"word1": score1, "word2": score2, ...}
                for word, emotion in data.items():
                    pairs.append((str(word).strip(), float(emotion)))
    
    elif path.endswith('.jsonl'):
        # JSONL format: one JSON object per line
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    word = obj.get('word') or obj.get('text') or obj.get('phrase') or ''
                    emotion = obj.get('emotion') or obj.get('valence') or obj.get('score') or 0.0
                    if word:
                        pairs.append((str(word).strip(), float(emotion)))
                except Exception:
                    continue
    
    else:
        # TSV/CSV format: word<TAB>emotion or word<COMMA>emotion
        import csv
        delimiter = '\t'  # Try tab first
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # Detect delimiter
            first_line = f.readline()
            f.seek(0)
            if ',' in first_line and '\t' not in first_line:
                delimiter = ','
            
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                word = row[0].strip()
                if not word:
                    continue
                try:
                    emotion = float(row[1])
                    pairs.append((word, emotion))
                except (ValueError, IndexError):
                    continue
    
    # Clamp emotion scores to valid range
    pairs = [(word, max(-1.0, min(1.0, emotion))) for word, emotion in pairs]
    
    return pairs


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
        self.embedder = getattr(stm, 'embedder', None)
    
    def tick(self) -> Optional[MemoryItem]:
        # 1) propose candidates (recent STM + a few from LTM)
        # Use enhanced focus if embedder available
        if self.embedder:
            # Create a default query embedding for default mode
            default_text = "default mode"
            q = self.embedder.embed_query(default_text)
            candidates = self.stm.recent(8) + self.ltm.focus(q, top_k=4, use_embedder=True)
        else:
            q = [5.0, 0.2]
            candidates = self.stm.recent(5) + self.ltm.focus(q, top_k=3, use_embedder=False)
        
        if not candidates:
            return None
        # 2) competition by drive-weighted salience
        scored = [(m, drive_weight(m, self.drives)) for m in candidates]
        winner = max(scored, key=lambda x: x[1])[0]
        TRACE.p("Cycle.tick", winner=winner.content[:50])
        # 3) broadcast winner
        self.gw.broadcast("workspace/winner", {"memory": winner})
        return winner
# -----------------------------
# QA Engine (simple retrieval + template answer)
# -----------------------------
class QAEngine:
    """Enhanced question-answering over STM/LTM with semantic embeddings."""
    def __init__(self, stm: 'STM', ltm: 'LTM', narrator: 'SelfNarrativeBuilder'):
        self.stm, self.ltm, self.narr = stm, ltm, narrator
        self.embedder = getattr(stm, 'embedder', None) or _default_embedder

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

    def _keyword_match(self, query: str, items: List[MemoryItem]) -> List[Tuple[MemoryItem, float]]:
        """Enhanced keyword matching with semantic similarity."""
        q_tokens = set(self._tokens(query))
        scored = []
        query_emb = self.embedder.embed_query(query)
        
        for m in items:
            # Token overlap score
            m_tokens = set(self._tokens(m.content))
            token_overlap = len(q_tokens & m_tokens) / max(len(q_tokens), 1)
            
            # Semantic similarity
            mem_emb = m.get_embedding(self.embedder)
            semantic_sim = cosine_similarity(query_emb, mem_emb)
            
            # Combined score (weighted)
            combined_score = 0.6 * token_overlap + 0.4 * semantic_sim
            scored.append((m, combined_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def answer(self, query: str) -> str:
        """
        Enhanced answer generation with semantic query embeddings.
        Generates a flexible response based on relevant memories without hard-coding.
        """
        # Generate query embedding (LLM-inspired)
        query_emb = self.embedder.embed_query(query)
        
        # 1) Retrieve from STM/LTM with enhanced matching
        recents = self.stm.recent(12)
        ltms = self.ltm.recent(64)
        km_stm = self._keyword_match(query, recents)[:4]
        km_ltm = self._keyword_match(query, ltms)[:6]
        
        # 2) Attention focus with semantic query
        f_stm = self.stm.focus(query_emb, top_k=3, use_embedder=True)
        f_ltm = self.ltm.focus(query_emb, top_k=4, use_embedder=True)
        f = f_stm + f_ltm
        
        # 3) Merge and deduplicate results
        seen_content = set()
        relevant_memories = []
        
        # Add keyword matches first (they have scores)
        for m, score in km_stm + km_ltm:
            if score > 0.1 and m.content not in seen_content:
                seen_content.add(m.content)
                relevant_memories.append((m, score))
        
        # Add focus results (avoid duplicates)
        for m in f:
            if m.content not in seen_content:
                seen_content.add(m.content)
                estimated_score = abs(m.emotion_score) * 0.8
                relevant_memories.append((m, estimated_score))
        
        # Sort by score (highest first)
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        
        # 4) Generate response based on context
        if relevant_memories:
            # Use top relevant memories to form response
            top_memories = [m for m, _ in relevant_memories[:3]]  # Top 3 most relevant
            
            # Form response from memories - let the content speak for itself
            if len(top_memories) == 1:
                return top_memories[0].content
            else:
                # Multiple relevant memories - combine or use most relevant
                return top_memories[0].content
        else:
            # No relevant memories found - echo back the query as acknowledgment
            return query


class AGIAgent:
    def __init__(self, embedding_dim: int = 64, use_multihead: bool = True,
                 use_physiological_emotion: bool = False, use_pure_cognitive: bool = True):
        """
        Enhanced AGI Agent with LLM-inspired architecture.
        
        Args:
            embedding_dim: Dimension of text embeddings (default 64)
            use_multihead: Whether to use multi-head attention (default True)
            use_physiological_emotion: Use physiologically-grounded emotion (default False)
                                     Simulates bodily states (heart rate, stress, etc.)
            use_pure_cognitive: Use pure cognitive emotion (default True)
                              Consciousness from cognitive processes alone, no physiology
                              Based on Buddhist philosophy: consciousness without sensory gates
        """
        self.gw = GlobalWorkspace()
        
        # Shared embedder for consistency (LLM-inspired)
        self.embedder = TextEmbedder(embedding_dim=embedding_dim)
        
        # Emotion evaluator selection
        if use_pure_cognitive:
            # Pure cognitive emotion - consciousness from cognition alone
            # No physiological simulation (aligns with Buddhist 12 links philosophy)
            self.evaluator = PureCognitiveEmotionEvaluator(
                embedder=self.embedder, 
                learning_rate=0.15, 
                debug=False
            )
        elif use_physiological_emotion:
            # Physiological emotion - includes bodily state simulation
            self.evaluator = PhysiologicallyGroundedEmotionEvaluator(
                embedder=self.embedder, 
                learning_rate=0.15, 
                debug=False
            )
        else:
            # Legacy evaluator (backward compatibility)
            self.evaluator = AdaptiveEmotionEvaluator(lr=0.12, debug=False)
        
        # Enhanced memory systems with embeddings and multi-head attention
        self.stm = STM(maxlen=32, evaluator=self.evaluator, 
                      embedder=self.embedder, use_multihead=use_multihead)
        self.ltm = LTM(embedder=self.embedder, use_multihead=use_multihead)
        self.stm.attach_ltm(self.ltm)
        
        self.mapper = RelationalMapper()
        self.narr = SelfNarrativeBuilder()
        self.out = OutputGenerator()
        self.loop = ThoughtLoop(self.stm, self.ltm, depth=2, use_normalization=True)
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
    
    def train_word_emotion_batch(
        self,
        word_emotion_pairs: List[Tuple[str, float]],
        process_text: bool = True,
        epochs: int = 1,
        save_every: int = 500,
        lexicon_path: str = 'lexicon.json',
        ltm_path: str = 'ltm.json',
    ) -> Dict[str, int]:
        """Batch training with word-emotion pairs.
        
        Args:
            word_emotion_pairs: List of (word/phrase, emotion_score) tuples where emotion_score ∈ [-1.0, 1.0]
            process_text: If True, also process each word through process_input (creates memories)
            epochs: Number of training passes (default: 1)
            save_every: Save checkpoint every N items (0 to disable intermediate saves)
            lexicon_path: Path to save lexicon
            ltm_path: Path to save long-term memory
        
        Returns:
            Dictionary with training statistics:
            {
                'items_processed': int,
                'associations_learned': int,
                'memories_created': int (if process_text=True)
            }
        """
        stats = {
            'items_processed': 0,
            'associations_learned': 0,
            'memories_created': 0
        }
        
        initial_ltm_size = len(self.ltm.store_)
        
        # Count initial associations (works for both evaluator types)
        initial_associations = 0
        if hasattr(self.evaluator, 'appraisal') and hasattr(self.evaluator.appraisal, 'context_associations'):
            initial_associations = len(self.evaluator.appraisal.context_associations)
        
        for epoch in range(max(1, epochs)):
            for word_or_phrase, emotion_score in word_emotion_pairs:
                # Clamp emotion score to valid range
                emotion_score = max(-1.0, min(1.0, float(emotion_score)))
                
                # Optionally process text to create memories
                if process_text:
                    self.process_input(word_or_phrase)
                
                # Learn emotion association
                self.feedback(word_or_phrase, emotion_score, debug=False)
                
                stats['items_processed'] += 1
                
                # Periodic save
                if save_every > 0 and stats['items_processed'] % save_every == 0:
                    self.save_lexicon(lexicon_path)
                    self.save_memory(ltm_path)
                    TRACE.p("WordEmotionBatch.autosave", items=stats['items_processed'])
        
        # Calculate final statistics
        stats['memories_created'] = len(self.ltm.store_) - initial_ltm_size
        if hasattr(self.evaluator, 'appraisal') and hasattr(self.evaluator.appraisal, 'context_associations'):
            stats['associations_learned'] = len(self.evaluator.appraisal.context_associations) - initial_associations
        else:
            stats['associations_learned'] = stats['items_processed']  # Approximation
        
        # Final save
        self.save_lexicon(lexicon_path)
        self.save_memory(ltm_path)
        
        return stats

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

        # ---- attention/focus (enhanced with embeddings) ----
        # Generate query embedding from input text (LLM-inspired)
        query_emb = self.embedder.embed_query(text)
        # Also include emotion as additional signal
        if len(query_emb) >= 2:
            query_emb[0] = (query_emb[0] + item.emotion_score) / 2.0
        
        fstm = self.stm.focus(query_emb, top_k=3, use_embedder=True)
        fltm = self.ltm.focus(query_emb, top_k=3, use_embedder=True)
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
    
    def reset_memory(self, clear_emotion: bool = False):
        """
        Clear STM and LTM to start learning from the beginning.
        
        Args:
            clear_emotion: If True, also clears learned emotion associations
        """
        # Clear STM
        self.stm.buf.clear()
        
        # Clear LTM
        self.ltm.store_.clear()
        
        # Clear emotion associations if requested
        if clear_emotion and hasattr(self.evaluator, 'appraisal'):
            if hasattr(self.evaluator.appraisal, 'context_associations'):
                self.evaluator.appraisal.context_associations.clear()
        
        # Reset physiological state if using physiological emotion
        if hasattr(self.evaluator, 'physiology'):
            self.evaluator.physiology = PhysiologicalState()
        
        # Clear relational mapper
        self.mapper = RelationalMapper()
        
        # Reset narrative builder
        self.narr = SelfNarrativeBuilder()
        
        # Reset world model
        self.world = WorldModel(k=5)
        
        # Reset maturity monitor
        self.maturity = MaturationMonitor()
        
        TRACE.p("Agent.reset_memory", clear_emotion=clear_emotion)


if __name__ == "__main__":
    import argparse, csv

    parser = argparse.ArgumentParser()
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

    parser.add_argument("--word-emotion-file", type=str, help="File with word-emotion pairs (TSV/CSV/JSON/JSONL): word<TAB>emotion")
    parser.add_argument("--word-emotion-epochs", type=int, default=1, help="Number of training epochs for word-emotion batch")
    parser.add_argument("--word-emotion-save-every", type=int, default=500, help="Save checkpoint every N items")
    parser.add_argument("--word-emotion-no-process", action="store_true", help="Don't process text through memory (only learn associations)")

    parser.add_argument("--gui", action="store_true", help="launch GUI interface")
    

    
    args = parser.parse_args()

    if args.gui:
        # Launch GUI
        try:
            import tkinter
            from agi_gui import main as gui_main
            gui_main()
        except ImportError as e:
            print(f"[ERROR] GUI not available: {e}")
            print("Make sure agi_gui.py is in the same directory.")
            input("Press Enter to exit...")
        except Exception as e:
            import traceback
            print("[ERROR] Failed to launch GUI:")
            traceback.print_exc()
            input("Press Enter to exit...")
    
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
    
    elif args.word_emotion_file:
        # Word-emotion batch training
        agent = AGIAgent()
        if args.trace:
            agent.enable_tracing(True)
        # Load existing state if present
        agent.load_lexicon('lexicon.json')
        agent.load_memory('ltm.json')
        
        print(f"=== WORD-EMOTION BATCH TRAINING ===")
        print(f"Loading word-emotion pairs from: {args.word_emotion_file}")
        
        try:
            pairs = load_word_emotion_pairs(args.word_emotion_file)
            print(f"Loaded {len(pairs)} word-emotion pairs")
            
            if len(pairs) == 0:
                print("[ERROR] No valid word-emotion pairs found in file")
                input("Press Enter to exit...")
            else:
                # Show sample
                print("\nSample pairs (first 5):")
                for word, emotion in pairs[:5]:
                    print(f"  '{word}' → {emotion:+.2f}")
                if len(pairs) > 5:
                    print(f"  ... and {len(pairs) - 5} more")
                
                print(f"\nStarting training (epochs={args.word_emotion_epochs}, process_text={not args.word_emotion_no_process})...")
                
                stats = agent.train_word_emotion_batch(
                    word_emotion_pairs=pairs,
                    process_text=not args.word_emotion_no_process,
                    epochs=args.word_emotion_epochs,
                    save_every=args.word_emotion_save_every,
                    lexicon_path='lexicon.json',
                    ltm_path='ltm.json',
                )
                
                print("\n=== TRAINING COMPLETE ===")
                print(f"Items processed: {stats['items_processed']}")
                print(f"Associations learned: {stats['associations_learned']}")
                print(f"Memories created: {stats['memories_created']}")
                print(f"State saved to: lexicon.json, ltm.json")
                
        except FileNotFoundError:
            print(f"[ERROR] File not found: {args.word_emotion_file}")
            import traceback
            traceback.print_exc()
            input("Press Enter to exit...")
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to process word-emotion file:")
            traceback.print_exc()
            input("Press Enter to exit...")
    
    else:
        # No mode specified - show help
        parser.print_help()
