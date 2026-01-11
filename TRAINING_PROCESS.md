# Training Process: Inputs and Outputs

## Overview

The AGI system has multiple learning mechanisms. This document explains what the inputs and outputs are for each training process.

---

## Training Type 1: Explicit Feedback Training (Supervised Learning)

**When**: You use the feedback slider in the GUI and click "Apply Feedback"

### Input
```
Text: str                    # The message text (e.g., "I'm happy today!")
Target Valence: float        # Desired emotion score [-1.0, 1.0]
                             # From feedback slider: -1.0 (negative) to +1.0 (positive)
```

**Example Input:**
```python
text = "I passed my exam!"
target_valence = 0.8  # Positive emotion
```

### Processing Steps

1. **Text Embedding**: Text → 64-dimensional embedding vector
   ```python
   context_embedding = embedder.embed_query(text)  # [0.12, 0.45, ..., 0.89]
   ```

2. **Arousal Estimation**: Target arousal derived from valence
   ```python
   target_arousal = 0.5 + abs(target_valence) * 0.3
   # If valence = 0.8: arousal = 0.5 + 0.8 * 0.3 = 0.74
   ```

3. **Association Learning**: 
   - Check if similar context exists (cosine similarity > 0.8)
   - If found: Update existing association with learning rate
   - If not found: Add new association

### Output
```
Learned Association: {
    context_embedding: List[float],    # 64-dim vector representing the text context
    valence: float,                    # Learned valence [-1.0, 1.0]
    arousal: float                     # Learned arousal [0.0, 1.0]
}
```

**What Gets Stored:**
- Context embedding (text representation)
- Valence value (emotional positivity/negativity)
- Arousal value (activation level)

**Where It's Saved:** `lexicon.json` (as "associations")

---

## Training Type 2: Automatic Learning During Processing

**When**: Text is processed via `process_input()` or `chat_once()`

### Input
```
Text: str                    # User message or input text
```

**Example Input:**
```python
text = "I feel anxious about the exam"
```

### Processing Steps

1. **Emotion Evaluation**: Text → (valence, arousal) via cognitive appraisal
   ```python
   valence, arousal = appraisal.appraise(text)
   ```

2. **Physiological Update**: Update internal bodily state
   ```python
   physiology.update_from_emotion(valence, arousal)
   ```

3. **Memory Storage**: Store as MemoryItem in STM
   ```python
   item = MemoryItem(
       content=text,
       emotion_score=emotion_score,
       timestamp=now(),
       metadata={...}
   )
   ```

4. **Automatic Learning Triggers** (if conditions met):
   - **Prediction Error Learning**: If prediction error > 0.4 or < -0.4
     ```python
     if prediction_error > 0.4:
         learn(text, -0.3)  # Negative adjustment
     ```
   
   - **Reward-Based Learning**: If reward >= 0.6 or <= -0.2
     ```python
     if reward >= 0.6:
         learn(text, +1.0)  # Positive reinforcement
     ```

### Output
```
Memory Item: {
    content: str,                      # The text itself
    emotion_score: float,              # Computed emotion score [-1.0, 1.0]
    timestamp: float,                  # When it was stored
    metadata: dict                     # Additional info (emotion keywords, etc.)
}
```

**Additional Outputs:**
- Updated physiological state (valence, arousal, heart rate, stress, energy)
- Possibly learned associations (if automatic learning triggered)
- Memory promotion to LTM (if emotion_score >= 0.75 or flagged)

**Where It's Saved:** `ltm.json` (for promoted memories)

---

## Training Type 3: Batch/Stream Training

**When**: Using `train_stream()` method with corpus data

### Input
```
Corpus: Iterable[str]                 # Stream of text data
Manual Labels: Optional[List[Tuple[str, float]]]  # (text, valence) pairs
AutoSupervise: bool                   # Whether to use weak supervision
```

**Example Input:**
```python
corpus = ["Text 1", "Text 2", "Text 3", ...]
manual_labels = [
    ("I'm happy", 0.8),
    ("I'm sad", -0.7),
    ...
]
```

### Processing Steps

1. **Strong Supervision**: Apply manual labels first
   ```python
   for text, valence in manual_labels:
       feedback(text, valence)
   ```

2. **Stream Processing**: Process each text
   ```python
   for text in corpus:
       process_input(text)           # Store memory
       if autosupervise:
           target = auto_teacher.target_for(text)
           if target:
               feedback(text, target)  # Weak supervision
   ```

### Output
```
Multiple Learned Associations: [
    {embedding, valence, arousal},
    {embedding, valence, arousal},
    ...
]
Multiple Memory Items: [
    {content, emotion_score, timestamp, metadata},
    ...
]
```

**Where It's Saved:** `lexicon.json` and `ltm.json`

---

## Complete Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PROCESS                         │
└─────────────────────────────────────────────────────────────┘

INPUT: Text + Optional Target Valence
    │
    ├─→ [Text Embedding] → 64-dim vector
    │
    ├─→ [Cognitive Appraisal] → (valence, arousal)
    │        │
    │        ├─→ Goal Relevance
    │        ├─→ Goal Congruence  
    │        ├─→ Coping Potential
    │        └─→ Novelty
    │
    ├─→ [Physiological State Update]
    │        │
    │        ├─→ Valence update
    │        ├─→ Arousal update
    │        ├─→ Heart rate adjustment
    │        ├─→ Stress level
    │        └─→ Energy level
    │
    └─→ [Memory Storage]
             │
             ├─→ STM (Short-Term Memory)
             └─→ LTM (if promoted)

OUTPUT: 
    ├─→ Learned Association (context → emotion)
    │     └─→ Saved to lexicon.json
    │
    ├─→ Memory Item (content + emotion)
    │     └─→ Saved to ltm.json (if promoted)
    │
    └─→ Updated Physiological State
          └─→ Used for future emotion evaluation
```

---

## Input-Output Summary Table

| Training Type | Input | Output | Saved To |
|--------------|-------|--------|----------|
| **Explicit Feedback** | Text + Target Valence [-1, 1] | Context → (Valence, Arousal) association | `lexicon.json` |
| **Automatic (Processing)** | Text only | Memory Item + Possible association | `ltm.json` (if promoted) |
| **Batch/Stream** | Corpus + Optional labels | Multiple associations + memories | Both files |

---

## Detailed Example: Explicit Feedback Training

### Step-by-Step

1. **User Input (GUI)**:
   ```
   Message: "I passed my exam!"
   Feedback Slider: +0.8 (moved to the right)
   Click: "Apply Feedback"
   ```

2. **System Receives**:
   ```python
   text = "I passed my exam!"
   target_valence = 0.8
   ```

3. **Processing**:
   ```python
   # Step 1: Create embedding
   embedding = [0.12, 0.45, 0.67, ..., 0.89]  # 64 dimensions
   
   # Step 2: Calculate arousal
   target_arousal = 0.5 + abs(0.8) * 0.3 = 0.74
   
   # Step 3: Check for similar contexts
   similarity = cosine_similarity(embedding, existing_embeddings)
   # If similarity > 0.8: update existing
   # Otherwise: add new association
   
   # Step 4: Store association
   association = {
       "embedding": embedding,
       "valence": 0.8,
       "arousal": 0.74
   }
   ```

4. **Output**:
   ```python
   # Added to context_associations list
   context_associations.append((embedding, (0.8, 0.74)))
   ```

5. **Future Use**:
   ```python
   # When similar text appears:
   new_text = "I passed the test!"
   new_embedding = embedder.embed_query(new_text)
   similarity = cosine_similarity(new_embedding, learned_embedding)
   
   if similarity > 0.6:
       # Use learned association
       valence = 0.8  # From learned association
       arousal = 0.74
   ```

---

## What Gets Learned

### 1. Context-Emotion Mappings
- **Input**: Text context (as embedding)
- **Output**: Emotional response (valence, arousal)
- **Purpose**: Understand emotional significance of situations

### 2. Memory Patterns
- **Input**: Text content
- **Output**: Memory items with emotional tags
- **Purpose**: Remember experiences with emotional context

### 3. Physiological Responses
- **Input**: Emotional appraisals
- **Output**: Updated bodily state (heart rate, stress, energy)
- **Purpose**: Model realistic emotional responses

---

## Key Concepts

### Input Formats

1. **Text Input**: Raw string
   ```python
   "I'm feeling great today!"
   ```

2. **Embedded Input**: 64-dimensional vector
   ```python
   [0.12, 0.45, 0.67, ..., 0.89]  # 64 values
   ```

3. **Target Output**: Valence value
   ```python
   0.8   # Positive
   -0.6  # Negative
   0.0   # Neutral
   ```

### Output Formats

1. **Association**: (embedding, (valence, arousal))
   ```python
   (embedding_vector, (0.8, 0.74))
   ```

2. **Memory Item**: Structured object
   ```python
   MemoryItem(
       content="text",
       emotion_score=0.75,
       timestamp=1234567890.0,
       metadata={...}
   )
   ```

3. **Physiological State**: Bodily parameters
   ```python
   {
       "valence": 0.6,
       "arousal": 0.5,
       "heart_rate": 78.5,
       "stress": 0.2,
       "energy": 0.7
   }
   ```

---

## Training Data Flow

```
USER/GUI
    │
    ├─→ Text Input
    └─→ Feedback (Valence)
         │
         ↓
AGENT.feedback(text, valence)
    │
    ├─→ Embedder.embed_query(text)
    │   └─→ Context Embedding [64-dim]
    │
    └─→ Evaluator.learn(text, valence)
        │
        ├─→ CognitiveAppraisal.learn(embedding, valence, arousal)
        │   └─→ Store/Update Association
        │
        └─→ Save to lexicon.json
             │
             └─→ {"associations": [...], "physiology": {...}}
```

---

## Summary

### Training Inputs:
- ✅ **Text**: Raw message/content
- ✅ **Target Valence**: Desired emotion [-1.0, 1.0] (for supervised learning)
- ✅ **Context Embeddings**: 64-dimensional vector representations

### Training Outputs:
- ✅ **Learned Associations**: Context → (Valence, Arousal) mappings
- ✅ **Memory Items**: Stored experiences with emotional context
- ✅ **Updated Physiology**: Current bodily/emotional state

### What Gets Saved:
- ✅ **lexicon.json**: Learned emotion associations
- ✅ **ltm.json**: Long-term memories

The system learns continuously from interactions, building up associations between contexts and emotions through experience! 🧠💡

