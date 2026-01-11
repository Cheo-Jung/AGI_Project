# Physiologically-Grounded Emotion System

## Overview

Your AGI system now uses a **physiologically-grounded emotion evaluation system** that simulates human consciousness through bodily states, replacing the old hard-coded keyword-based approach.

## Philosophy: Emotion from Physiology

Based on psychological and neuroscientific theories:

1. **James-Lange Theory**: Emotions arise from physiological responses (we feel sad because we cry, not cry because we're sad)
2. **Circumplex Model of Affect**: Emotions are organized in a 2D space: **Valence** (pleasure-displeasure) × **Arousal** (activation)
3. **Cognitive Appraisal Theory**: Emotions emerge from evaluating situations/contexts

## Key Components

### 1. PhysiologicalState
Models the body's internal state:
- **Valence** [-1, 1]: Pleasure (positive) to displeasure (negative)
- **Arousal** [0, 1]: Low activation (calm) to high activation (excited)
- **Heart Rate**: Simulated physiological response
- **Stress Level**: Stress/activation indicator
- **Energy Level**: Energy/tiredness state

### 2. CognitiveAppraisal
Evaluates situations to determine emotional significance:
- **No hard-coded keywords** - uses context understanding
- Appraises dimensions:
  - **Goal relevance**: Does this matter?
  - **Goal congruence**: Is this good/bad for my goals?
  - **Coping potential**: Can I handle this?
  - **Novelty**: Is this unexpected?
- Learns associations between contexts and physiological responses

### 3. PhysiologicallyGroundedEmotionEvaluator
Main emotion evaluator that:
- Uses cognitive appraisal to evaluate situations
- Updates physiological state based on appraisals
- Emotions emerge from physiological states
- Learns from experience (no hard-coded dictionaries)

## How It Works

```
Input Text
    ↓
Cognitive Appraisal (evaluates situation)
    ↓
Generates (valence, arousal) prediction
    ↓
Updates Physiological State (bodily response)
    ↓
Emotion emerges from physiology
    ↓
Output: emotion_score (compatible with old system)
```

### Example Flow:

1. **Input**: "I am very happy today!"

2. **Cognitive Appraisal**:
   - Goal congruence: Positive (personal statement, exclamation)
   - Relevance: High (personal pronoun "I", exclamation mark)
   - Coping: High (positive statement)
   - Novelty: Moderate (exclamation indicates significance)
   - Result: High valence (+0.8), Moderate arousal (0.6)

3. **Physiological Response**:
   - Heart rate increases slightly
   - Stress decreases
   - Energy increases
   - State updates with decay (maintains temporal continuity)

4. **Emotion Emerges**:
   - Emotion score = valence × (0.7 + 0.3 × arousal)
   - Result: +0.8 × 0.88 ≈ +0.70 (positive emotion)

## Key Advantages

### ✅ No Hard-Coded Keywords
- Works with any language
- Understands context, not just words
- Adapts to new situations

### ✅ Biologically Plausible
- Models actual physiological processes
- Emotions emerge naturally from states
- Temporal continuity (emotions persist and decay)

### ✅ Learns from Experience
- Builds associations between contexts and responses
- Adapts through feedback
- No manual dictionary maintenance

### ✅ Theoretical Foundation
- Based on established psychological theories
- Grounded in neuroscience
- Models human consciousness more accurately

## Usage

### Default (Recommended):
```python
agent = AGIAgent(use_physiological_emotion=True)  # Default
```

### Legacy Mode (backward compatible):
```python
agent = AGIAgent(use_physiological_emotion=False)  # Uses old keyword-based system
```

### Accessing Physiological State:
```python
agent = AGIAgent()
agent.process("I feel anxious about the exam")

# Get physiological state
physio = agent.evaluator.get_physiological_state()
print(f"Valence: {physio.valence:.2f}")
print(f"Arousal: {physio.arousal:.2f}")
print(f"Heart Rate: {physio.heart_rate_current:.1f} BPM")
print(f"Stress: {physio.stress_level:.2f}")

# Get emotion label
label = agent.evaluator.get_emotion_label()
print(f"Emotion: {label}")
```

### Learning from Feedback:
```python
agent = AGIAgent()
agent.feedback("This situation is terrible", -0.8)  # Negative feedback
# System learns to associate this context with negative valence
```

## Emotion Mapping (Circumplex Model)

Emotions are mapped in 2D space:

```
High Arousal
    |
Angry ←------→ Excited
    |    |
    |  (Valence, Arousal)
    |
Upset ←------→ Happy
    |
Low Arousal
    |
Sad   ←------→ Content
```

- **Valence > 0.5, Arousal > 0.7**: Excited
- **Valence > 0.5, Arousal < 0.3**: Content
- **Valence < -0.5, Arousal > 0.7**: Angry
- **Valence < -0.5, Arousal < 0.3**: Sad
- **Valence > 0.5, 0.3 ≤ Arousal ≤ 0.7**: Happy
- **Valence < -0.5, 0.3 ≤ Arousal ≤ 0.7**: Upset
- **Else**: Neutral/Calm

## Comparison with Old System

| Aspect | Old (Keyword-based) | New (Physiological) |
|--------|-------------------|-------------------|
| **Approach** | Hard-coded dictionaries | Context understanding |
| **Language** | Korean/English only | Universal (any language) |
| **Learning** | Keyword weights | Context associations |
| **Theoretical** | Ad-hoc | James-Lange, Circumplex Model |
| **Physiology** | None | Simulated heart rate, stress, energy |
| **Temporal** | Instant | Continuous with decay |
| **Adaptability** | Limited | High (learns from experience) |

## Persistence

The system saves/loads learned associations:

```python
agent.save_lexicon('emotion_associations.json')  # Saves physiological associations
agent.load_lexicon('emotion_associations.json')  # Loads learned patterns
```

## Future Enhancements

Potential improvements:
1. **More physiological signals**: Cortisol, dopamine, serotonin levels
2. **Individual differences**: Personality traits affecting emotion
3. **Cultural factors**: Cultural variations in emotion expression
4. **Body language**: Incorporate posture, gesture indicators
5. **Memory integration**: Emotions from past experiences influence current appraisal

## References

- **James-Lange Theory**: Emotions as perception of bodily changes
- **Circumplex Model of Affect**: Russell (1980)
- **Cognitive Appraisal Theory**: Lazarus & Folkman (1984)
- **Psi-Theory**: Dörner's psychological theory of human behavior

This system brings your AGI closer to modeling actual human consciousness and emotion!

