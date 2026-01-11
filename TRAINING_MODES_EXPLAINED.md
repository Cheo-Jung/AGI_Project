# Training Modes in the GUI: Manual Feedback vs Batch Training

## Overview

The GUI has two training modes that let you teach the AGI agent in different ways:

1. **Manual Feedback** - Interactive, one-at-a-time training
2. **Batch Training** - Process many texts from a file at once

---

## Mode 1: Manual Feedback Training

**Location**: "Emotion Feedback (Learning)" section (top of right panel)

### How It Works:

1. **Send a message** to the AGI (type in chat and press Send)
2. **Adjust the feedback slider** to indicate the emotion you want:
   - **Left side (-1.0)**: Negative emotion (sad, angry, etc.)
   - **Center (0.0)**: Neutral
   - **Right side (+1.0)**: Positive emotion (happy, excited, etc.)
3. **Click "Apply Feedback"** button
4. The agent learns: "When I see text like this, the emotion should be X"

### What Happens Behind the Scenes:

```
User sends: "I'm happy today!"
User sets slider to: +0.8 (positive)
User clicks "Apply Feedback"

→ Agent learns association:
   Text context → (valence=0.8, arousal=0.74)
   
→ Stored in: lexicon.json (as context association)
→ Training log shows: "Feedback training: 'I'm happy today!' → valence=0.80"
```

### Characteristics:
- **Interactive**: You control exactly what gets learned
- **Precise**: You decide the exact emotion value
- **One at a time**: Train on individual messages
- **Best for**: Teaching specific examples, correcting mistakes, fine-tuning

### Example:
```
You: "That movie was terrible!"
Slider: -0.9 (very negative)
Click "Apply Feedback"
→ Agent learns: "terrible" contexts → negative emotion

You: "I love chocolate!"
Slider: +0.7 (positive)
Click "Apply Feedback"
→ Agent learns: "love" contexts → positive emotion
```

---

## Mode 2: Batch Training

**Location**: "Training Panel" section (right panel, below Emotion Feedback)

### How It Works:

1. **Select "Batch Training"** radio button (in Training Mode)
2. **Click "Browse..."** to select a training file (.txt or .jsonl)
3. **Choose options**:
   - ☑ **Enable Auto-Supervision**: System generates emotion labels automatically
   - ☐ **Disable**: Only processes texts (stores in memory, no automatic labeling)
4. **Click "Start Training"** button
5. The agent processes all texts in the file

### What Happens Behind the Scenes:

```
File: corpus.txt (contains 1000 sentences)
Enable Auto-Supervision: ✓

→ For each text in file:
   1. Process text → Store in memory
   2. Evaluate emotion → Update cognitive state
   3. (If auto-supervision enabled):
      - Generate weak emotion labels automatically
      - Create associations if conditions met
   4. Update statistics

→ Progress shown in Training Log
→ Statistics updated: Items Processed, Associations Learned, Memories Created
```

### File Formats Supported:

**Text file (.txt)**:
```
I'm happy today!
The weather is nice.
This is terrible.
I love learning.
```

**JSONL file (.jsonl)**:
```
{"text": "I'm happy today!"}
{"text": "The weather is nice."}
{"content": "This is terrible."}
```

### Characteristics:
- **Bulk processing**: Train on many texts at once
- **Automated**: Can generate labels automatically (auto-supervision)
- **Scalable**: Process thousands of texts
- **Best for**: Training on large datasets, corpus learning, bulk knowledge

### Auto-Supervision Option:

**When Enabled (✓)**:
- System automatically generates emotion labels based on patterns
- Creates associations between contexts and emotions
- Weak supervision (not as precise as manual feedback)
- Helps learn from unlabeled data

**When Disabled (☐)**:
- Only processes texts and stores them in memory
- No automatic emotion labeling
- Texts are stored for later retrieval/attention
- No new emotion associations created

### Example:
```
File: conversations.txt (500 lines)

Batch Training:
- File selected: conversations.txt
- Auto-Supervision: ✓ Enabled
- Click "Start Training"

→ Processes 500 texts
→ Creates ~50-100 emotion associations (auto-supervised)
→ Stores all texts in memory
→ Updates: Items Processed: 500, Associations Learned: 75, Memories Created: 120
```

---

## Comparison Table

| Feature | Manual Feedback | Batch Training |
|---------|----------------|----------------|
| **Speed** | Slow (one at a time) | Fast (bulk processing) |
| **Precision** | High (you control exact value) | Medium/Low (automatic or none) |
| **Control** | Full control | Limited control |
| **Best For** | Specific examples, corrections | Large datasets, corpus learning |
| **Input** | Individual messages | File with many texts |
| **Output** | One association per feedback | Many associations/processings |
| **Interactivity** | Interactive (you provide feedback) | Automated (runs in background) |

---

## When to Use Which?

### Use Manual Feedback When:
- ✅ You want to teach specific examples
- ✅ You need precise control over emotions
- ✅ You're correcting mistakes
- ✅ You're fine-tuning behavior
- ✅ You have a few important examples

### Use Batch Training When:
- ✅ You have a large dataset
- ✅ You want to process many texts quickly
- ✅ You have unlabeled data (use auto-supervision)
- ✅ You want to build general knowledge
- ✅ You're doing bulk corpus learning

---

## Combining Both

You can use both modes together:

1. **Batch train** on a large corpus to build general knowledge
2. **Manual feedback** on specific examples to refine/correct
3. **Save state** to preserve all learning

The agent learns from both methods and combines them in its emotion evaluation system.

---

## Training Statistics

Both modes update the **Training Stats**:
- **Associations Learned**: Number of context→emotion associations
- **Items Processed**: Number of texts processed
- **Memories Created**: Number of memories in LTM

The **Training Log** shows detailed progress for both modes.
