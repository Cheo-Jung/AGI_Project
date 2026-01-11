# GUI Training Features Guide

## Overview

The GUI now includes comprehensive training features that allow you to train the AGI agent in multiple ways.

## Training Methods Available in GUI

### 1. Manual Feedback Training (Already Available)

**Location**: Right panel → "Emotion Feedback (Learning)" section

**How to Use**:
1. Send a message to the agent
2. Adjust the feedback slider (left = negative, right = positive)
3. Click "Apply Feedback"
4. The agent learns the emotion association

**What Happens**:
- Text → Context embedding (64-dim vector)
- Association created: Context → (Valence, Arousal)
- Saved to `lexicon.json` when you save state
- Training log shows the learning event

### 2. Batch Training (NEW!)

**Location**: Right panel → "Training Panel" section

**Features**:
- File selection for corpus training
- Auto-supervision option
- Real-time progress tracking
- Training statistics
- Training log

**How to Use**:

1. **Select Training File**:
   - Click "Browse..." button
   - Select a text file (.txt) or JSONL file (.jsonl)
   - File name appears next to "Corpus File:"

2. **Configure Options**:
   - **Auto-Supervision**: Checkbox to enable/disable weak supervision
     - Enabled: System generates labels automatically
     - Disabled: Only processes texts (no automatic labeling)

3. **Start Training**:
   - Click "Start Training" button
   - Progress bar shows training in progress
   - Status updates: "Training in progress..."
   - Statistics update in real-time

4. **Monitor Progress**:
   - **Training Stats** shows:
     - Associations Learned: Number of context→emotion associations
     - Items Processed: Number of texts processed
     - Memories Created: Number of memories stored in LTM
   - **Training Log** shows detailed progress messages

5. **Stop Training** (if needed):
   - Click "Stop" button
   - Training stops at next item
   - Progress is saved up to that point

**What Happens During Batch Training**:
```
For each text in file:
  1. Process text → Store in memory
  2. Evaluate emotion → Update physiological state
  3. (If auto-supervision enabled) Generate weak labels
  4. Create associations if conditions met
  5. Update statistics
  6. Log progress
```

**File Formats Supported**:
- **.txt files**: Each line is treated as a separate text
- **.jsonl files**: JSON Lines format, extracts 'text' or 'content' field
- **Directories**: Recursively processes all files

### 3. Automatic Learning (Always Active)

**When**: During normal conversation/processing

**What Happens**:
- Text is processed → Memory stored
- Emotion evaluated → Physiological state updated
- Automatic learning triggers (if conditions met):
  - Prediction error learning
  - Reward-based learning
- Training log shows automatic learning events

## Training Panel Interface

```
┌─────────────────────────────────────┐
│ Training Panel                      │
├─────────────────────────────────────┤
│ Training Mode:                      │
│ ○ Manual Feedback                   │
│ ● Batch Training                    │
│                                     │
│ Corpus File: [filename.txt] [Browse]│
│ ☑ Enable Auto-Supervision           │
│                                     │
│ Status: Training in progress...     │
│ [Progress Bar - animated]           │
│                                     │
│ [Start Training] [Stop]             │
│                                     │
│ Training Stats:                     │
│ Associations Learned: 42            │
│ Items Processed: 150                │
│ Memories Created: 87                │
└─────────────────────────────────────┘
```

## Training Log

The Training Log shows:
- Training start/completion
- Progress updates (every N items)
- Feedback training events
- Errors (if any)
- Statistics summaries

**Example Log**:
```
[14:23:15] Training system ready. Use feedback slider or batch training.
[14:25:30] Starting batch training from: corpus.txt
[14:25:31] Processed 50/500 items...
[14:25:32] Processed 100/500 items...
[14:25:45] Feedback training: 'I'm happy today!' → valence=0.80
[14:26:10] Processed 500/500 items...
[14:26:10] Training completed: 500 items processed, 45 associations learned, 120 memories created.
```

## Training Statistics

Real-time statistics displayed:
- **Associations Learned**: Total context→emotion associations
- **Items Processed**: Number of texts processed
- **Memories Created**: Number of memories in LTM

These update during batch training and when using feedback.

## Best Practices

### For Manual Feedback Training:
1. **Be consistent**: Use similar valence for similar contexts
2. **Use frequently**: The more feedback, the better the agent learns
3. **Check training log**: Verify your feedbacks are being logged
4. **Save regularly**: Click "Save State" to persist training

### For Batch Training:
1. **Start small**: Test with a small file first
2. **Monitor progress**: Watch the log and statistics
3. **Use auto-supervision**: Helps with unlabeled data
4. **Stop if needed**: Use Stop button if something goes wrong
5. **Save after training**: Always save state after batch training

### File Preparation:
- **Text files**: One text per line
- **JSONL files**: Each line is JSON with 'text' or 'content' field
- **Encoding**: UTF-8 recommended
- **Size**: No strict limit, but very large files may take time

## Training Flow Summary

```
User Action
    │
    ├─→ Manual Feedback (Slider)
    │   └─→ Learn association immediately
    │
    ├─→ Batch Training (File)
    │   └─→ Process all texts in file
    │       └─→ Learn associations + memories
    │
    └─→ Normal Conversation
        └─→ Automatic learning (if conditions met)
```

## Example Training Workflow

1. **Initial Setup**:
   - Start GUI
   - Agent loads existing state (if available)

2. **Manual Training** (Quick Start):
   - Send: "I'm happy!"
   - Slider: +0.8
   - Click "Apply Feedback"
   - Check log: "Feedback training: 'I'm happy!' → valence=0.80"

3. **Batch Training** (Scale Up):
   - Click "Browse..." → Select corpus.txt
   - Enable "Auto-Supervision"
   - Click "Start Training"
   - Watch progress and statistics
   - Wait for completion
   - Check final stats

4. **Save Results**:
   - Click "Save State"
   - Training results saved to lexicon.json and ltm.json

5. **Continue Training**:
   - Repeat steps 2-4 as needed
   - Agent builds knowledge over time

## Troubleshooting

**Training doesn't start**:
- Check that file is selected
- Verify file exists and is readable
- Ensure agent is initialized

**Training stops unexpectedly**:
- Check training log for errors
- Verify file format is correct
- Check available memory

**Statistics don't update**:
- Wait a moment (updates in batches)
- Check if training is actually running
- Verify agent is using physiological emotion system

**Auto-supervision not working**:
- This is normal - AutoTeacher is designed for legacy evaluator
- Physiological evaluator handles learning differently
- Manual feedback still works perfectly

Enjoy training your AGI agent! 🧠📚
