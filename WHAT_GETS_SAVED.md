# What Gets Saved When You Click "Save State"

## Yes, Your Trained Results ARE Saved! ✅

When you click **"Save State"** in the GUI (or call `save_memory()` and `save_lexicon()`), the following trained results are persisted:

## 1. Learned Emotion Associations (`lexicon.json`)

For the **physiological emotion system** (default), this includes:

- **Context → Emotion Mappings**: All learned associations between text contexts (embeddings) and emotional responses (valence, arousal)
- **Physiological State**: Current valence, arousal, heart rate, stress, and energy levels

**Example saved data structure:**
```json
{
  "associations": [
    {
      "embedding": [0.12, 0.45, ...],  // Context embedding
      "valence": 0.8,                  // Learned valence
      "arousal": 0.6                   // Learned arousal
    },
    // ... more associations
  ],
  "physiology": {
    "valence": 0.5,
    "arousal": 0.4,
    "heart_rate": 75.2,
    "stress": 0.2,
    "energy": 0.6
  }
}
```

**What this means:**
- Every time you use the **feedback slider** and click "Apply Feedback", the agent learns an association
- These associations are saved in `lexicon.json`
- When you reload, the agent remembers what contexts → emotions you taught it

## 2. Long-Term Memories (`ltm.json`)

This includes:

- **All LTM items**: Every memory that was promoted to long-term storage
- **Content**: The actual text of each memory
- **Emotion scores**: The emotion evaluation for each memory
- **Timestamps**: When each memory was created
- **Metadata**: Additional information (emotion keywords, focus flags, etc.)

**Example saved data structure:**
```json
[
  {
    "content": "I passed my exam!",
    "emotion_score": 0.75,
    "timestamp": 1234567890.123,
    "metadata": {
      "emotion_keywords": [...],
      "focus": true
    }
  },
  // ... more memories
]
```

**What this means:**
- All your conversations/interactions are stored
- The agent remembers what you told it
- Emotion evaluations for each memory are preserved
- The agent can recall and reference past conversations

## What Gets Loaded on Startup

When you start the GUI (or create a new `AGIAgent`), it automatically:

1. **Loads memories** from `ltm.json` (if it exists)
2. **Loads emotion associations** from `lexicon.json` (if it exists)

So your trained agent continues from where it left off!

## What Does NOT Get Saved

These are reset when you restart (by design, as they're runtime states):

- **STM (Short-Term Memory)**: Only recent items, cleared on restart
- **Relational Mapper edges**: Concept graph connections (rebuilt from memories)
- **World Model predictions**: Prediction error tracking (resets)
- **Current physiological state**: Starts from baseline (but learns patterns)
- **Maturity score counters**: Resets (but recalculates from loaded memories)

## Example Workflow

1. **You chat**: "I'm happy today!"
   - Agent processes and stores in STM/LTM
   - Agent evaluates emotion

2. **You give feedback**: Move slider to +0.8, click "Apply Feedback"
   - Agent learns: "happy today" context → positive emotion
   - This association is stored in memory

3. **You click "Save State"**:
   - `ltm.json`: Saves the memory "I'm happy today!"
   - `lexicon.json`: Saves the learned association "happy today" → positive emotion

4. **You close GUI and restart**:
   - Agent loads `ltm.json` → remembers "I'm happy today!"
   - Agent loads `lexicon.json` → knows "happy today" contexts are positive

5. **You chat again**: "I'm happy today! (again)"
   - Agent recognizes similar context
   - Uses learned association to evaluate emotion
   - Responds based on past experience

## Verification

You can verify what's saved by checking the JSON files:

```bash
# View learned emotion associations
cat lexicon.json

# View long-term memories
cat ltm.json
```

## Best Practices

1. **Save regularly**: Click "Save State" after important interactions
2. **Feedback is crucial**: Use the feedback slider to teach the agent
3. **Backup files**: Keep copies of `ltm.json` and `lexicon.json` if they're valuable
4. **Both files matter**: 
   - `ltm.json` = what the agent remembers (content)
   - `lexicon.json` = how the agent interprets emotions (training)

## Summary

✅ **YES, your trained results ARE saved!**
- Emotion associations (feedback training) → `lexicon.json`
- Long-term memories (conversations) → `ltm.json`
- Both are automatically loaded on startup

The agent "remembers" what you taught it and what you told it! 🧠💾
