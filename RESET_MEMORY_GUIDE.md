# How to Reset/Clear Memory and Start Fresh

## Overview

To start learning from the beginning, you can clear the STM (Short-Term Memory) and LTM (Long-Term Memory) of the AGI agent.

## Method 1: Using the Reset Method (Programmatic)

The `AGIAgent` class now has a `reset_memory()` method:

```python
from AGI_v2 import AGIAgent

agent = AGIAgent()
# ... use the agent ...

# Clear all memories
agent.reset_memory()

# Clear memories AND emotion associations
agent.reset_memory(clear_emotion=True)
```

### What Gets Cleared:

- **STM (Short-Term Memory)**: All items in the short-term buffer
- **LTM (Long-Term Memory)**: All stored memories
- **Relational Mapper**: Concept relationship graph
- **Narrative Builder**: Self-narrative state
- **World Model**: Prediction error tracking
- **Maturation Monitor**: Maturity scores
- **Emotion Associations** (optional): Learned emotion mappings
- **Physiological State** (if using physiological emotion): Resets to default

## Method 2: Delete Saved Files

If you've saved state to files, delete them:

```bash
# Delete memory and lexicon files
rm ltm.json
rm lexicon.json
```

Or in Python:
```python
import os
if os.path.exists('ltm.json'):
    os.remove('ltm.json')
if os.path.exists('lexicon.json'):
    os.remove('lexicon.json')
```

## Method 3: Create a New Agent

The simplest way to start fresh:

```python
from AGI_v2 import AGIAgent

# Create a new agent (starts with empty memory)
agent = AGIAgent()
```

## Complete Reset Example

```python
from AGI_v2 import AGIAgent
import os

# Create or load agent
agent = AGIAgent()

# Option 1: Reset memory programmatically
agent.reset_memory(clear_emotion=True)

# Option 2: Also delete saved files
if os.path.exists('ltm.json'):
    os.remove('ltm.json')
if os.path.exists('lexicon.json'):
    os.remove('lexicon.json')

# Now the agent is completely fresh
agent.process("Hello!")  # First memory
```

## Using in GUI

Currently, there's no GUI button for reset (you can add one if needed). To reset from GUI:

1. Close the GUI
2. Delete `ltm.json` and `lexicon.json` files
3. Restart the GUI

Or add a reset button to the GUI (future enhancement).

## Notes

- Resetting memory **does NOT** clear the agent's architecture/parameters
- Text embedder vocabulary is retained (for consistency)
- You may want to also reset the embedder if starting completely fresh:
  ```python
  agent.embedder = TextEmbedder(embedding_dim=64)
  ```

