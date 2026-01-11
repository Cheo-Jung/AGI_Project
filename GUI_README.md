# AGI GUI User Guide

## Overview

The AGI GUI provides an interactive graphical interface for the AGI Consciousness Simulator. It allows you to chat with the agent, view its physiological and memory state, and teach it emotions through feedback.

## Features

### 1. **Chat Interface**
- **Input Area**: Type your messages here
- **Send Button**: Click to send or press `Ctrl+Enter`
- **Chat Display**: Shows conversation history with timestamps
- **Clear Chat**: Clear the conversation history

### 2. **Physiological State Panel**
Real-time display of the agent's internal state:
- **Valence**: Pleasure-displeasure dimension [-1, 1]
- **Arousal**: Activation level [0, 1]
- **Heart Rate**: Simulated heart rate in BPM
- **Stress**: Stress level [0, 1]
- **Energy**: Energy level [0, 1]
- **Emotion Label**: Current emotion (e.g., "Happy", "Sad", "Excited")

### 3. **Memory Statistics**
- **STM Items**: Number of items in Short-Term Memory
- **LTM Items**: Number of items in Long-Term Memory
- **Maturity Score**: Consciousness maturity indicator [0, 1]

### 4. **Emotion Feedback System**
- **Feedback Slider**: Adjust valence from -1.0 (negative) to +1.0 (positive)
- **Apply Feedback**: Teach the agent by providing emotion labels for messages
- The agent learns associations between contexts and emotions

### 5. **State Management**
- **Save State**: Save agent's memory and learned associations to files
- State is automatically loaded on startup if files exist

## Usage

### Starting the GUI

```bash
python agi_gui.py
```

### Basic Interaction

1. **Type a message** in the input area
2. **Press Ctrl+Enter** or click "Send" to send
3. **View the response** in the chat display
4. **Monitor physiological state** in the right panel

### Teaching Emotions

1. Send a message to the agent
2. Adjust the **feedback slider** to indicate the emotion valence:
   - Move right → Positive emotion (happy, excited, content)
   - Move left → Negative emotion (sad, angry, upset)
   - Center → Neutral
3. Click **"Apply Feedback"** to teach the agent
4. The agent learns the association between that context and emotion

### Example Workflow

```
1. User: "I passed my exam!"
   → Agent responds

2. User adjusts slider to +0.8 (positive)
   → Clicks "Apply Feedback"

3. Agent learns: "passing exam" → positive emotion

4. Future similar messages will be interpreted as positive
```

### Saving State

- Click **"Save State"** to persist:
  - Long-term memories (`ltm.json`)
  - Learned emotion associations (`lexicon.json`)
- State is automatically loaded on next startup

## Keyboard Shortcuts

- **Ctrl+Enter**: Send message
- **Enter**: New line in input (when not holding Ctrl)
- **Window Close**: Prompts to save state

## Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  AGI Consciousness Simulator v2.0                           │
├──────────────────────────┬──────────────────────────────────┤
│                          │                                  │
│   Chat Interface         │   Physiological State            │
│   ┌──────────────────┐   │   Valence: +0.65                │
│   │ Conversation...  │   │   Arousal: 0.72                 │
│   │                  │   │   Heart Rate: 85.3 BPM          │
│   │                  │   │   Stress: 0.15                  │
│   │                  │   │   Energy: 0.78                  │
│   └──────────────────┘   │   Emotion: Happy                │
│                          │                                  │
│   Input:                 │   Memory Statistics              │
│   ┌──────────────────┐   │   STM: 12                       │
│   │ Type message...  │   │   LTM: 45                       │
│   └──────────────────┘   │   Maturity: 0.342               │
│                          │                                  │
│   [Send] [Clear] [Save]  │   Emotion Feedback              │
│                          │   ─────●─────────                │
│                          │   Positive (0.6)                │
│                          │   [Apply Feedback]               │
│                          │                                  │
│                          │   System Info                    │
│                          │   ┌──────────────┐               │
│                          │   │ Features...  │               │
│                          │   └──────────────┘               │
└──────────────────────────┴──────────────────────────────────┘
```

## Tips

1. **Start with simple messages** to see how the agent responds
2. **Use feedback frequently** to teach the agent your emotional associations
3. **Save state regularly** to preserve learned patterns
4. **Monitor physiological state** to understand the agent's internal processing
5. **Check memory statistics** to see how the agent's knowledge grows

## Troubleshooting

### GUI doesn't start
- Ensure Python has tkinter installed (usually included by default)
- On Linux, you may need: `sudo apt-get install python3-tk`

### Agent not responding
- Check the console for error messages
- Ensure `AGI_v2.py` is in the same directory
- Try restarting the application

### State not loading
- Check if `ltm.json` and `lexicon.json` exist
- Ensure files are valid JSON
- Check file permissions

### Feedback not working
- Make sure you've sent a message before applying feedback
- Check that the agent is using physiological emotion (default)
- Try restarting and sending a new message

## Technical Details

- **Framework**: tkinter (built-in Python GUI)
- **Threading**: Background processing to prevent GUI freezing
- **State Persistence**: JSON files for memory and associations
- **Compatibility**: Works with physiological emotion system (default)

## Advanced Usage

### Programmatic Access

The GUI uses the standard `AGIAgent` interface, so you can also:
- Access agent programmatically: `gui.agent`
- Use agent methods directly
- Integrate with other tools

### Customization

Modify `agi_gui.py` to:
- Change window size/layout
- Add new panels/displays
- Customize colors and fonts
- Add additional features

Enjoy interacting with your AGI consciousness simulator!
