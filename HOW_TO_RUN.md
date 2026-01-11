# How to Run AGI_v2.py with GUI

## Option 1: Run GUI directly (Recommended)

```bash
python agi_gui.py
```

This is the simplest way to launch the GUI.

## Option 2: Run AGI_v2.py with --gui flag

```bash
python AGI_v2.py --gui
```

This launches the GUI through the main AGI_v2.py script.

## Other Ways to Run AGI_v2.py

### Command-line Chat Mode
```bash
python AGI_v2.py --chat
```

### Demo Mode
```bash
python AGI_v2.py --demo
```

### Help
```bash
python AGI_v2.py --help
```

## Quick Start

1. **Open terminal/command prompt**
2. **Navigate to the project directory**
   ```bash
   cd C:\Users\pstcw\Desktop\AGI_Project
   ```
3. **Run the GUI**
   ```bash
   python agi_gui.py
   ```
   
   OR
   
   ```bash
   python AGI_v2.py --gui
   ```

## Requirements

- Python 3.x
- tkinter (usually included with Python)
  - On Linux, you may need: `sudo apt-get install python3-tk`

## GUI Features

Once the GUI opens, you can:
- Chat with the AGI agent
- View physiological state (valence, arousal, heart rate, etc.)
- See memory statistics
- Teach emotions using the feedback slider
- Save/load agent state

See `GUI_README.md` for detailed GUI usage instructions.
