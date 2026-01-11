"""
AGI_v2 GUI Application
======================
Interactive graphical user interface for the AGI agent system.
Uses tkinter (built-in Python GUI framework).
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
from datetime import datetime
from typing import Optional, List, Tuple
import json
import os
import csv

# Import AGI components
from AGI_v2 import AGIAgent, set_trace, iter_corpus, load_word_emotion_pairs


class AGIGUI:
    """Main GUI application for AGI Agent."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AGI Consciousness Simulator v2.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize agent
        self.agent: Optional[AGIAgent] = None
        self.init_agent()
        
        # Message queue for thread-safe GUI updates
        self.message_queue = queue.Queue()
        
        # Training state
        self.training_active = False
        self.selected_training_file = None
        self.selected_word_emotion_file = None
        
        # Create GUI components
        self.create_widgets()
        
        # Start message queue processor
        self.process_queue()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def init_agent(self):
        """Initialize the AGI agent."""
        try:
            self.agent = AGIAgent(use_pure_cognitive=True)  # Pure cognitive consciousness
            # Try to load existing state
            if os.path.exists('ltm.json'):
                try:
                    self.agent.load_memory('ltm.json')
                except:
                    pass
            if os.path.exists('lexicon.json'):
                try:
                    self.agent.load_lexicon('lexicon.json')
                except:
                    pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize agent: {e}")
            self.agent = None
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left panel (Chat and Input)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        main_frame.columnconfigure(0, weight=2)
        main_frame.rowconfigure(0, weight=1)
        
        # Right panel (State Information)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        main_frame.columnconfigure(1, weight=1)
        
        # === LEFT PANEL ===
        # Title
        title_label = ttk.Label(left_frame, text="AGI Chat Interface", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Chat display area
        chat_frame = ttk.LabelFrame(left_frame, text="Conversation", padding="5")
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=25,
            font=('Consolas', 10),
            bg='#ffffff',
            fg='#000000'
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = ttk.Frame(left_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        left_frame.columnconfigure(0, weight=1)
        
        ttk.Label(input_frame, text="Your message:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD, font=('Arial', 11))
        self.input_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        input_frame.columnconfigure(0, weight=1)
        
        # Bind Enter key (Ctrl+Enter to send, Enter for newline)
        self.input_text.bind('<Control-Return>', lambda e: self.send_message())
        self.input_text.bind('<Return>', lambda e: self.insert_newline() if not e.state else self.send_message())
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.send_button = ttk.Button(button_frame, text="Send (Ctrl+Enter)", 
                                      command=self.send_message, width=20)
        self.send_button.grid(row=0, column=0, padx=(0, 5))
        
        self.clear_button = ttk.Button(button_frame, text="Clear Chat", 
                                       command=self.clear_chat, width=15)
        self.clear_button.grid(row=0, column=1, padx=(0, 5))
        
        self.save_button = ttk.Button(button_frame, text="Save State", 
                                      command=self.save_state, width=15)
        self.save_button.grid(row=0, column=2, padx=(0, 5))
        
        self.reset_button = ttk.Button(button_frame, text="Reset Memory", 
                                       command=self.reset_memory, width=15)
        self.reset_button.grid(row=0, column=3)
        
        # === RIGHT PANEL ===
        # Memory Statistics
        memory_frame = ttk.LabelFrame(right_frame, text="Memory Statistics", padding="10")
        memory_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        self.memory_labels = {}
        memory_items = [
            ("STM Items", "stm_count"),
            ("LTM Items", "ltm_count"),
            ("Maturity Score", "maturity")
        ]
        
        for i, (label, key) in enumerate(memory_items):
            ttk.Label(memory_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            value_label = ttk.Label(memory_frame, text="0", font=('Arial', 10, 'bold'))
            value_label.grid(row=i, column=1, sticky=tk.W, padx=(10, 0))
            self.memory_labels[key] = value_label
        
        # Emotion Feedback
        feedback_frame = ttk.LabelFrame(right_frame, text="Emotion Feedback (Learning)", padding="10")
        feedback_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        right_frame.columnconfigure(0, weight=1)
        
        ttk.Label(feedback_frame, text="Last message valence:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.feedback_var = tk.DoubleVar(value=0.0)
        self.feedback_scale = ttk.Scale(
            feedback_frame, 
            from_=-1.0, 
            to=1.0, 
            orient=tk.HORIZONTAL,
            variable=self.feedback_var,
            length=200
        )
        self.feedback_scale.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        feedback_frame.columnconfigure(0, weight=1)
        
        self.feedback_label = ttk.Label(feedback_frame, text="Neutral (0.0)")
        self.feedback_label.grid(row=2, column=0, sticky=tk.W)
        
        self.feedback_button = ttk.Button(feedback_frame, text="Apply Feedback", 
                                         command=self.apply_feedback)
        self.feedback_button.grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        
        self.last_user_message = ""
        self.feedback_scale.bind('<Motion>', self.update_feedback_label)
        
        # Training Panel
        training_frame = ttk.LabelFrame(right_frame, text="Training Panel", padding="10")
        training_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        right_frame.columnconfigure(0, weight=1)
        
        # Training mode selector
        ttk.Label(training_frame, text="Training Mode:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.training_mode = tk.StringVar(value="manual")
        ttk.Radiobutton(training_frame, text="Manual Feedback", variable=self.training_mode, 
                       value="manual").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(training_frame, text="Batch Training", variable=self.training_mode, 
                       value="batch").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(training_frame, text="Word-Emotion Batch", variable=self.training_mode, 
                       value="word_emotion").grid(row=3, column=0, sticky=tk.W)
        
        # Batch training controls
        batch_frame = ttk.Frame(training_frame)
        batch_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 5))
        training_frame.columnconfigure(0, weight=1)
        
        self.batch_file_var = tk.StringVar(value="No file selected")
        ttk.Label(batch_frame, text="Corpus File:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        file_frame = ttk.Frame(batch_frame)
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        batch_frame.columnconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)
        
        ttk.Label(file_frame, textvariable=self.batch_file_var, 
                 font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(file_frame, text="Browse...", command=self.select_training_file,
                  width=10).grid(row=0, column=1, padx=(5, 0))
        
        # Auto-supervision checkbox
        self.autosupervise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(batch_frame, text="Enable Auto-Supervision", 
                       variable=self.autosupervise_var).grid(row=2, column=0, sticky=tk.W, pady=(5, 5))
        
        # Word-emotion training controls
        word_emotion_frame = ttk.Frame(training_frame)
        word_emotion_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 5))
        training_frame.columnconfigure(0, weight=1)
        
        self.word_emotion_file_var = tk.StringVar(value="No file selected")
        ttk.Label(word_emotion_frame, text="Word-Emotion File:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        word_emotion_file_frame = ttk.Frame(word_emotion_frame)
        word_emotion_file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        word_emotion_frame.columnconfigure(0, weight=1)
        word_emotion_file_frame.columnconfigure(0, weight=1)
        
        ttk.Label(word_emotion_file_frame, textvariable=self.word_emotion_file_var, 
                 font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(word_emotion_file_frame, text="Browse...", command=self.browse_word_emotion_file,
                  width=10).grid(row=0, column=1, padx=(5, 0))
        
        # Word-emotion options
        word_emotion_options_frame = ttk.Frame(word_emotion_frame)
        word_emotion_options_frame.grid(row=2, column=0, sticky=tk.W, pady=(5, 5))
        
        ttk.Label(word_emotion_options_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.word_emotion_epochs_var = tk.StringVar(value="1")
        word_emotion_epochs_entry = ttk.Entry(word_emotion_options_frame, textvariable=self.word_emotion_epochs_var, width=5)
        word_emotion_epochs_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 10))
        
        self.word_emotion_process_text_var = tk.BooleanVar(value=True)
        word_emotion_process_check = ttk.Checkbutton(word_emotion_options_frame, text="Process text (create memories)", 
                                                     variable=self.word_emotion_process_text_var)
        word_emotion_process_check.grid(row=0, column=2, sticky=tk.W)
        
        # Training progress
        self.training_progress_var = tk.StringVar(value="Ready")
        ttk.Label(batch_frame, text="Status:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(batch_frame, textvariable=self.training_progress_var, 
                 font=('Arial', 9, 'bold')).grid(row=4, column=0, sticky=tk.W)
        
        self.training_progress_bar = ttk.Progressbar(batch_frame, mode='indeterminate')
        self.training_progress_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(5, 5))
        batch_frame.columnconfigure(0, weight=1)
        
        # Training buttons
        train_button_frame = ttk.Frame(batch_frame)
        train_button_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.start_training_button = ttk.Button(train_button_frame, text="Start Training", 
                                               command=self.start_batch_training, width=15)
        self.start_training_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_training_button = ttk.Button(train_button_frame, text="Stop", 
                                              command=self.stop_training, width=10, state=tk.DISABLED)
        self.stop_training_button.grid(row=0, column=1)
        
        # Training statistics
        stats_frame = ttk.LabelFrame(training_frame, text="Training Stats", padding="5")
        stats_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N), pady=(10, 0))
        
        self.training_stats_labels = {}
        stats_items = [
            ("Associations Learned", "associations"),
            ("Items Processed", "processed"),
            ("Memories Created", "memories")
        ]
        
        for i, (label, key) in enumerate(stats_items):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            value_label = ttk.Label(stats_frame, text="0", font=('Arial', 9))
            value_label.grid(row=i, column=1, sticky=tk.W, padx=(10, 0))
            self.training_stats_labels[key] = value_label
        
        # Training Log
        log_frame = ttk.LabelFrame(right_frame, text="Training Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.rowconfigure(3, weight=1)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD, 
                                             font=('Consolas', 8), state=tk.DISABLED)
        self.training_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_training("Training system ready. Use feedback slider or batch training.")
        
        # Initial welcome message
        if self.agent:
            self.add_to_chat("AGI", "Hello! I'm an AGI consciousness simulator. "
                           "How can I help you today? You can teach me emotions using the feedback slider.")
            self.update_state_display()
    
    def insert_newline(self):
        """Insert newline in text widget (default Enter behavior)."""
        self.input_text.insert(tk.INSERT, '\n')
        return "break"
    
    def send_message(self):
        """Send user message to agent."""
        if not self.agent:
            messagebox.showerror("Error", "Agent not initialized!")
            return
        
        user_text = self.input_text.get('1.0', tk.END).strip()
        if not user_text:
            return
        
        # Store for feedback
        self.last_user_message = user_text
        
        # Clear input
        self.input_text.delete('1.0', tk.END)
        
        # Add user message to chat
        self.add_to_chat("You", user_text)
        
        # Process in background thread to avoid freezing GUI
        threading.Thread(target=self.process_message, args=(user_text,), daemon=True).start()
    
    def process_message(self, user_text: str):
        """Process message in background thread."""
        try:
            # Process input
            result = self.agent.process(user_text)
            
            # Get response
            response = self.agent.chat_once(user_text)
            
            # Update GUI via message queue (thread-safe)
            self.message_queue.put(('response', response))
            self.message_queue.put(('update_state', None))
            
        except Exception as e:
            self.message_queue.put(('error', str(e)))
    
    def add_to_chat(self, sender: str, message: str):
        """Add message to chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format message
        if sender == "You":
            prefix = f"[{timestamp}] You:\n"
            tag = "user"
        else:
            prefix = f"[{timestamp}] AGI:\n"
            tag = "agent"
        
        self.chat_display.insert(tk.END, prefix, tag)
        self.chat_display.insert(tk.END, message + "\n\n")
        
        # Configure tags
        self.chat_display.tag_config("user", foreground="#0066cc", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("agent", foreground="#cc6600", font=('Arial', 10, 'bold'))
        
        # Scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def update_feedback_label(self, event=None):
        """Update feedback label as slider moves."""
        val = self.feedback_var.get()
        if val > 0.3:
            label = f"Positive ({val:.2f})"
        elif val < -0.3:
            label = f"Negative ({val:.2f})"
        else:
            label = f"Neutral ({val:.2f})"
        self.feedback_label.config(text=label)
    
    def apply_feedback(self):
        """Apply emotion feedback to last message."""
        if not self.agent or not self.last_user_message:
            messagebox.showwarning("Warning", "No message to give feedback on!")
            return
        
        valence = self.feedback_var.get()
        try:
            self.agent.feedback(self.last_user_message, valence)
            self.add_to_chat("System", f"Learned emotion feedback: {valence:.2f} for last message")
            self.log_training(f"Feedback training: '{self.last_user_message[:40]}...' → valence={valence:.2f}")
            
            # Update training stats
            if hasattr(self.agent.evaluator, 'appraisal'):
                assoc_count = len(self.agent.evaluator.appraisal.context_associations)
                self.training_stats_labels["associations"].config(text=str(assoc_count))
            
            self.message_queue.put(('update_state', None))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply feedback: {e}")
            self.log_training(f"Feedback error: {e}")
    
    def update_state_display(self):
        """Update memory statistics."""
        if not self.agent:
            return
        
        try:
            # Memory statistics
            stm_count = len(self.agent.stm.buf)
            ltm_count = len(self.agent.ltm.store_)
            maturity = self.agent.maturity.score()
            
            self.memory_labels['stm_count'].config(text=str(stm_count))
            self.memory_labels['ltm_count'].config(text=str(ltm_count))
            self.memory_labels['maturity'].config(text=f"{maturity:.3f}")
            
        except Exception as e:
            print(f"Error updating state: {e}")
    
    def process_queue(self):
        """Process messages from queue (thread-safe GUI updates)."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == 'response':
                    self.add_to_chat("AGI", data)
                elif msg_type == 'update_state':
                    self.update_state_display()
                elif msg_type == 'error':
                    messagebox.showerror("Error", f"Processing error: {data}")
                elif msg_type == 'training_progress':
                    self.update_training_stats(data)
                elif msg_type == 'training_log':
                    self.log_training(data)
                elif msg_type == 'training_complete':
                    self.training_progress_bar.stop()
                    self.training_progress_var.set("Training complete!")
                    self.start_training_button.config(state=tk.NORMAL)
                    self.stop_training_button.config(state=tk.DISABLED)
                    self.update_training_stats(data)
                    self.log_training(f"Training completed: {data['processed']} items processed, "
                                    f"{data['associations']} associations learned, "
                                    f"{data['memories']} memories created.")
                    messagebox.showinfo("Training Complete", 
                                      f"Training completed!\n"
                                      f"Processed: {data['processed']} items\n"
                                      f"Associations: {data['associations']}\n"
                                      f"Memories: {data['memories']}")
                elif msg_type == 'training_error':
                    self.training_progress_bar.stop()
                    self.training_progress_var.set("Training error!")
                    self.start_training_button.config(state=tk.NORMAL)
                    self.stop_training_button.config(state=tk.DISABLED)
                    self.log_training(f"ERROR: {data}")
                    messagebox.showerror("Training Error", f"Training error: {data}")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)
    
    def clear_chat(self):
        """Clear chat display."""
        if messagebox.askyesno("Confirm", "Clear chat history?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete('1.0', tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.add_to_chat("System", "Chat cleared.")
    
    def save_state(self):
        """Save agent state to files."""
        if not self.agent:
            messagebox.showerror("Error", "Agent not initialized!")
            return
        
        try:
            self.agent.save_memory('ltm.json')
            self.agent.save_lexicon('lexicon.json')
            self.add_to_chat("System", "State saved successfully!")
            messagebox.showinfo("Success", "Agent state saved to ltm.json and lexicon.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save state: {e}")
    
    def reset_memory(self):
        """Reset agent memory (STM and LTM) to start fresh."""
        if not self.agent:
            messagebox.showerror("Error", "Agent not initialized!")
            return
        
        # Confirmation dialog
        if not messagebox.askyesno(
            "Reset Memory", 
            "This will clear all STM and LTM memories.\n"
            "The agent will start learning from scratch.\n\n"
            "Do you want to continue?"
        ):
            return
        
        try:
            # Reset memory
            self.agent.reset_memory(clear_emotion=True)
            
            # Update state display
            self.update_state_display()
            
            # Add message to chat
            self.add_to_chat("System", "Memory reset! Starting fresh.")
            
            # Log to training log if available
            if hasattr(self, 'log_training'):
                self.log_training("Memory reset - STM and LTM cleared")
            
            messagebox.showinfo("Success", "Memory has been reset. The agent is now starting fresh.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset memory: {e}")
    
    def select_training_file(self):
        """Select file for batch training."""
        filename = filedialog.askopenfilename(
            title="Select Training File",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSONL files", "*.jsonl"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.batch_file_var.set(os.path.basename(filename))
            self.selected_training_file = filename
    
    def log_training(self, message: str):
        """Add message to training log."""
        self.training_log.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.training_log.see(tk.END)
        self.training_log.config(state=tk.DISABLED)
    
    def start_batch_training(self):
        """Start batch training from selected file."""
        if not self.agent:
            messagebox.showerror("Error", "Agent not initialized!")
            return
        
        mode = self.training_mode.get()
        
        if mode == "batch":
            if not hasattr(self, 'selected_training_file') or not self.selected_training_file:
                messagebox.showwarning("Warning", "Please select a training file first!")
                return
            
            if not os.path.exists(self.selected_training_file):
                messagebox.showerror("Error", "Training file not found!")
                return
            
            if self.training_active:
                messagebox.showwarning("Warning", "Training already in progress!")
                return
            
            # Start batch training in background thread
            self.training_active = True
            self.start_training_button.config(state=tk.DISABLED)
            self.stop_training_button.config(state=tk.NORMAL)
            self.training_progress_bar.start()
            self.training_progress_var.set("Training in progress...")
            
            threading.Thread(target=self.run_batch_training, daemon=True).start()
        
        elif mode == "word_emotion":
            if not hasattr(self, 'selected_word_emotion_file') or not self.selected_word_emotion_file:
                messagebox.showwarning("Warning", "Please select a word-emotion file first!")
                return
            
            if not os.path.exists(self.selected_word_emotion_file):
                messagebox.showerror("Error", "Word-emotion file not found!")
                return
            
            if self.training_active:
                messagebox.showwarning("Warning", "Training already in progress!")
                return
            
            try:
                epochs = int(self.word_emotion_epochs_var.get())
                if epochs < 1:
                    epochs = 1
                    self.word_emotion_epochs_var.set("1")
            except ValueError:
                epochs = 1
                self.word_emotion_epochs_var.set("1")
            
            # Start word-emotion training in background thread
            self.training_active = True
            self.start_training_button.config(state=tk.DISABLED)
            self.stop_training_button.config(state=tk.NORMAL)
            self.training_progress_bar.start()
            self.training_progress_var.set("Training in progress...")
            
            threading.Thread(target=self.run_word_emotion_training, 
                           args=(self.selected_word_emotion_file, epochs), daemon=True).start()
    
    def run_batch_training(self):
        """Run batch training in background thread."""
        try:
            self.log_training(f"Starting batch training from: {os.path.basename(self.selected_training_file)}")
            
            # Reset stats
            self.training_stats_labels["associations"].config(text="0")
            self.training_stats_labels["processed"].config(text="0")
            self.training_stats_labels["memories"].config(text="0")
            
            initial_associations = 0
            if hasattr(self.agent.evaluator, 'appraisal'):
                initial_associations = len(self.agent.evaluator.appraisal.context_associations)
            initial_ltm = len(self.agent.ltm.store_)
            
            # Create corpus iterator
            corpus_iter = iter_corpus(self.selected_training_file)
            
            # Train with auto-supervision
            autosupervise = self.autosupervise_var.get()
            
            # Count items for progress
            items_processed = 0
            batch_size = 50  # Update progress every 50 items
            
            for text in corpus_iter:
                if not self.training_active:
                    break
                
                # Process each text (this handles automatic learning internally)
                self.agent.process_input(text)
                items_processed += 1
                
                # Update progress every batch
                if items_processed % batch_size == 0:
                    current_associations = 0
                    if hasattr(self.agent.evaluator, 'appraisal'):
                        current_associations = len(self.agent.evaluator.appraisal.context_associations)
                    current_ltm = len(self.agent.ltm.store_)
                    
                    self.message_queue.put(('training_progress', {
                        'processed': items_processed,
                        'associations': current_associations - initial_associations,
                        'memories': current_ltm - initial_ltm
                    }))
                    self.message_queue.put(('training_log', f"Processed {items_processed} items..."))
            
            # Final stats
            final_associations = 0
            if hasattr(self.agent.evaluator, 'appraisal'):
                final_associations = len(self.agent.evaluator.appraisal.context_associations)
            final_ltm = len(self.agent.ltm.store_)
            
            self.message_queue.put(('training_complete', {
                'processed': items_processed,
                'associations': final_associations - initial_associations,
                'memories': final_ltm - initial_ltm
            }))
            
        except Exception as e:
            self.message_queue.put(('training_error', str(e)))
        finally:
            self.training_active = False
    
    def browse_word_emotion_file(self):
        """Select file for word-emotion batch training."""
        filename = filedialog.askopenfilename(
            title="Select Word-Emotion File",
            filetypes=[
                ("TSV files", "*.tsv"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("JSONL files", "*.jsonl"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.word_emotion_file_var.set(os.path.basename(filename))
            self.selected_word_emotion_file = filename
    
    def run_word_emotion_training(self, file_path, epochs):
        """Run word-emotion batch training in background thread."""
        try:
            self.log_training(f"Loading word-emotion pairs from: {os.path.basename(file_path)}")
            
            pairs = load_word_emotion_pairs(file_path)
            total_items = len(pairs) * epochs
            
            if total_items == 0:
                self.message_queue.put(('training_error', "No valid word-emotion pairs found in file"))
                return
            
            self.log_training(f"Loaded {len(pairs)} word-emotion pairs")
            self.log_training(f"Starting training: {epochs} epoch(s), process_text={self.word_emotion_process_text_var.get()}")
            
            # Show sample
            sample_text = "\n".join([f"  '{word}' → {emotion:+.2f}" for word, emotion in pairs[:5]])
            if len(pairs) > 5:
                sample_text += f"\n  ... and {len(pairs) - 5} more"
            self.log_training(f"Sample pairs:\n{sample_text}")
            
            # Reset stats
            self.root.after(0, lambda: self.training_stats_labels["associations"].config(text="0"))
            self.root.after(0, lambda: self.training_stats_labels["processed"].config(text="0"))
            self.root.after(0, lambda: self.training_stats_labels["memories"].config(text="0"))
            
            stats = self.agent.train_word_emotion_batch(
                word_emotion_pairs=pairs,
                process_text=self.word_emotion_process_text_var.get(),
                epochs=epochs,
                save_every=500,  # Save every 500 items
                lexicon_path='lexicon.json',
                ltm_path='ltm.json',
            )
            
            self.log_training("\n=== TRAINING COMPLETE ===")
            self.log_training(f"Items processed: {stats['items_processed']}")
            self.log_training(f"Associations learned: {stats['associations_learned']}")
            self.log_training(f"Memories created: {stats['memories_created']}")
            self.log_training("State saved to: lexicon.json, ltm.json")
            
            # Update statistics labels and complete training
            self.message_queue.put(('training_complete', {
                'processed': stats['items_processed'],
                'associations': stats['associations_learned'],
                'memories': stats['memories_created']
            }))
            
        except Exception as e:
            import traceback
            error_msg = f"ERROR during word-emotion training: {str(e)}\n{traceback.format_exc()}"
            self.log_training(error_msg)
            self.message_queue.put(('training_error', str(e)))
        finally:
            self.training_active = False
    
    def stop_training(self):
        """Stop batch training."""
        self.training_active = False
        self.log_training("Training stopped by user.")
    
    def update_training_stats(self, stats: dict):
        """Update training statistics display."""
        if 'processed' in stats:
            self.training_stats_labels["processed"].config(text=str(stats['processed']))
        if 'associations' in stats:
            self.training_stats_labels["associations"].config(text=str(stats['associations']))
        if 'memories' in stats:
            self.training_stats_labels["memories"].config(text=str(stats['memories']))
    
    def on_closing(self):
        """Handle window closing."""
        if self.training_active:
            if not messagebox.askyesno("Training Active", "Training is in progress. Stop and quit?"):
                return
            self.training_active = False
        
        if messagebox.askyesno("Quit", "Do you want to save state before quitting?"):
            if self.agent:
                try:
                    self.agent.save_memory('ltm.json')
                    self.agent.save_lexicon('lexicon.json')
                except:
                    pass
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = AGIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

