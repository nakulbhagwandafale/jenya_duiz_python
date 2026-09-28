"""
Generative AI with LSTM - Text Generation
==========================================

This script implements a character-level text generator using an LSTM (Long Short-Term Memory)
neural network. It is trained on Shakespeare's complete works and generates new text based
on seed input.

Author: Nakul
Date: September 2026

Sections:
    1. Dataset Loading and Preprocessing
    2. Model Design
    3. Model Training
    4. Text Generation
    5. Architecture Experiments (Bonus)
"""

import os
import re
import argparse
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# ============================================================================
# 1. DATASET LOADING AND PREPROCESSING
# ============================================================================

def download_dataset(data_dir="data", filename="shakespeare.txt"):
    """
    Download Shakespeare's complete works from Project Gutenberg.
    
    Args:
        data_dir (str): Directory to store the dataset.
        filename (str): Name of the downloaded file.
    
    Returns:
        str: Path to the downloaded file.
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        print(f"[INFO] Dataset already exists at: {filepath}")
        return filepath
    
    url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    print(f"[INFO] Downloading Shakespeare's works from {url}...")
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print(f"[INFO] Dataset saved to: {filepath}")
    return filepath


def load_and_preprocess_text(filepath, max_chars=None):
    """
    Load text from file and perform preprocessing:
    - Convert to lowercase
    - Remove non-printable and unnecessary characters
    - Keep only letters, basic punctuation, and spaces
    
    Args:
        filepath (str): Path to the text file.
        max_chars (int): Maximum number of characters to use (None = all).
    
    Returns:
        str: Cleaned text string.
    """
    print("[INFO] Loading and preprocessing text...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove Project Gutenberg header/footer markers
    # The actual text starts after "*** START OF" and ends before "*** END OF"
    start_marker = "*** start of"
    end_marker = "*** end of"
    
    start_idx = text.find(start_marker)
    if start_idx != -1:
        # Move past the marker line
        start_idx = text.find("\n", start_idx) + 1
    else:
        start_idx = 0
    
    end_idx = text.rfind(end_marker)
    if end_idx == -1:
        end_idx = len(text)
    
    text = text[start_idx:end_idx]
    
    # Remove punctuation and special characters, keep letters, spaces, and basic marks
    text = re.sub(r"[^a-z\s.,;:!?'\-]", "", text)
    
    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()
    
    # Limit text length if specified (useful for faster training/testing)
    if max_chars:
        text = text[:max_chars]
    
    print(f"[INFO] Text length: {len(text):,} characters")
    return text


def create_char_mappings(text):
    """
    Create character-to-index and index-to-character mappings.
    
    Args:
        text (str): The preprocessed text.
    
    Returns:
        tuple: (char_to_idx dict, idx_to_char dict, vocab_size int)
    """
    # Get sorted list of unique characters
    chars = sorted(set(text))
    vocab_size = len(chars)
    
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"[INFO] Vocabulary size: {vocab_size} unique characters")
    print(f"[INFO] Characters: {''.join(chars)}")
    
    return char_to_idx, idx_to_char, vocab_size


def prepare_sequences(text, char_to_idx, seq_length=40):
    """
    Prepare input-output pairs for training.
    
    Each input is a sequence of `seq_length` characters, and the output (label)
    is the next character following the sequence.
    
    Args:
        text (str): The preprocessed text.
        char_to_idx (dict): Character to index mapping.
        seq_length (int): Length of each input sequence.
    
    Returns:
        tuple: (X input array, y output array)
    """
    print(f"[INFO] Creating sequences with length {seq_length}...")
    
    # Encode entire text as integers
    encoded = [char_to_idx[ch] for ch in text]
    
    X, y = [], []
    
    # Slide a window across the text to create input-output pairs
    for i in range(len(encoded) - seq_length):
        X.append(encoded[i : i + seq_length])       # Input: sequence of characters
        y.append(encoded[i + seq_length])            # Output: next character
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"[INFO] Total sequences: {len(X):,}")
    return X, y


def split_data(X, y, val_split=0.1):
    """
    Split data into training and validation sets.
    
    Args:
        X: Input sequences.
        y: Output labels.
        val_split (float): Fraction of data for validation.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    split_idx = int(len(X) * (1 - val_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"[INFO] Training samples: {len(X_train):,}")
    print(f"[INFO] Validation samples: {len(X_val):,}")
    
    return X_train, y_train, X_val, y_val


# ============================================================================
# 2. MODEL DESIGN
# ============================================================================

def build_model(vocab_size, seq_length, embedding_dim=256, lstm_units=256,
                num_lstm_layers=2, dropout_rate=0.2):
    """
    Build an LSTM-based text generation model.
    
    Architecture:
        Embedding → LSTM (× num_layers) → Dropout → Dense (softmax)
    
    Args:
        vocab_size (int): Number of unique characters (output classes).
        seq_length (int): Length of input sequences.
        embedding_dim (int): Dimension of the embedding layer.
        lstm_units (int or list): Number of units in LSTM layers.
        num_lstm_layers (int): Number of stacked LSTM layers.
        dropout_rate (float): Dropout rate between layers.
    
    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    # Allow passing a list of units for each layer, or a single int for all layers
    if isinstance(lstm_units, int):
        lstm_units = [lstm_units] * num_lstm_layers
    
    model = Sequential(name="LSTM_TextGenerator")
    
    # Embedding layer: maps character indices to dense vectors
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                        input_length=seq_length, name="embedding"))
    
    # Stack LSTM layers
    for i, units in enumerate(lstm_units):
        # return_sequences=True for all layers except the last LSTM
        return_seq = (i < len(lstm_units) - 1)
        model.add(LSTM(units, return_sequences=return_seq,
                       name=f"lstm_{i+1}"))
        model.add(Dropout(dropout_rate, name=f"dropout_{i+1}"))
    
    # Dense output layer with softmax for character prediction
    model.add(Dense(vocab_size, activation="softmax", name="output"))
    
    # Compile with categorical crossentropy and Adam optimizer
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    model.summary()
    return model


# ============================================================================
# 3. MODEL TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, batch_size=128, model_dir="models"):
    """
    Train the LSTM model with early stopping and model checkpointing.
    
    Args:
        model: The compiled Keras model.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        model_dir (str): Directory to save model checkpoints.
    
    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_model.keras")
    
    # Callbacks to prevent overfitting and save best model
    callbacks = [
        # Stop training if validation loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Save the model with the best validation loss
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n[INFO] Training complete.")
    print(f"[INFO] Best model saved to: {checkpoint_path}")
    
    return history


# ============================================================================
# 4. TEXT GENERATION
# ============================================================================

def sample_with_temperature(predictions, temperature=0.5):
    """
    Sample a character index from the prediction probabilities with temperature.
    
    Temperature controls randomness:
        - Low (e.g., 0.2): More conservative/deterministic predictions
        - Medium (e.g., 0.5): Balanced predictions
        - High (e.g., 1.0): More creative/random predictions
    
    Args:
        predictions (np.array): Probability distribution over characters.
        temperature (float): Sampling temperature.
    
    Returns:
        int: Sampled character index.
    """
    predictions = np.asarray(predictions).astype("float64")
    
    # Apply temperature scaling
    log_preds = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(log_preds)
    probabilities = exp_preds / np.sum(exp_preds)
    
    # Sample from the distribution
    return np.random.choice(len(probabilities), p=probabilities)


def generate_text(model, seed_text, char_to_idx, idx_to_char,
                  seq_length, length=200, temperature=0.5):
    """
    Generate new text by feeding a seed sequence to the trained model.
    
    The model predicts the next character iteratively, appending each predicted
    character to the input to generate a sequence of desired length.
    
    Args:
        model: Trained LSTM model.
        seed_text (str): Initial text to start generation from.
        char_to_idx (dict): Character to index mapping.
        idx_to_char (dict): Index to character mapping.
        seq_length (int): Expected input sequence length for the model.
        length (int): Number of characters to generate.
        temperature (float): Sampling temperature.
    
    Returns:
        str: The generated text (seed + newly generated characters).
    """
    # Ensure seed text is long enough; pad with spaces if needed
    if len(seed_text) < seq_length:
        seed_text = " " * (seq_length - len(seed_text)) + seed_text
    
    # Use only the last seq_length characters as the initial input
    current_input = seed_text[-seq_length:]
    generated = list(seed_text)
    
    for _ in range(length):
        # Encode current input sequence
        encoded = [char_to_idx.get(ch, 0) for ch in current_input]
        encoded = np.array(encoded).reshape(1, seq_length)
        
        # Predict next character probabilities
        predictions = model.predict(encoded, verbose=0)[0]
        
        # Sample next character using temperature
        next_idx = sample_with_temperature(predictions, temperature)
        next_char = idx_to_char.get(next_idx, " ")
        
        # Append predicted character and slide the window
        generated.append(next_char)
        current_input = current_input[1:] + next_char
    
    return "".join(generated)


# ============================================================================
# 5. ARCHITECTURE EXPERIMENTS (BONUS)
# ============================================================================

def run_experiments(text, char_to_idx, idx_to_char, vocab_size, seq_length,
                    X_train, y_train, X_val, y_val):
    """
    Experiment with different model architectures and compare their performance.
    
    Tests four configurations:
        1. Shallow: 1 LSTM layer, 128 units
        2. Base: 2 LSTM layers, 256 units (default)
        3. Deep: 3 LSTM layers, 256→256→128 units
        4. Wide: 2 LSTM layers, 512 units
    
    Args:
        text: Preprocessed text for seed extraction.
        char_to_idx, idx_to_char: Character mappings.
        vocab_size: Size of character vocabulary.
        seq_length: Input sequence length.
        X_train, y_train, X_val, y_val: Training and validation data.
    
    Returns:
        dict: Results for each architecture configuration.
    """
    # Define different architectures to test
    architectures = {
        "Shallow (1 LSTM, 128 units)": {
            "lstm_units": [128],
            "num_lstm_layers": 1,
            "embedding_dim": 128,
        },
        "Base (2 LSTM, 256 units)": {
            "lstm_units": [256, 256],
            "num_lstm_layers": 2,
            "embedding_dim": 256,
        },
        "Deep (3 LSTM, 256-256-128 units)": {
            "lstm_units": [256, 256, 128],
            "num_lstm_layers": 3,
            "embedding_dim": 256,
        },
        "Wide (2 LSTM, 512 units)": {
            "lstm_units": [512, 512],
            "num_lstm_layers": 2,
            "embedding_dim": 256,
        },
    }
    
    # Use a seed from the beginning of the text for consistent comparison
    seed_text = text[:seq_length]
    results = {}
    
    print("\n" + "=" * 60)
    print("ARCHITECTURE EXPERIMENTS")
    print("=" * 60)
    
    for name, config in architectures.items():
        print(f"\n{'─' * 60}")
        print(f"Experiment: {name}")
        print(f"{'─' * 60}")
        
        # Build model with this configuration
        model = build_model(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=config["embedding_dim"],
            lstm_units=config["lstm_units"],
            num_lstm_layers=config["num_lstm_layers"],
        )
        
        # Train for limited epochs for comparison (10 epochs each)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=128,
            verbose=1
        )
        
        # Generate sample text
        generated = generate_text(
            model, seed_text, char_to_idx, idx_to_char,
            seq_length, length=200, temperature=0.5
        )
        
        # Store results
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        final_accuracy = history.history["accuracy"][-1]
        
        results[name] = {
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
            "accuracy": final_accuracy,
            "generated_text": generated,
            "total_params": model.count_params(),
        }
        
        print(f"\n  Train Loss: {final_train_loss:.4f}")
        print(f"  Val Loss:   {final_val_loss:.4f}")
        print(f"  Accuracy:   {final_accuracy:.4f}")
        print(f"  Parameters: {model.count_params():,}")
        print(f"  Sample:     {generated[:100]}...")
        
        # Clear model to free memory
        del model
        tf.keras.backend.clear_session()
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<35} {'Params':>10} {'Train Loss':>12} {'Val Loss':>10} {'Accuracy':>10}")
    print("─" * 80)
    for name, res in results.items():
        print(f"{name:<35} {res['total_params']:>10,} {res['train_loss']:>12.4f} "
              f"{res['val_loss']:>10.4f} {res['accuracy']:>10.4f}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function that orchestrates the full pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LSTM Text Generator - Train on Shakespeare and generate text"
    )
    parser.add_argument("--mode", type=str, default="full",
                        choices=["train", "generate", "full", "experiment"],
                        help="Execution mode (default: full)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size (default: 128)")
    parser.add_argument("--seq_length", type=int, default=40,
                        help="Input sequence length (default: 40)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Seed text for generation")
    parser.add_argument("--length", type=int, default=200,
                        help="Number of characters to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature (default: 0.5)")
    parser.add_argument("--max_chars", type=int, default=200000,
                        help="Max characters from dataset to use (default: 200000)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  LSTM TEXT GENERATOR - Shakespeare Edition")
    print("=" * 60)
    print(f"  Mode:        {args.mode}")
    print(f"  Seq Length:   {args.seq_length}")
    print(f"  Temperature:  {args.temperature}")
    print("=" * 60)
    
    # ---- Step 1: Load and preprocess data ----
    filepath = download_dataset()
    text = load_and_preprocess_text(filepath, max_chars=args.max_chars)
    char_to_idx, idx_to_char, vocab_size = create_char_mappings(text)
    
    # ---- Step 2: Prepare sequences ----
    X, y = prepare_sequences(text, char_to_idx, seq_length=args.seq_length)
    X_train, y_train, X_val, y_val = split_data(X, y)
    
    model_path = os.path.join("models", "best_model.keras")
    
    # ---- Mode: Train or Full ----
    if args.mode in ("train", "full"):
        # Build the model
        model = build_model(
            vocab_size=vocab_size,
            seq_length=args.seq_length
        )
        
        # Train the model
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Print final metrics
        print(f"\n[RESULTS] Final Training Loss:   {history.history['loss'][-1]:.4f}")
        print(f"[RESULTS] Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"[RESULTS] Final Accuracy:        {history.history['accuracy'][-1]:.4f}")
    
    # ---- Mode: Generate or Full ----
    if args.mode in ("generate", "full"):
        # Load the best saved model
        if os.path.exists(model_path):
            print(f"\n[INFO] Loading best model from {model_path}...")
            model = load_model(model_path)
        elif args.mode == "generate":
            print("[ERROR] No trained model found. Run with --mode train first.")
            return
        
        # Define seed texts for generation
        if args.seed:
            seeds = [args.seed]
        else:
            seeds = [
                text[:args.seq_length],                    # Beginning of dataset
                "shall i compare thee to",                 # Famous Sonnet 18 opening
                "to be or not to be",                      # Hamlet
                "friends, romans, countrymen",             # Julius Caesar
            ]
        
        # Generate and display text for each seed
        print("\n" + "=" * 60)
        print("GENERATED TEXT SAMPLES")
        print("=" * 60)
        
        all_outputs = []
        
        for i, seed in enumerate(seeds):
            seed_clean = seed.lower()
            
            # Generate with different temperatures
            for temp in [0.2, 0.5, 1.0]:
                generated = generate_text(
                    model, seed_clean, char_to_idx, idx_to_char,
                    args.seq_length, length=args.length, temperature=temp
                )
                
                header = f"Seed: \"{seed_clean}\" | Temperature: {temp}"
                print(f"\n{'─' * 60}")
                print(header)
                print(f"{'─' * 60}")
                print(generated)
                
                all_outputs.append(f"{header}\n{'─' * 40}\n{generated}\n")
        
        # Save generated outputs to file
        with open("sample_outputs.txt", "w", encoding="utf-8") as f:
            f.write("LSTM Text Generator - Sample Outputs\n")
            f.write("=" * 50 + "\n\n")
            for output in all_outputs:
                f.write(output + "\n")
        
        print(f"\n[INFO] Sample outputs saved to: sample_outputs.txt")
    
    # ---- Mode: Experiment ----
    if args.mode == "experiment":
        results = run_experiments(
            text, char_to_idx, idx_to_char, vocab_size, args.seq_length,
            X_train, y_train, X_val, y_val
        )
        
        # Save experiment results
        with open("experiment_results.txt", "w", encoding="utf-8") as f:
            f.write("Architecture Experiment Results\n")
            f.write("=" * 50 + "\n\n")
            for name, res in results.items():
                f.write(f"Architecture: {name}\n")
                f.write(f"  Parameters:  {res['total_params']:,}\n")
                f.write(f"  Train Loss:  {res['train_loss']:.4f}\n")
                f.write(f"  Val Loss:    {res['val_loss']:.4f}\n")
                f.write(f"  Accuracy:    {res['accuracy']:.4f}\n")
                f.write(f"  Generated:   {res['generated_text'][:200]}...\n\n")
        
        print(f"\n[INFO] Experiment results saved to: experiment_results.txt")
    
    print("\n[DONE] Pipeline complete!")


if __name__ == "__main__":
    main()
