#!/usr/bin/env python3
"""
Emotion classification inference script.
Uses quantized ONNX model and tokenizer from ./onnxmodel directory.
"""

import os
import sys
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from transformers import AutoTokenizer, BertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tokenizers
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class EmotionClassifier:
    """Emotion classification using ONNX model."""

    def __init__(self, model_dir: str = "./onnxmodel"):
        """
        Initialize emotion classifier.

        Args:
            model_dir: Directory containing model and tokenizer files
        """
        self.model_dir = Path(model_dir)

        # Load labels
        labels_path = self.model_dir / "emotion_model_labels.json"
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        self.id2label = labels_data["id2label"]
        self.label2id = labels_data["label2id"]
        self.num_labels = len(self.id2label)

        # Load model
        model_path = self._find_model()
        print(f"Loading model from: {model_path}")
        self.session = ort.InferenceSession(str(model_path))

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Get model info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"Model inputs: {self.input_names}")
        print(f"Model outputs: {self.output_names}")
        print(f"Number of emotion classes: {self.num_labels}")

    def _find_model(self) -> Path:
        """Find the ONNX model file."""
        # Look for quantized model first
        quantized = self.model_dir / "emotion_model_optimized_quantized.onnx"
        if quantized.exists():
            return quantized

        # Look for any .onnx file
        onnx_files = list(self.model_dir.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]

        raise FileNotFoundError(f"No ONNX model found in {self.model_dir}")

    def _load_tokenizer(self):
        """Load tokenizer from files."""
        tokenizer_path = self.model_dir

        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to load as BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
                print("Loaded tokenizer using transformers.BertTokenizer")
                return tokenizer
            except Exception as e:
                print(f"Failed to load tokenizer with transformers: {e}")

        # Fallback: try tokenizers library
        if TOKENIZERS_AVAILABLE:
            try:
                from tokenizers import Tokenizer
                tokenizer_path_json = tokenizer_path / "tokenizer.json"
                if tokenizer_path_json.exists():
                    tokenizer = Tokenizer.from_file(str(tokenizer_path_json))
                    print("Loaded tokenizer using tokenizers library")
                    return tokenizer
            except Exception as e:
                print(f"Failed to load tokenizer with tokenizers library: {e}")

        # Last resort: simple whitespace tokenizer
        print("Warning: Using simple whitespace tokenizer as fallback")
        print("Install transformers: pip install transformers")
        return SimpleTokenizer(str(tokenizer_path / "vocab.txt"))

    def preprocess(self, text: str, max_length: int = 512) -> Dict[str, np.ndarray]:
        """
        Preprocess text into model inputs.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Dictionary with input tensors
        """
        if hasattr(self.tokenizer, 'encode_plus'):
            # Transformers tokenizer
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            input_ids = encoding['input_ids'].astype(np.int64)
            attention_mask = encoding['attention_mask'].astype(np.int64)
        elif hasattr(self.tokenizer, 'encode'):
            # Tokenizers library
            encoding = self.tokenizer.encode(text)
            input_ids = encoding.ids
            attention_mask = encoding.attention_mask

            # Pad/truncate
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            else:
                pad_length = max_length - len(input_ids)
                input_ids = input_ids + [0] * pad_length
                attention_mask = attention_mask + [0] * pad_length

            input_ids = np.array([input_ids], dtype=np.int64)
            attention_mask = np.array([attention_mask], dtype=np.int64)
        else:
            # Simple tokenizer
            input_ids, attention_mask = self.tokenizer.encode(text, max_length)
            input_ids = np.array([input_ids], dtype=np.int64)
            attention_mask = np.array([attention_mask], dtype=np.int64)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def predict(self, text: str) -> Dict:
        """
        Predict emotion from text.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        inputs = self.preprocess(text)

        # Run inference
        ort_inputs = {name: inputs[name] for name in self.input_names}
        ort_outputs = self.session.run(None, ort_inputs)

        # Get logits (first output)
        logits = ort_outputs[0]
        probabilities = self._softmax(logits[0])
        predicted_idx = int(np.argmax(probabilities))
        predicted_emotion = self.id2label[str(predicted_idx)]

        # Get top-k predictions
        top_k = 3
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_emotions = [
            {
                'emotion': self.id2label[str(idx)],
                'confidence': float(probabilities[idx]),
                'id': int(idx)
            }
            for idx in top_indices
        ]

        return {
            'text': text,
            'predicted_emotion': predicted_emotion,
            'confidence': float(probabilities[predicted_idx]),
            'all_probabilities': probabilities.tolist(),
            'top_emotions': top_emotions
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for array."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Predict emotions for multiple texts."""
        return [self.predict(text) for text in texts]

    def print_prediction(self, result: Dict):
        """Print prediction results."""
        print(f"\nText: {result['text']}")
        print(f"Predicted emotion: {result['predicted_emotion']} (confidence: {result['confidence']:.4f})")
        print("Top emotions:")
        for i, emotion in enumerate(result['top_emotions']):
            print(f"  {i+1}. {emotion['emotion']}: {emotion['confidence']:.4f}")


class SimpleTokenizer:
    """Simple tokenizer fallback using vocab.txt."""

    def __init__(self, vocab_path: str):
        self.vocab = {}
        self.id_to_token = {}

        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    token = line.strip()
                    self.vocab[token] = idx
                    self.id_to_token[idx] = token
        else:
            # Create basic vocab
            basic_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            for idx, token in enumerate(basic_tokens):
                self.vocab[token] = idx
                self.id_to_token[idx] = token

        self.unk_token_id = self.vocab.get("[UNK]", 100)
        self.cls_token_id = self.vocab.get("[CLS]", 101)
        self.sep_token_id = self.vocab.get("[SEP]", 102)
        self.pad_token_id = self.vocab.get("[PAD]", 0)

    def encode(self, text: str, max_length: int = 512) -> Tuple[List[int], List[int]]:
        """Simple tokenization by splitting on whitespace."""
        # Convert to lowercase (assuming do_lower_case=True)
        text = text.lower()

        # Split into tokens
        tokens = text.split()

        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)

        # Add special tokens: [CLS] tokens [SEP]
        input_ids = [self.cls_token_id] + token_ids[:max_length-2] + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        if len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            # Ensure last token is [SEP]
            input_ids[-1] = self.sep_token_id

        return input_ids, attention_mask


def main():
    """Main inference function."""
    import argparse

    # Declare globals
    global TRANSFORMERS_AVAILABLE, TOKENIZERS_AVAILABLE

    parser = argparse.ArgumentParser(description="Emotion classification inference")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="File containing texts (one per line)")
    parser.add_argument("--model-dir", type=str, default="./onnxmodel",
                       help="Directory containing model files")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install required dependencies")

    args = parser.parse_args()

    # Check dependencies
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: transformers library not installed.")
        print("Tokenization may be less accurate.")
        print("Install with: pip install transformers")

    # Install dependencies if requested
    if args.install_deps:
        print("Installing dependencies...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                  "transformers", "tokenizers", "onnxruntime"])
            print("Dependencies installed successfully.")
            # Update availability flags
            TRANSFORMERS_AVAILABLE = True
            TOKENIZERS_AVAILABLE = True
        except Exception as e:
            print(f"Failed to install dependencies: {e}")

    # Initialize classifier
    try:
        classifier = EmotionClassifier(args.model_dir)
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        print(f"Make sure model files exist in {args.model_dir}")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        print("\n" + "="*50)
        print("Emotion Classification Interactive Mode")
        print("Enter text to classify (or 'quit' to exit)")
        print("="*50)

        while True:
            try:
                text = input("\nEnter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue

                result = classifier.predict(text)
                classifier.print_prediction(result)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

        return

    # Single text mode
    if args.text:
        result = classifier.predict(args.text)
        classifier.print_prediction(result)
        return

    # File mode
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)

        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(texts)} texts from {args.file}")
        for i, text in enumerate(texts):
            print(f"\n[{i+1}/{len(texts)}]")
            result = classifier.predict(text)
            classifier.print_prediction(result)

        return

    # No input provided, show help
    parser.print_help()
    print("\nExamples:")
    print("  python inference.py --text \"我今天很高兴\"")
    print("  python inference.py --file texts.txt")
    print("  python inference.py --interactive")
    print("  python inference.py --install-deps --interactive")


if __name__ == "__main__":
    main()