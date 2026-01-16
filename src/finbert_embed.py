"""
Generate FinBERT embeddings for LLM analysis text.

Takes the analysis_text from LLM outputs and generates 768-dim CLS embeddings.

Usage:
    python -m src.finbert_embed --config configs/config.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_llm_outputs(cache_path: str) -> dict:
    """Load LLM analysis outputs from JSONL cache."""
    outputs = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    outputs[entry["sample_id"]] = entry
    return outputs


def create_text_for_embedding(entry: dict) -> str:
    """
    Create text string for embedding from LLM analysis.

    Prioritizes analysis_text, falls back to stringified JSON.
    """
    if entry.get("analysis_text"):
        return entry["analysis_text"]

    # Fall back to stringified JSON
    if entry.get("analysis_json"):
        aj = entry["analysis_json"]
        parts = []
        if aj.get("trend"):
            parts.append(f"Trend: {aj['trend']}")
        if aj.get("momentum"):
            parts.append(f"Momentum: {aj['momentum']}")
        if aj.get("volatility"):
            parts.append(f"Volatility: {aj['volatility']}")
        if aj.get("summary"):
            parts.append(aj["summary"])
        return ". ".join(parts)

    return ""


class FinBERTEmbedder:
    """FinBERT embedding generator."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        """
        Initialize FinBERT model and tokenizer.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu/mps). Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Loading FinBERT model on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        print("FinBERT loaded successfully")

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate CLS embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for inference

        Returns:
            numpy array of shape (n_texts, 768)
        """
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch_texts = texts[i:i + batch_size]

                # Handle empty texts
                batch_texts = [t if t else "neutral market conditions" for t in batch_texts]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # Extract CLS token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)

        return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Generate FinBERT embeddings")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if not config.get("use_text_embeddings", True):
        print("Text embeddings disabled in config. Exiting.")
        return

    # Load samples
    samples_path = Path("data/samples/samples.parquet")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    samples_df = pd.read_parquet(samples_path)
    print(f"Loaded {len(samples_df)} samples")

    # Load LLM outputs
    cache_path = config.get("cache_path", "data/llm/analysis.jsonl")
    llm_outputs = load_llm_outputs(cache_path)
    print(f"Loaded {len(llm_outputs)} LLM analysis entries")

    # Prepare texts for embedding
    sample_ids = samples_df["sample_id"].tolist()
    texts = []
    missing_count = 0

    for sample_id in sample_ids:
        if sample_id in llm_outputs:
            text = create_text_for_embedding(llm_outputs[sample_id])
            texts.append(text)
        else:
            # Use placeholder for missing LLM outputs
            texts.append("")
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} samples missing LLM analysis")

    # Initialize embedder
    model_name = config.get("finbert_model", "ProsusAI/finbert")
    embedder = FinBERTEmbedder(model_name)

    # Generate embeddings
    print(f"\nGenerating embeddings for {len(texts)} samples...")
    embeddings = embedder.embed(texts, batch_size=args.batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as .npy
    embeddings_path = output_dir / "finbert.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Save sample_id to index mapping
    mapping = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    mapping_path = output_dir / "sample_id_to_idx.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    print(f"Saved mapping to {mapping_path}")

    # Print summary statistics
    print("\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")


if __name__ == "__main__":
    main()
