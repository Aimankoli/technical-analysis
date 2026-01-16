"""
Generate LLM technical analysis for each sample window.

Uses ONLY past data (indicators within the window, no future information).

Supports providers:
- local: Ollama HTTP API
- openai: OpenAI API
- gemini: Google Gemini API (free tier available)

Usage:
    python -m src.llm_analyze --config configs/config.yaml
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


SYSTEM_PROMPT = """You are a professional financial analyst. Use only the provided indicators; do not assume any other info. Do not reference specific ticker names, dates, or any information not in the provided data."""

USER_PROMPT_TEMPLATE = """Here is a JSON object with technical indicators computed from the last {lookback_days} trading days of a stock. Analyze the latest day's market conditions.

Indicators:
{indicators_json}

Return STRICT JSON only with this exact schema:
{{
  "trend": "bullish" | "bearish" | "sideways",
  "momentum": "strong" | "weak" | "neutral",
  "volatility": "high" | "normal" | "low",
  "support_resistance": {{"support": <float or null>, "resistance": <float or null>}},
  "summary": "<concise analysis in 80 words or less>"
}}

Return only the JSON object, no other text."""


def compute_hash(text: str) -> str:
    """Compute MD5 hash of text for caching."""
    return hashlib.md5(text.encode()).hexdigest()


def load_cache(cache_path: str) -> dict:
    """Load cached LLM responses from JSONL file."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cache[entry["sample_id"]] = entry
    return cache


def save_cache_entry(cache_path: str, entry: dict) -> None:
    """Append a single entry to the cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def prepare_indicators_json(
    window_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Prepare indicators JSON for the LLM.

    Includes last day values and window summary statistics.
    """
    last_row = window_df.iloc[-1]

    # Extract last day indicators (round for cleaner output)
    def safe_round(val, decimals=4):
        if pd.isna(val) or np.isinf(val):
            return None
        return round(float(val), decimals)

    indicators = {
        "window_size": len(window_df),
        "last_day": {
            "close": safe_round(last_row["Close"], 2),
            "sma20": safe_round(last_row.get("sma20"), 2),
            "bb_upper": safe_round(last_row.get("bb_upper"), 2),
            "bb_lower": safe_round(last_row.get("bb_lower"), 2),
            "rsi14": safe_round(last_row.get("rsi14"), 2),
            "bb_percent_b": safe_round(last_row.get("bb_percent_b"), 4),
            "distance_to_sma20": safe_round(last_row.get("distance_to_sma20"), 4),
            "return_1d": safe_round(last_row.get("return_1d"), 4),
            "return_5d": safe_round(last_row.get("return_5d"), 4),
            "volume_zscore": safe_round(last_row.get("volume_zscore"), 2),
        },
        "trend_slope_10d": safe_round(last_row.get("trend_slope_10d"), 6),
        "volatility_20d": safe_round(last_row.get("volatility_20d"), 4),
        "window_stats": {
            "price_high": safe_round(window_df["High"].max(), 2),
            "price_low": safe_round(window_df["Low"].min(), 2),
            "avg_volume": safe_round(window_df["Volume"].mean(), 0),
            "rsi_avg": safe_round(window_df["rsi14"].mean(), 2),
        }
    }

    return indicators


def call_ollama(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 220
) -> Optional[str]:
    """Call Ollama HTTP API for local LLM inference."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.RequestException as e:
        print(f"Ollama API error: {e}")
        return None


def call_openai(
    prompt: str,
    system_prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    max_tokens: int = 220
) -> Optional[str]:
    """Call OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"OpenAI API error: {e}")
        return None


def call_gemini(
    prompt: str,
    system_prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0,
    max_tokens: int = 220
) -> Optional[str]:
    """
    Call Google Gemini API.

    Free tier: 15 RPM (requests per minute) for gemini-1.5-flash
    Set GOOGLE_API_KEY environment variable.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Get one at https://aistudio.google.com/app/apikey")

    # Combine system prompt and user prompt for Gemini
    full_prompt = f"{system_prompt}\n\n{prompt}"

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json"
                }
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        # Extract text from Gemini response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0].get("text", "")

        return None
    except requests.RequestException as e:
        print(f"Gemini API error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def parse_llm_response(response: str) -> tuple[Optional[dict], str]:
    """
    Parse LLM response into structured JSON.

    Returns:
        Tuple of (parsed_json, analysis_text)
    """
    if not response:
        return None, ""

    # Try to extract JSON from response
    text = response.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're code block markers
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)

        # Validate required fields
        required_fields = ["trend", "momentum", "volatility", "support_resistance", "summary"]
        for field in required_fields:
            if field not in parsed:
                parsed[field] = None

        # Create analysis text from JSON
        analysis_text = f"Trend: {parsed.get('trend', 'unknown')}. "
        analysis_text += f"Momentum: {parsed.get('momentum', 'unknown')}. "
        analysis_text += f"Volatility: {parsed.get('volatility', 'unknown')}. "
        if parsed.get("summary"):
            analysis_text += parsed["summary"]

        return parsed, analysis_text

    except json.JSONDecodeError:
        # If parsing fails, return None but keep the raw text
        return None, text


def analyze_sample(
    sample_id: int,
    window_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Generate LLM analysis for a single sample.

    Returns dict with sample_id, prompt_hash, analysis_json, analysis_text
    """
    # Prepare indicators
    indicators = prepare_indicators_json(window_df, config)
    indicators_json_str = json.dumps(indicators, indent=2)

    # Build prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        lookback_days=config["lookback_days"],
        indicators_json=indicators_json_str
    )

    prompt_hash = compute_hash(SYSTEM_PROMPT + user_prompt)

    # Call LLM based on provider
    provider = config.get("llm_provider", "local")
    model = config.get("llm_model", "llama3.1:8b-instruct")
    temperature = config.get("temperature", 0)
    max_tokens = config.get("max_tokens", 220)

    if provider == "local":
        response = call_ollama(user_prompt, SYSTEM_PROMPT, model, temperature, max_tokens)
    elif provider == "openai":
        response = call_openai(user_prompt, SYSTEM_PROMPT, model, temperature, max_tokens)
    elif provider == "gemini":
        response = call_gemini(user_prompt, SYSTEM_PROMPT, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # Parse response
    analysis_json, analysis_text = parse_llm_response(response)

    return {
        "sample_id": sample_id,
        "prompt_hash": prompt_hash,
        "analysis_json": analysis_json,
        "analysis_text": analysis_text,
        "raw_response": response
    }


def main():
    parser = argparse.ArgumentParser(description="Generate LLM technical analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    ticker = config["ticker"]

    if not config.get("llm_enabled", True):
        print("LLM analysis is disabled in config. Exiting.")
        return

    # Load data with indicators
    data_path = Path("data/raw") / f"{ticker}_indicators.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Indicators file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["Date"])

    # Load samples
    samples_path = Path("data/samples/samples.parquet")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    samples_df = pd.read_parquet(samples_path)
    print(f"Loaded {len(samples_df)} samples")

    # Load cache
    cache_path = config.get("cache_path", "data/llm/analysis.jsonl")
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached entries")

    # Process samples
    if args.limit:
        samples_df = samples_df.head(args.limit)

    new_count = 0
    cached_count = 0
    error_count = 0
    rate_limit_delay = config.get("llm_rate_limit_delay", 4)  # seconds between calls

    for _, sample in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Analyzing"):
        sample_id = sample["sample_id"]

        # Check cache
        if sample_id in cache:
            cached_count += 1
            continue

        # Extract window
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]
        window_df = df.iloc[start_idx:end_idx + 1].copy()

        # Analyze
        result = analyze_sample(sample_id, window_df, config)

        # Check if we got a valid response
        if result.get("raw_response") is None:
            error_count += 1
            print(f"\nWarning: No response for sample {sample_id}")
        else:
            # Save to cache only if successful
            save_cache_entry(cache_path, result)
            new_count += 1

        # Rate limiting for API calls (respect free tier limits)
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    print(f"\nDone! Processed {new_count} new samples, {cached_count} from cache, {error_count} errors")
    print(f"Results saved to {cache_path}")


if __name__ == "__main__":
    main()
