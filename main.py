from dataset import create_dataset, DEFAULT_SYMBOLS


def main():
    """Generate the technical analysis dataset."""
    print("Technical Analysis Dataset Generator")
    print("=" * 40)

    df = create_dataset(
        symbols=DEFAULT_SYMBOLS,
        start_date="2015-01-01",
        horizon=5,
        threshold=0.0,
    )

    print(f"\nDataset ready in ./data/ directory")
    print(f"Features: {len([c for c in df.columns if c not in ['date', 'symbol', 'label', 'label_multiclass', 'future_return']])}")


if __name__ == "__main__":
    main()
