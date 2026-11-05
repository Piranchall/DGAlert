"""How easily one hyphen hides a DGA domain from the current model.

Takes N_DOMAINS test_seen DGA domains (seed 42) that the model in models/latest.json labels dga,
inserts one hyphen in the middle of each second-level label (e.g. abcdef.com -> abc-def.com), and
prints the share still labelled dga. Also counts the training DGA rows whose second-level label
contains a hyphen, which explains the result.

Needs data/processed/ (python -m src.prepare_data) and a trained model (python -m src.train).
Run with: python -m scripts.hyphen_evasion
"""

import pandas as pd

from src import config
from src.features import FEATURE_NAMES, feature_frame, public_suffix
from src.model_loader import load_model

N_DOMAINS = 2000


def insert_hyphen(label: str) -> str:
    """Put one hyphen in the middle of a label: 'abcdef' -> 'abc-def', 'abcde' -> 'ab-cde'."""
    middle = len(label) // 2
    return f"{label[:middle]}-{label[middle:]}"


def hyphenated_domain(sld: str, domain: str) -> str:
    """The domain rebuilt as <sld with a hyphen>.<public suffix>."""
    return f"{insert_hyphen(sld)}.{public_suffix(domain)}"


def p_dga(model, domains: list[str]) -> pd.Series:
    """P(dga) for each domain, with features built exactly as the API builds them."""
    features = pd.concat([feature_frame(d) for d in domains], ignore_index=True)
    return pd.Series(model.predict_proba(features[FEATURE_NAMES])[:, 1])


def main() -> None:
    model, metadata = load_model()
    print(
        f"model {metadata['version']} ({metadata['model_type']}), threshold {config.DGA_THRESHOLD}"
    )

    test_seen = pd.read_parquet(config.TEST_SEEN_PATH)
    dga = test_seen[test_seen["label"] == "dga"]
    detected = dga[model.predict_proba(dga[FEATURE_NAMES])[:, 1] >= config.DGA_THRESHOLD]
    sample = detected.sample(N_DOMAINS, random_state=config.SEED)

    hyphenated = [
        hyphenated_domain(s, d) for s, d in zip(sample["sld"], sample["domain"], strict=True)
    ]
    still_dga = (p_dga(model, hyphenated) >= config.DGA_THRESHOLD).mean()
    print(f"test_seen DGA domains labelled dga: {len(detected)} of {len(dga)}; sampled {N_DOMAINS}")
    print(f"example: {sample['domain'].iloc[0]} -> {hyphenated[0]}")
    print(f"still labelled dga after inserting one hyphen: {still_dga:.4f}")

    train = pd.read_parquet(config.TRAIN_PATH, columns=["sld", "label"])
    train_dga = train[train["label"] == "dga"]
    with_hyphen = int(train_dga["sld"].str.contains("-", regex=False).sum())
    print(f"training DGA rows with a hyphen in the label: {with_hyphen} of {len(train_dga)}")


if __name__ == "__main__":
    main()
