"""Create deterministic synthetic rows for local pipeline experiments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_dataset(rows: int = 800, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=rows)
    charges = np.clip(rng.normal(68, 22, size=rows), 15, 140).round(2)
    tickets = rng.poisson(1.2, size=rows)
    contract = rng.choice(["month_to_month", "one_year", "two_year"], rows, p=[0.58, 0.25, 0.17])
    payment = rng.choice(["electronic", "bank_transfer", "card"], rows)
    logits = -1.25 + (contract == "month_to_month") * 1.05 - (contract == "two_year") * 0.85
    logits = logits + (tenure < 10) * 0.75 + (charges > 90) * 0.45 + tickets * 0.24
    probability = 1 / (1 + np.exp(-logits))
    churned = rng.binomial(1, probability)
    return pd.DataFrame({"tenure_months": tenure, "monthly_charges": charges,
                         "support_tickets_90d": tickets, "contract": contract,
                         "payment_method": payment, "churned": churned})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/customers.csv"))
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_dataset(args.rows, args.seed).to_csv(args.out, index=False)
    print(f"Wrote {args.rows} synthetic rows to {args.out}")
