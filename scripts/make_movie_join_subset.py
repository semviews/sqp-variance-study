#!/usr/bin/env python3
"""Inputs for Movie Q5-Q7 with fewer reviews of the joined film.

SemBench's sampler keeps all 256 reviews of the film that Q5-Q7 self-join at
every scale factor, so a smaller scale factor does not shrink the join. This
script copies SemBench's sf_2000 inputs and keeps a seeded sample of distinct
reviews of that film; every other row is unchanged. The gold SQL is then evaluated
on the same files, so ground truth matches the reduced input.

Run with SemBench/.venvs/sembench/bin/python.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SemBench" / "files" / "movie" / "data" / "sf_2000"
FILM = "ant_man_and_the_wasp_quantumania"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    target = ROOT / "experiments" / "data" / f"movie-join{args.reviews}"
    target.mkdir(parents=True, exist_ok=True)
    reviews = pd.read_csv(SOURCE / "Reviews.csv")
    # SemBench stores each of this film's 128 reviews twice; keep one copy.
    film = reviews[reviews["id"] == FILM].drop_duplicates()
    film = film.sample(n=args.reviews, random_state=args.seed)
    kept = reviews[(reviews["id"] != FILM) | reviews.index.isin(film.index)]
    kept.to_csv(target / "Reviews.csv", index=False)
    pd.read_csv(SOURCE / "Movies.csv").to_csv(target / "Movies.csv", index=False)
    counts = film["scoreSentiment"].value_counts().to_dict()
    print(f"{target.relative_to(ROOT)}: {len(kept)} reviews, {args.reviews} of {FILM} {counts}, "
          f"{args.reviews * args.reviews} ordered pairs per join")


if __name__ == "__main__":
    main()
