"""
Part 1: Deterministic candidate pair generation.
Generates high-recall duplicate candidates using simple heuristics.
"""

import re
import itertools
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

ACCOUNT_FUZZY_THRESHOLD = 88   # rapidfuzz 0-100 scale
SUBJECT_TOKEN_MIN_OVERLAP = 2  # at least 2 shared non-trivial tokens

_STOPWORDS = {
    "the", "a", "an", "in", "on", "for", "to", "of", "and", "or", "is",
    "are", "with", "our", "we", "not", "at", "this", "that",
}


def normalize_account(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_email(email) -> str:
    if pd.isna(email):
        return ""
    return str(email).strip().lower()


def subject_tokens(subject: str) -> set:
    words = re.findall(r"[a-z0-9]+", str(subject).lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


@dataclass
class CandidatePair:
    case_id_a: str
    case_id_b: str
    reasons: list = field(default_factory=list)   # which heuristic(s) fired
    account_fuzzy_score: float = 0.0
    same_email: bool = False
    shared_subject_tokens: int = 0
    time_gap_hours: float = None


def generate_candidate_pairs(df: pd.DataFrame) -> list[CandidatePair]:
    df = df.copy()
    # Note: itertuples() can't expose leading-underscore columns as attributes
    # (namedtuple restriction), so these are named without a leading underscore.
    df["norm_account"] = df["account_name"].apply(normalize_account)
    df["norm_email"] = df["contact_email"].apply(normalize_email)
    df["subj_tokens"] = df["subject"].apply(subject_tokens)
    df["created_at"] = pd.to_datetime(df["created_at"])

    pairs: dict[tuple, CandidatePair] = {}

    def get_or_create(id_a, id_b):
        key = tuple(sorted((id_a, id_b)))
        if key not in pairs:
            pairs[key] = CandidatePair(case_id_a=key[0], case_id_b=key[1])
        return pairs[key]

    # --- Heuristic 1: fuzzy account match ---
    # Group by exact-normalized account first (cheap), then fuzzy-compare
    # across distinct normalized names to catch typos/casing.
    unique_accounts = df["norm_account"].unique().tolist()
    account_clusters = {}  # normalized name -> cluster representative
    for acc in unique_accounts:
        placed = False
        for rep in account_clusters:
            if fuzz.token_sort_ratio(acc, rep) >= ACCOUNT_FUZZY_THRESHOLD:
                account_clusters[rep].append(acc)
                placed = True
                break
        if not placed:
            account_clusters[acc] = [acc]

    for rep, members in account_clusters.items():
        if len(members) == 1 and (df["norm_account"] == rep).sum() < 2:
            continue  # no fuzzy variants AND fewer than 2 cases -> skip
        sub = df[df["norm_account"].isin(members)]
        for (_, row_a), (_, row_b) in itertools.combinations(sub.iterrows(), 2):
            p = get_or_create(row_a.case_id, row_b.case_id)
            score = fuzz.token_sort_ratio(row_a["norm_account"], row_b["norm_account"])
            p.account_fuzzy_score = max(p.account_fuzzy_score, score)
            if "same_account" not in p.reasons:
                p.reasons.append("same_account")

    # --- Heuristic 2: same contact email ---
    email_groups = df[df["norm_email"] != ""].groupby("norm_email")
    for email, group in email_groups:
        if len(group) < 2:
            continue
        for (_, row_a), (_, row_b) in itertools.combinations(group.iterrows(), 2):
            p = get_or_create(row_a.case_id, row_b.case_id)
            p.same_email = True
            if "same_email" not in p.reasons:
                p.reasons.append("same_email")

    # --- Heuristic 3: overlapping subject tokens (weak, cross-account) ---
    # O(n^2) over 269 rows is ~36k comparisons -- trivial cost, fine for
    # this scale. Would need blocking for a much larger dataset.
    rows = list(df.itertuples())
    for row_a, row_b in itertools.combinations(rows, 2):
        overlap = row_a.subj_tokens & row_b.subj_tokens
        if len(overlap) >= SUBJECT_TOKEN_MIN_OVERLAP:
            p = get_or_create(row_a.case_id, row_b.case_id)
            p.shared_subject_tokens = len(overlap)
            if "subject_overlap" not in p.reasons:
                p.reasons.append("subject_overlap")

    # --- Attach time-gap metadata (not a filter) ---
    lookup = df.set_index("case_id")["created_at"]
    for p in pairs.values():
        t_a, t_b = lookup[p.case_id_a], lookup[p.case_id_b]
        p.time_gap_hours = abs((t_a - t_b).total_seconds()) / 3600

    return list(pairs.values())


if __name__ == "__main__":
    df = pd.read_csv("support_cases_1.csv")  # adjust path as needed
    candidates = generate_candidate_pairs(df)
    print(f"Total rows: {len(df)}")
    print(f"Candidate pairs generated: {len(candidates)}")

    by_reason = {}
    for p in candidates:
        for r in p.reasons:
            by_reason[r] = by_reason.get(r, 0) + 1
    print("Breakdown by heuristic (a pair can count in multiple):", by_reason)

    print("\nSample: multi-signal pairs (highest confidence)")
    multi = [p for p in candidates if len(p.reasons) >= 2]
    for p in sorted(multi, key=lambda x: x.time_gap_hours)[:5]:
        print(f"  {p.case_id_a} <-> {p.case_id_b} | reasons={p.reasons} "
              f"| gap={p.time_gap_hours:.1f}h | acct_score={p.account_fuzzy_score:.0f}")