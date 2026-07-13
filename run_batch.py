import os

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

import storage
from tools import CaseDataStore
from candidate_pairs import generate_candidate_pairs
from agent import run_investigation

load_dotenv()

print("=" * 60)
print("Loading support cases...")
print("=" * 60)

df = pd.read_csv("support_cases_1.csv")

store = CaseDataStore(df)

print(f"Loaded {len(df)} support cases.")

print("\nGenerating candidate pairs...")

pairs = generate_candidate_pairs(df)

print(f"Generated {len(pairs)} candidate pairs.")

# Assignment only requires at least 10 investigations.
pairs = [p for p in pairs if len(p.reasons) >= 2][:10]

print(f"\nRunning investigations on {len(pairs)} candidate pairs...\n")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

for index, pair in enumerate(pairs, start=1):

    print(f"[{index}/{len(pairs)}] {pair.case_id_a} <-> {pair.case_id_b}")

    investigation = run_investigation(
        pair.case_id_a,
        pair.case_id_b,
        pair.reasons,
        store,
        client,
    )

    storage.save_investigation(investigation)

    print(
        f"   Verdict : {investigation.verdict['label']}"
    )
    print(
        f"   Confidence : {investigation.verdict['confidence']:.2f}"
    )
    print(
        f"   Trace Steps : {len(investigation.trace)}"
    )
    print()

print("=" * 60)
print("Finished.")
print("Investigations saved to investigations.json")
print("=" * 60)