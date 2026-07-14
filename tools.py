"""
Part 2: Deterministic investigation tools used by the AI agent.
Each tool operates only on the CRM dataset.
"""

import pandas as pd
from rapidfuzz import fuzz

from sanitize import wrap_untrusted, clean_field

FUZZY_FIELDS = {"account_name", "contact_name", "subject", "description"}


class CaseDataStore:
    """Thin wrapper so tools can look up case rows by case_id."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.set_index("case_id", drop=False)

    def get(self, case_id: str) -> dict:
        if case_id not in self.df.index:
            raise KeyError(f"Unknown case_id: {case_id}")
        return self.df.loc[case_id].to_dict()

    # ---- Tool 1 ----
    def compare_fields(self, case_id_a: str, case_id_b: str) -> dict:
        """Structured, exact/near-exact comparison of non-free-text fields."""
        a, b = self.get(case_id_a), self.get(case_id_b)
        same_channel = a["channel"] == b["channel"]
        same_priority = a["priority"] == b["priority"]
        same_status = a["status"] == b["status"]
        email_a = str(a.get("contact_email") or "").strip().lower()
        email_b = str(b.get("contact_email") or "").strip().lower()
        same_email_exact = bool(email_a) and email_a == email_b
        return {
            "case_id_a": case_id_a,
            "case_id_b": case_id_b,
            "same_channel": same_channel,
            "channel_a": a["channel"],
            "channel_b": b["channel"],
            "same_priority": same_priority,
            "same_status": same_status,
            "same_contact_email_exact": same_email_exact,
            "contact_name_a": clean_field(str(a["contact_name"])),
            "contact_name_b": clean_field(str(b["contact_name"])),
            "account_name": clean_field(str(a["account_name"])),
            "contact_email": email_a,
        }

    # ---- Tool 2 ----
    def fuzzy_score(self, case_id_a: str, case_id_b: str, field: str) -> dict:
        """Fuzzy text-similarity score (0-100) for one free-text/name field."""
        if field not in FUZZY_FIELDS:
            return {"error": f"field must be one of {sorted(FUZZY_FIELDS)}"}
        a, b = self.get(case_id_a), self.get(case_id_b)
        val_a = clean_field(str(a.get(field, "")))
        val_b = clean_field(str(b.get(field, "")))
        score = fuzz.token_sort_ratio(val_a.lower(), val_b.lower())
        result = {
            "case_id_a": case_id_a,
            "case_id_b": case_id_b,
            "field": field,
            "similarity_score": round(score, 1),
        }
        # Only echo the actual text back for longer free-text fields, where
        # the model plausibly needs the content itself (not just the score)
        # to judge whether the match is meaningful. Wrapped as untrusted.
        if field in {"subject", "description"}:
            result["text_a"] = wrap_untrusted(f"{field}_a", val_a)
            result["text_b"] = wrap_untrusted(f"{field}_b", val_b)
        return result

    # ---- Tool 3 ----
    def timeline_gap(self, case_id_a: str, case_id_b: str) -> dict:
        """Human-readable + numeric time gap between the two cases."""
        a, b = self.get(case_id_a), self.get(case_id_b)
        t_a, t_b = pd.to_datetime(a["created_at"]), pd.to_datetime(b["created_at"])
        gap = abs((t_a - t_b).total_seconds())
        if gap < 3600:
            human = f"{gap / 60:.0f} minutes apart"
        elif gap < 86400:
            human = f"{gap / 3600:.1f} hours apart"
        else:
            human = f"{gap / 86400:.1f} days apart"
        return {
            "case_id_a": case_id_a,
            "case_id_b": case_id_b,
            "gap_seconds": gap,
            "gap_human_readable": human,
            "earlier_case": case_id_a if t_a < t_b else case_id_b,
        }

    # ---- Tool 4 ----
    def find_other_cases(self, account_name: str = None,
                          contact_email: str = None, exclude_ids=None) -> dict:
        """
        Look for OTHER cases tied to the same account or contact, beyond the
        current pair -- lets the agent check whether this looks like an
        isolated resubmission or part of a broader recurring pattern (e.g.
        the account routinely files many distinct real tickets, which should
        lower suspicion of any single pair being a duplicate).
        """
        exclude_ids = set(exclude_ids or [])
        mask = pd.Series(False, index=self.df.index)
        if account_name:
            mask |= self.df["account_name"].str.strip().str.lower() == account_name.strip().lower()
        if contact_email:
            mask |= self.df["contact_email"].fillna("").str.strip().str.lower() == contact_email.strip().lower()
        matches = self.df[mask]
        matches = matches[~matches["case_id"].isin(exclude_ids)]
        matches = matches.head(10)  # cap payload size
        return {
            "count_found": len(matches),
            "cases": [
                {
                    "case_id": r["case_id"],
                    "created_at": str(r["created_at"]),
                    "subject": clean_field(str(r["subject"])),
                    "status": r["status"],
                }
                for _, r in matches.iterrows()
            ],
        }


# --- Tool schemas for Groq/OpenAI-style function calling ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "compare_fields",
            "description": "Compare structured fields (channel, priority, status, exact email match, contact names) between the two cases in the pair being investigated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id_a": {"type": "string"},
                    "case_id_b": {"type": "string"},
                },
                "required": ["case_id_a", "case_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fuzzy_score",
            "description": "Get a fuzzy text-similarity score (0-100) for one specific field between the two cases. Call once per field you actually need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id_a": {"type": "string"},
                    "case_id_b": {"type": "string"},
                    "field": {
                        "type": "string",
                        "enum": sorted(FUZZY_FIELDS),
                    },
                },
                "required": ["case_id_a", "case_id_b", "field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timeline_gap",
            "description": "Get the time gap between when the two cases were created, human-readable and in seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id_a": {"type": "string"},
                    "case_id_b": {"type": "string"},
                },
                "required": ["case_id_a", "case_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_other_cases",
            "description": "Find OTHER cases (beyond this pair) tied to the same account and/or contact email, to check whether this account just files a lot of distinct real tickets vs this pair being an isolated resubmission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_name": {"type": "string"},
                    "contact_email": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": "Submit your final recommendation for this pair. Only call this once you have gathered enough evidence. This does NOT finalize anything -- it drafts a recommendation that a human must approve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["DUPLICATE", "NOT_DUPLICATE", "UNSURE"],
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0 to 1.0",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_tool": {"type": "string"},
                                "finding": {"type": "string"},
                            },
                            "required": ["source_tool", "finding"],
                        },
                    },
                    "rationale": {
                        "type": "string",
                        "description": "1-2 sentence summary of your reasoning.",
                    },
                },
                "required": ["label", "confidence", "evidence", "rationale"],
            },
        },
    },
]