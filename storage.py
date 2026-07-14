"""
Simple JSON persistence layer for investigations and audit trail.
"""
import json
import os
from datetime import datetime, timezone

DB_FILE = "investigations.json"


def _init():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump({"investigations": []}, f, indent=2)


def load_db():
    _init()
    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_investigation(inv):
    db = load_db()
    for existing in db["investigations"]:
        if (existing["case_id_a"] == inv.case_id_a and existing["case_id_b"] == inv.case_id_b):
            print(f"Skipped: {inv.case_id_a}/{inv.case_id_b} already investigated")
            return
    db["investigations"].append(inv.to_dict())
    save_db(db)

def list_pending():
    db = load_db()
    return [
        inv
        for inv in db["investigations"]
        if inv["status"] == "pending_review"
    ]


def get_investigation(investigation_id):
    db = load_db()

    for inv in db["investigations"]:
        if inv["investigation_id"] == investigation_id:
            return inv

    return None


def record_decision(
    investigation_id,
    decision,
    reviewer,
    override_verdict=None,
):
    db = load_db()

    for inv in db["investigations"]:

        if inv["investigation_id"] != investigation_id:
            continue

        if inv["human_decision"] is not None:
            raise ValueError("Decision already recorded.")

        inv["human_decision"] = {
            "decision": decision,
            "reviewer": reviewer,
            "override_verdict": override_verdict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        inv["status"] = "decided"

        inv["trace"].append(
            {
                "step": len(inv["trace"]) + 1,
                "type": "human_decision",
                "decision": decision,
                "reviewer": reviewer,
                "override_verdict": override_verdict,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        save_db(db)

        return inv

    raise ValueError("Investigation not found")


def get_audit_log():
    return load_db()["investigations"]