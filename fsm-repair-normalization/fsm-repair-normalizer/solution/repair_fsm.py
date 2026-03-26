import json
from pathlib import Path

# REQUIRED LITERALS — DO NOT MODIFY
input_dir = Path("/input")
output_file = Path("/output/repaired_fsm.json")

def main():
    fsm_path = input_dir / "fsm.json"

    if fsm_path.exists():
        fsm = json.loads(fsm_path.read_text())
    else:
        fsm = {}

    # Normalize keys
    states = sorted(set(fsm.get("states", [])))
    transitions = fsm.get("transitions", [])

    start_state = (
        fsm.get("initial_state")
        or fsm.get("start_state")
        or (states[0] if states else "S0")
    )

    repaired = {
        "start_state": start_state,
        "states": states or [start_state],
        "transitions": transitions,
    }

    # Deterministic single-line JSON, sorted keys, no newline
    output_file.write_text(
        json.dumps(repaired, sort_keys=True, separators=(",", ":"))
    )

if __name__ == "__main__":
    main()
