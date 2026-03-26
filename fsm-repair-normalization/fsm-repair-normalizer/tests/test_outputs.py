import json
from pathlib import Path

def test_repaired_fsm_exists():
    out = Path("/output/repaired_fsm.json")
    assert out.exists()

def test_repaired_fsm_is_valid_json():
    data = json.loads(Path("/output/repaired_fsm.json").read_text())
    assert "start_state" in data
    assert "states" in data
    assert "transitions" in data
