FSM Repair, Validation, and Normalization
Overview
You are given a finite state machine (FSM) definition along with an event sequence and a set of constraints.
Your task is to validate, repair, and normalize the FSM so that it becomes deterministic, executable for the given event sequence, and compliant with all constraints.
If validation or repair is impossible, your program must exit with a non-zero status code.
Input / Output (STRICT)
Your program must read exactly the following files:
•	/input/fsm.json
•	/input/events.json
•	/input/constraints.json
Your program must write exactly:
•	/output/repaired_fsm.json
No other output files are allowed.
Required solver file and literals (MANDATORY)
Your solver must exist at:
/app/repair_fsm.py
That file must contain these two lines exactly (do not modify spacing, quotes, or structure):
input_dir = Path("/input")
output_file = Path("/output/repaired_fsm.json")
Any solution that changes or omits these literals will be rejected.
FSM Structure
The input FSM (fsm.json) has the following structure:
{
  "states": ["A", "B", "C"],
  "start_state": "A",
  "transitions": {
    "A": { "x": "B" },
    "B": { "y": "C" }
  }
}
Structural requirements
•	states must be a list of unique strings
•	start_state must be a string and must appear in states
•	transitions must be a dictionary:
o	keys are state names
o	values are dictionaries mapping event strings to target state strings
•	All state names and event names must be strings
Violations of these rules must result in non-zero exit.
Events
events.json must be a list of strings.
If events.json is not a list, or contains non-string values, your program must error.
Constraints
constraints.json is a JSON object that may contain:
•	forbid_cycles (boolean)
•	max_states (integer)
•	state_name_pattern (regex string)
Constraint defaults
If a constraint field is missing:
•	forbid_cycles → false
•	max_states → unlimited
•	state_name_pattern → no regex enforcement
Validation Rules
Your program must fail (exit non-zero) if:
•	start_state is missing or not in states
•	states contains duplicates
•	any state name violates state_name_pattern
•	max_states is exceeded at any point
•	transitions reference non-existent states
•	transitions contain non-string keys or non-string targets
•	events.json is invalid
•	constraints.json is not an object
•	cycles exist when forbid_cycles == true
Repair Rules
If the FSM is structurally valid, you must repair it as follows:
1. Reachability
•	Remove all states that are unreachable from start_state
•	Remove all transitions belonging to removed states
•	If any remaining transition references a removed state → error
2. Missing transitions (fallback)
For every (state, event) pair used during execution:
•	If a transition is missing, create a fallback transition
•	The fallback target must be the lexicographically smallest existing state
•	Fallback transitions must NOT create new states
3. Missing target states
If a transition references a state not in states:
•	That state must be generated
•	Generated states must:
o	respect state_name_pattern
o	not cause max_states to be exceeded
•	If generation is impossible → error
Execution Requirement
Starting from start_state, the FSM must be able to process the full events sequence after repair.
If execution fails at any step → error.
Determinism and Output Format (STRICT)
Your output must be deterministic and byte-identical for identical inputs.
Formatting requirements
•	Single-line JSON
•	No trailing newline
•	No extra whitespace
•	Keys sorted lexicographically at all levels
Serialization must match:
json.dumps(obj, separators=(",", ":"), sort_keys=True)
Ordering Rules
•	Original reachable states must appear first, sorted lexicographically
•	Generated states must appear after originals, sorted lexicographically
•	The transitions object must list states in the same order as states
Numeric Normalization
Apply recursively across the entire output:
•	Integers with absolute value > 2⁵³ → convert to strings
•	Floats with trailing zeros → trim zeros
•	Floats equal to integers (e.g. 1.0) → convert to integers
Booleans and null must be preserved exactly.
Failure Semantics
If any rule above cannot be satisfied:
•	Exit with a non-zero exit code
•	Do not write partial or malformed output
Summary
This task evaluates your ability to:
•	validate complex structured input
•	enforce constraints precisely
•	repair incomplete FSMs deterministically
•	produce byte-stable normalized output
Correctness, determinism, and strict adherence to the specification are required.
