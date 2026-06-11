# Output Slimming Plan

## Goal

Reduce noisy console output and prevent `outputs/` from growing with large, low-value log artifacts.

## Scope

1. Update `v2/sft.sh` to:
   - default to epoch-level logging
   - default to epoch-level checkpoint saving
   - keep only the latest checkpoint by default
   - disable external logger backends by default
   - show only epoch averages, final checkpoint/runtime, and errors
   - launch through the active virtual environment's Python
2. Update `v2/eval.py` and `v2/eval_hitk.py` to:
   - print summary metrics only by default
   - hide progress bars by default
   - write detailed prediction JSONL only when explicitly enabled
3. Add a cleanup script under `ops/` to remove historical large pure-log artifacts:
   - `logging.jsonl`
   - `completions.jsonl`
   - `events.out.tfevents*`
   - `*predictions*.jsonl`
   - `*eval.jsonl`
   - `*.log`

Checkpoint directories are deliberately excluded because they contain model
weights, not pure logs.

## Rollout

1. Dry-run: verify shell commands and CLI argument composition.
2. Smoke: run tiny eval/train command construction with new defaults.
3. Sample: apply cleanup to a sample subtree or with preview mode first.
4. Full: run cleanup without preview on the target remote workspace.
