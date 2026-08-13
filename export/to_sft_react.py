"""IR → cold-start SFT in ReAct message format (CLAUDE.md §7).

Emits the exact shape from §7:

    {"messages":[
      {"role":"system","content":"<tools/instructions>"},
      {"role":"user","content":"<task>"},
      {"role":"assistant","content":"Thought: ...\\nAction: relax(structure, INCAR={...})",
        "tool_calls":[{"id":"c1","type":"function","function":{"name":"relax","arguments":"{...}"}}]},
      {"role":"tool","tool_call_id":"c1","content":"Observation: converged=True, E=-..."},
      {"role":"assistant","content":"Thought: ...\\nFinal Answer: ..."}]}

Observation tokens must be masked out of the loss (§7). We don't tokenize here
(tokenizer-specific), so each message carries a ``loss`` flag in a side channel
(``loss_mask`` list aligned to ``messages``): assistant messages → train (1),
system/user/tool(observation) → masked (0). ``export.to_rlvr`` consumes the same
convention for ``response_mask``.

Admission: only ``Trajectory.is_admissible()`` trajectories are exported
(execution-verified + multi-judge, §1.1/§1.5). Failure branches are KEPT (§1.3):
a recovered error appears as a tool observation reporting the error followed by
the assistant's correction action — exactly the error→recovery supervision we
want.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from ir import Step, Trajectory

DEFAULT_SYSTEM_PROMPT = (
    "You are a computational-chemistry experiment agent. Think step by step, then "
    "either call a tool (Action) or give a Final Answer. Use the ReAct format."
)


def _render_action_text(step: Step) -> str:
    args = json.dumps(step.action.params, sort_keys=True)
    return f"Action: {step.action.name}({args})"


def _tool_call(step: Step, call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": step.action.name,
            "arguments": json.dumps(step.action.params, sort_keys=True),
        },
    }


def _observation_text(step: Step) -> str:
    obs = step.observation
    if obs.text:
        return obs.text
    body = json.dumps(obs.content, sort_keys=True) if obs.content else ""
    prefix = "Observation:"
    if obs.exit_status not in (None, 0):
        prefix += f" [error exit_status={obs.exit_status} {obs.exit_message or ''}]".rstrip()
    return f"{prefix} {body}".strip()


def trajectory_to_messages(
    traj: Trajectory,
    *,
    system_prompt: Optional[str] = None,
    require_admissible: bool = True,
) -> dict[str, Any]:
    """Convert one trajectory into a ReAct ``{"messages": [...], "loss_mask": [...]}``.

    `loss_mask[i]` is 1 where message i should contribute to the loss (assistant
    turns) and 0 where it must be masked (system/user/tool observations) — §7.
    """
    if require_admissible and not traj.is_admissible():
        raise ValueError(
            f"trajectory {traj.id!r} is not admissible (verified={traj.verification.passed}); "
            "refusing to export un-verified data (§1.1). Pass require_admissible=False only for inspection."
        )

    messages: list[dict[str, Any]] = []
    loss_mask: list[int] = []

    def add(msg: dict[str, Any], loss: int) -> None:
        messages.append(msg)
        loss_mask.append(loss)

    sys_content = system_prompt or traj.system_prompt or DEFAULT_SYSTEM_PROMPT
    add({"role": "system", "content": sys_content}, 0)
    add({"role": "user", "content": traj.goal}, 0)

    for i, step in enumerate(traj.steps):
        call_id = f"c{i+1}"
        thought = step.thought or ""
        content = (f"Thought: {thought}\n" if thought else "") + _render_action_text(step)
        add(
            {"role": "assistant", "content": content, "tool_calls": [_tool_call(step, call_id)]},
            1,  # train on the assistant action (thought + action)
        )
        add(
            {"role": "tool", "tool_call_id": call_id, "content": _observation_text(step)},
            0,  # mask the observation (§7)
        )

    # Final assistant turn (Final Answer), trained.
    final = _final_answer(traj)
    add({"role": "assistant", "content": f"Final Answer: {final}"}, 1)

    return {"messages": messages, "loss_mask": loss_mask}


def _final_answer(traj: Trajectory) -> str:
    t = traj.terminal_step
    if t is None:
        return "done"
    if traj.succeeded:
        res = (t.observation.content or {}).get("res")
        return f"Completed successfully. Result: {json.dumps(res, sort_keys=True)}" if res else "Completed successfully."
    return f"Run did not converge cleanly: {t.observation.exit_message or 'see observations'}."


def export_jsonl(
    trajectories: Iterable[Trajectory],
    path: str,
    *,
    system_prompt: Optional[str] = None,
    require_admissible: bool = True,
) -> int:
    """Write admissible trajectories as JSONL (one ReAct record per line).

    Returns the number of records written; skips non-admissible ones (logged via
    return delta vs input count by the caller).
    """
    n = 0
    with open(path, "w") as f:
        for traj in trajectories:
            if require_admissible and not traj.is_admissible():
                continue
            record = trajectory_to_messages(traj, system_prompt=system_prompt, require_admissible=require_admissible)
            f.write(json.dumps(record) + "\n")
            n += 1
    return n
