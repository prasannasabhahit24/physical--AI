"""Optional language-model parser for looser phrasing than the rule-based parser handles.

It asks Claude to return the SAME structure the rule-based parser produces, as JSON, and
falls back to the rule-based parser on any error. The model only extracts intent; it never
sees the camera or controls the robot, and everything it returns still goes through
grounding and the safety checks.

Needs:  pip install anthropic   and the ANTHROPIC_API_KEY environment variable.
Note: this module has not been exercised against the live API in this project's tests.
"""
import json
import re

from . import config
from .command_parser import DestRef, ObjectRef, ParsedCommand, parse_command

SYSTEM_PROMPT = """You convert robot commands into JSON. Reply with JSON only, no prose.
Schema: {"action": "pick_and_place" | "stop" | "unknown",
 "object": {"color": str|null, "size_rank": "largest"|"smallest"|null, "deictic": bool} | null,
 "destination": {"kind": "tray"|"center"|"zone"|"beside"|"unsupported", "side": "left"|"right"|"center"|null,
                 "ref": {"color": str|null, "size_rank": str|null, "deictic": bool} | null} | null}
Rules: never guess a missing colour or destination - use null. "that"/"this" with no colour means deictic=true."""


def _obj(d):
    if not d:
        return None
    return ObjectRef(color=d.get("color"), size_rank=d.get("size_rank"), deictic=bool(d.get("deictic")))


def parse_with_llm(text: str, client=None) -> ParsedCommand:
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        msg = client.messages.create(model=config.LLM_MODEL, max_tokens=300, system=SYSTEM_PROMPT,
                                     messages=[{"role": "user", "content": text}])
        raw = re.search(r"\{.*\}", msg.content[0].text, re.S).group(0)
        data = json.loads(raw)
        cmd = ParsedCommand(action=data.get("action", "unknown"), text=text, obj=_obj(data.get("object")))
        d = data.get("destination")
        if d:
            cmd.dest = DestRef(kind=d.get("kind", "unsupported"), side=d.get("side"), ref=_obj(d.get("ref")))
        if cmd.action == "pick_and_place":
            if cmd.obj is None:
                cmd.problems.append("no_object")
            if cmd.dest is None:
                cmd.problems.append("missing_destination")
            elif cmd.dest.kind == "unsupported":
                cmd.problems.append("unsupported_destination")
            elif cmd.dest.kind == "beside" and cmd.dest.ref is None:
                cmd.problems.append("unclear_destination")
        return cmd
    except Exception:
        return parse_command(text)          # safe fallback
