"""Language understanding: turn a typed or transcribed command into a structured task.

Rule-based on purpose: it is fast, runs offline on a Raspberry Pi, is fully
deterministic (easy to test) and never invents an object or destination.
An optional LLM parser (llm_parser.py) produces the same ParsedCommand structure.

Example:
    "Pick up the red block and place it in the tray on the left."
    -> action=pick_and_place, object=red, destination=tray(side=left)
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

KNOWN_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black"]
TRAY_WORDS = {"tray", "trays", "box", "boxes", "bin", "bins", "container", "bowl", "basket"}
OBJECT_WORDS = {"block", "cube", "object", "item", "thing", "piece", "one", "blocks", "cubes"}
LARGEST_WORDS = {"largest", "biggest", "big", "large", "bigger"}
SMALLEST_WORDS = {"smallest", "tiniest", "small", "tiny", "smaller"}
DEICTIC_WORDS = {"that", "this", "those", "these"}
SIDE_WORDS = {"left": "left", "right": "right", "center": "center", "middle": "center"}
BESIDE_MARKERS = {"beside", "next to", "near", "by"}
FILLERS = r"\b(please|kindly|robot|hey|hi|could you|can you|would you|will you|now)\b"
VERBS = r"\b(pick up|pick|grab|take|lift|move|put|place|bring|carry|transfer|relocate|drop)\b"
STOP_RE = re.compile(r"\b(stop|halt|abort|freeze)\b")
MARKER_RE = re.compile(r"\b(next to|on top of|inside|into|onto|beside|towards|toward|near|in|on|to|by|at)\b")
SPLIT_RE = re.compile(
    r"\b(?:and|then)\s+(?:then\s+)?(?:place|put|move|drop|bring|carry|take)\s+(?:it|them|that|this)\b")


@dataclass
class ObjectRef:
    color: Optional[str] = None
    size_rank: Optional[str] = None      # "largest" | "smallest"
    deictic: bool = False                # "that object" - cannot be resolved without pointing
    raw: str = ""

    @property
    def is_vague(self) -> bool:
        return self.color is None and self.size_rank is None


@dataclass
class DestRef:
    kind: str                            # tray | center | zone | beside | unsupported
    side: Optional[str] = None           # left | right | center
    ref: Optional[ObjectRef] = None      # for "beside <object>"
    raw: str = ""


@dataclass
class ParsedCommand:
    action: str                          # pick_and_place | stop | unknown
    obj: Optional[ObjectRef] = None
    dest: Optional[DestRef] = None
    text: str = ""
    problems: List[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    t = text.lower().replace("centre", "center")
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(FILLERS, " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_object(phrase: str) -> Optional[ObjectRef]:
    words = phrase.split()
    if not words:
        return None
    ref = ObjectRef(raw=phrase.strip())
    for w in words:
        if w in KNOWN_COLORS and ref.color is None:
            ref.color = w
        elif w in LARGEST_WORDS:
            ref.size_rank = "largest"
        elif w in SMALLEST_WORDS:
            ref.size_rank = "smallest"
        elif w in DEICTIC_WORDS:
            ref.deictic = True
    if ref.color is None and ref.size_rank is None and not (set(words) & (OBJECT_WORDS | DEICTIC_WORDS)):
        return None
    return ref


def parse_destination(marker: str, phrase: str) -> DestRef:
    words = phrase.split()
    raw = f"{marker} {phrase}".strip()
    if marker in BESIDE_MARKERS:
        return DestRef(kind="beside", ref=parse_object(phrase), raw=raw)
    side = next((SIDE_WORDS[w] for w in words if w in SIDE_WORDS), None)
    if set(words) & TRAY_WORDS:
        return DestRef(kind="tray", side=side, raw=raw)
    if side == "center":
        return DestRef(kind="center", side="center", raw=raw)
    if side in ("left", "right"):
        return DestRef(kind="zone", side=side, raw=raw)
    if marker in ("on top of", "onto") or any(w in KNOWN_COLORS for w in words):
        return DestRef(kind="unsupported", raw=raw)       # stacking is not supported
    return DestRef(kind="unsupported", raw=raw)


def _find_dest_marker(text: str):
    """First destination preposition, skipping ones that just describe the object
    (e.g. 'the block on the left to the tray' -> the marker is 'to')."""
    matches = list(MARKER_RE.finditer(text))
    for i, m in enumerate(matches):
        after = text[m.end():].strip()
        starts_with_side = re.match(r"(the\s+)?(left|right|center|middle)\b(?!\s+(tray|box|bin|container|bowl|basket))", after)
        later_marker = i + 1 < len(matches)
        if starts_with_side and later_marker:
            continue
        return m
    return None


def parse_command(text: str) -> ParsedCommand:
    t = _normalise(text)
    cmd = ParsedCommand(action="unknown", text=text)
    if not t:
        cmd.problems.append("empty_command")
        return cmd
    if STOP_RE.search(t):
        cmd.action = "stop"
        return cmd
    if not re.search(VERBS, t):
        cmd.problems.append("no_action")
        return cmd
    cmd.action = "pick_and_place"

    split = SPLIT_RE.search(t)
    if split:
        obj_part, rest = t[:split.start()], t[split.end():].strip()
        m = MARKER_RE.match(rest) or MARKER_RE.search(rest)
        marker, dest_part = (m.group(1), rest[m.end():].strip()) if m else (None, "")
    else:
        m = _find_dest_marker(t)
        if m:
            obj_part, marker, dest_part = t[:m.start()], m.group(1), t[m.end():].strip()
        else:
            obj_part, marker, dest_part = t, None, ""

    obj_part = re.sub(VERBS, " ", obj_part)
    obj_part = re.sub(r"\b(up|it|them|and|then|the|a|an)\b", " ", obj_part)
    cmd.obj = parse_object(re.sub(r"\s+", " ", obj_part).strip())
    if cmd.obj is None:
        cmd.problems.append("no_object")
    if marker is None:
        cmd.problems.append("missing_destination")
    else:
        cmd.dest = parse_destination(marker, dest_part)
        if cmd.dest.kind == "unsupported":
            cmd.problems.append("unsupported_destination")
        if cmd.dest.kind == "beside" and cmd.dest.ref is None:
            cmd.problems.append("unclear_destination")
    return cmd
