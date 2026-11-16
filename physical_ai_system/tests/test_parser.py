import pytest

from physical_ai.command_parser import parse_command


def test_pick_and_place_into_left_tray():
    c = parse_command("Pick up the red block and place it in the tray on the left.")
    assert c.action == "pick_and_place" and c.problems == []
    assert c.obj.color == "red"
    assert (c.dest.kind, c.dest.side) == ("tray", "left")


def test_put_blue_in_right_tray():
    c = parse_command("Put the blue block in the right tray.")
    assert c.obj.color == "blue" and (c.dest.kind, c.dest.side) == ("tray", "right")


def test_box_is_a_tray():
    c = parse_command("move the blue object into the box")
    assert c.obj.color == "blue" and c.dest.kind == "tray" and c.dest.side is None


def test_largest_to_centre():
    c = parse_command("Move the largest object to the centre.")
    assert c.obj.size_rank == "largest" and c.dest.kind == "center"


def test_beside():
    c = parse_command("Put the blue block beside the red block.")
    assert c.obj.color == "blue" and c.dest.kind == "beside" and c.dest.ref.color == "red"


def test_next_to():
    c = parse_command("place the yellow block next to the blue block")
    assert c.dest.kind == "beside" and c.dest.ref.color == "blue"


def test_missing_destination():
    c = parse_command("Pick up the green block.")
    assert c.obj.color == "green" and "missing_destination" in c.problems


def test_deictic_object_is_vague():
    c = parse_command("Move that object to the tray.")
    assert c.obj.is_vague and c.obj.deictic and c.dest.kind == "tray"


def test_object_described_by_side_then_destination():
    c = parse_command("Take the red block on the left to the right tray")
    assert c.obj.color == "red" and (c.dest.kind, c.dest.side) == ("tray", "right")


def test_please_and_politeness():
    c = parse_command("Could you please put the red block in the left tray?")
    assert c.obj.color == "red" and c.dest.side == "left"


def test_stacking_is_unsupported():
    c = parse_command("put the red block on top of the blue block")
    assert "unsupported_destination" in c.problems


@pytest.mark.parametrize("text", ["stop", "STOP!", "halt now"])
def test_stop(text):
    assert parse_command(text).action == "stop"


@pytest.mark.parametrize("text", ["", "hello there", "what is the weather"])
def test_unknown(text):
    assert parse_command(text).action == "unknown"
