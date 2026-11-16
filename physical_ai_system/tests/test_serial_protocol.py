"""Tests the Arduino serial driver against a fake serial port that behaves like the firmware."""
import pytest

from physical_ai import config
from physical_ai.kinematics import solve_ik
from physical_ai.robot_arm import EmergencyStopped, RobotError, SerialArmRobot


class FakeSerial:
    def __init__(self, current=0, estop=False):
        self.sent, self._replies = [], []
        self.current, self.estop, self.closed = current, estop, False

    def write(self, data: bytes):
        line = data.decode().strip()
        self.sent.append(line)
        cmd = line[0]
        if cmd == "P":
            r = "PONG"
        elif cmd in "JG" and self.estop:
            r = "ESTOP"
        elif cmd in "JG":
            r = "OK"
        elif cmd == "Q":
            r = f"S {self.current} 120"
        elif cmd == "E":
            r = "OK"
        elif cmd == "R":
            r = "OK"
        else:
            r = "ERR"
        self._replies.append(r + "\n")

    def readline(self):
        return self._replies.pop(0).encode() if self._replies else b""

    def close(self):
        self.closed = True


def make(fake):
    r = SerialArmRobot("fake", serial_factory=lambda: fake, sleep=lambda s: None)
    r.connect()
    return r


def test_connect_pings():
    fake = FakeSerial()
    make(fake)
    assert fake.sent == ["P"]


def test_move_sends_servo_angles_within_range():
    fake = FakeSerial()
    r = make(fake)
    r.move_joints(solve_ik(150, 30, 20), speed=0.5)
    parts = fake.sent[-1].split()
    assert parts[0] == "J" and len(parts) == 6
    assert all(0 <= float(p) <= 180 for p in parts[1:5])
    assert float(parts[5]) == pytest.approx(config.MAX_JOINT_SPEED_DPS * 0.5)


def test_gripper_commands():
    fake = FakeSerial()
    r = make(fake)
    r.set_gripper(True)
    r.set_gripper(False)
    assert fake.sent[-2:] == [f"G {config.GRIPPER_CLOSED_DEG:.0f}", f"G {config.GRIPPER_OPEN_DEG:.0f}"]


def test_holding_uses_current_sensing():
    assert make(FakeSerial(current=520)).gripper_holding() is True
    assert make(FakeSerial(current=40)).gripper_holding() is False


def test_hardware_estop_raises_and_latches():
    r = make(FakeSerial(estop=True))
    with pytest.raises(EmergencyStopped):
        r.move_joints(solve_ik(150, 0, 30), 0.3)
    assert r.is_stopped
    with pytest.raises(EmergencyStopped):
        r.set_gripper(True)


def test_software_estop_then_reset():
    fake = FakeSerial()
    r = make(fake)
    r.emergency_stop()
    assert r.is_stopped and "E" in fake.sent
    r.reset_emergency_stop()
    assert not r.is_stopped


def test_timeout_raises():
    fake = FakeSerial()
    r = make(fake)
    fake.write = lambda data: None          # controller goes silent
    with pytest.raises(RobotError):
        r._cmd("J 1 2 3 4 5", timeout_s=0.05)
