"""Robot hardware interface. The rest of the system only talks to RobotArm,
so a simulated arm, an Arduino servo arm or a ROS 2 arm are interchangeable."""
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from . import config
from .kinematics import Joints, to_servo


class RobotError(RuntimeError):
    pass


class EmergencyStopped(RobotError):
    pass


class RobotArm(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def move_joints(self, joints: Joints, speed: float) -> None:
        """Move to the given joint angles (deg). `speed` is 0..1 of the maximum joint speed.
        Blocks until the motion is finished."""

    @abstractmethod
    def get_joints(self) -> Joints: ...

    @abstractmethod
    def set_gripper(self, closed: bool) -> None: ...

    @abstractmethod
    def gripper_holding(self) -> bool:
        """True if the closed gripper is gripping an object (current / stall sensing)."""

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def reset_emergency_stop(self) -> None: ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...


class SerialArmRobot(RobotArm):
    """Arduino-driven servo arm over a serial link (see arduino/arm_controller).

    Protocol, one ASCII line per command, 115200 baud:
        P                       -> PONG
        J b s e w dps           -> OK   (after the move finishes; angles are servo degrees)
        G deg                   -> OK   (gripper servo angle)
        Q                       -> S <gripper_current_adc> <gripper_deg>
        E                       -> OK   (servo power cut)
        R                       -> OK   (re-enable after an emergency stop)
    The firmware replies "ESTOP" to any motion command while the hardware E-stop is pressed.
    """

    def __init__(self, port: str, baud: int = 115200, serial_factory: Optional[Callable] = None,
                 hold_current_threshold: int = 300, sleep: Callable[[float], None] = time.sleep):
        self.port, self.baud = port, baud
        self._factory = serial_factory
        self._ser = None
        self._hold_threshold = hold_current_threshold
        self._sleep = sleep
        self._joints: Joints = {}
        self._stopped = False

    # ---- connection
    def connect(self) -> None:
        if self._factory is None:
            import serial  # pyserial, imported lazily so simulation needs no hardware libraries
            self._factory = lambda: serial.Serial(self.port, self.baud, timeout=3.0)
        self._ser = self._factory()
        self._sleep(2.0)                      # Arduino resets when the port opens
        if self._cmd("P", timeout_s=3.0) != "PONG":
            raise RobotError("arm controller did not answer the ping")

    def disconnect(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def _cmd(self, line: str, timeout_s: float = 5.0) -> str:
        if self._ser is None:
            raise RobotError("not connected")
        self._ser.write((line + "\n").encode())
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            reply = self._ser.readline().decode().strip()
            if reply:
                if reply == "ESTOP":
                    self._stopped = True
                    raise EmergencyStopped("hardware emergency stop is active")
                return reply
        raise RobotError(f"timeout waiting for reply to '{line}'")

    # ---- motion
    def move_joints(self, joints: Joints, speed: float) -> None:
        if self._stopped:
            raise EmergencyStopped("emergency stop is active")
        servo = to_servo(joints)
        dps = max(5.0, config.MAX_JOINT_SPEED_DPS * speed)
        line = "J {base:.1f} {shoulder:.1f} {elbow:.1f} {wrist:.1f} {dps:.1f}".format(dps=dps, **servo)
        if self._cmd(line, timeout_s=30.0) != "OK":
            raise RobotError("move rejected by the controller")
        self._joints = dict(joints)

    def get_joints(self) -> Joints:
        return dict(self._joints)

    def set_gripper(self, closed: bool) -> None:
        if self._stopped:
            raise EmergencyStopped("emergency stop is active")
        deg = config.GRIPPER_CLOSED_DEG if closed else config.GRIPPER_OPEN_DEG
        if self._cmd(f"G {deg:.0f}") != "OK":
            raise RobotError("gripper command rejected")
        self._sleep(config.GRIPPER_SETTLE_S)

    def gripper_holding(self) -> bool:
        reply = self._cmd("Q").split()
        if len(reply) != 3 or reply[0] != "S":
            raise RobotError(f"unexpected status reply: {reply}")
        return int(reply[1]) >= self._hold_threshold

    # ---- safety
    def emergency_stop(self) -> None:
        self._stopped = True
        try:
            self._cmd("E", timeout_s=1.0)
        except RobotError:
            pass

    def reset_emergency_stop(self) -> None:
        if self._cmd("R") == "OK":
            self._stopped = False

    @property
    def is_stopped(self) -> bool:
        return self._stopped
