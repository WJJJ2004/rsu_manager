#!/usr/bin/env python3

import math
import os
import time
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

from roa_interfaces.msg import (
    RsuTarget,
    RsuSolution,
    MotorCommand,
    MotorCommandArray,
    MotorStateArray,
    RsuState,
    RsuStateArray
)

from rsu_manager.util.gamepad_reader import Gamepad


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rad(deg):
    return math.radians(deg)


cmd_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
)

rsu_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)

state_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class RsuHwGamepadTestNode(Node):
    def __init__(self):
        super().__init__("hw_controll_test_node")

        # ===== Parameters =====
        self.rate_hz = float(self.declare_parameter("rate_hz", 100.0).value)
        self.input_deadzone = float(self.declare_parameter("input_deadzone", 0.05).value)
        self.wait_first_input = bool(self.declare_parameter("wait_first_input", True).value)

        self.roll_rate_scale = float(self.declare_parameter("roll_rate_scale", 1.5).value)
        self.pitch_rate_scale = float(self.declare_parameter("pitch_rate_scale", 1.5).value)

        self.roll_min = float(self.declare_parameter("roll_min_limit_rad", rad(-30.0)).value)
        self.roll_max = float(self.declare_parameter("roll_max_limit_rad", rad(30.0)).value)
        self.pitch_min = float(self.declare_parameter("pitch_min_limit_rad", rad(-30.0)).value)
        self.pitch_max = float(self.declare_parameter("pitch_max_limit_rad", rad(30.0)).value)

        self.device_path = str(self.declare_parameter("device_path", "").value)
        self.vendor_id = int(self.declare_parameter("vendor_id", 0x046D).value)
        self.product_id = int(self.declare_parameter("product_id", 0xC219).value)

        # ===== Gamepad =====
        self.gamepad = Gamepad(
            vendor_id=self.vendor_id,
            product_id=self.product_id,
            vel_scale_x=1.0,
            vel_scale_y=1.0,
            vel_scale_rot=1.0,
            device_path=(self.device_path if self.device_path else None),
            prefer_name_contains="Logitech",
        )

        # ===== Publishers / Subscribers =====
        self.pub_rsu_target = self.create_publisher(RsuTarget, "/rsu/target", rsu_qos)
        self.pub_motor_cmd = self.create_publisher(MotorCommandArray, "/hardware_interface/command", cmd_qos)

        self.sub_rsu_solution = self.create_subscription(
            RsuSolution,
            "/rsu/solution",
            self.on_rsu_solution,
            rsu_qos,
        )

        self.sub_motor_state = self.create_subscription(
            MotorStateArray,
            "/hardware_interface/state",
            self.on_motor_state,
            state_qos,
        )

        self.sub_rsu_state = self.create_subscription(
            RsuStateArray,
            "/rsu/state",
            self.on_rsu_state,
            rsu_qos,
        )

        # ===== Motor config =====
        self.motor_ids = []
        self.motor_types = []
        self.load_motor_yaml()

        # ===== RSU target state =====
        self.left_roll = 0.0
        self.left_pitch = 0.0
        self.right_roll = 0.0
        self.right_pitch = 0.0

        self.selected_foot = "left"

        # gamepad_rpy_node와 동일한 내부 적분 상태
        self.roll = 0.0
        self.pitch = 0.0
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.received_first_input = False
        self.last_t = time.time()

        # 최근 API의 4번째 값 edge 감지용
        self.prev_button_state = None

        # ===== RSU actuator command state =====
        self.rsu_solution_ready = False
        self.actuator_cmd = {
            18: None,  # left actuator 1
            20: None,  # left actuator 2
            19: None,  # right actuator 1
            21: None,  # right actuator 2
        }

        # 현재 encoder 기반 hold용
        self.latest_motor_pos = {}

        # ===== Initial full body pose =====
        self.initial_position = {
            9: 0.0,

            10: rad(-20.0),
            12: 0.0,
            14: 0.0,
            16: rad(50.0),

            11: rad(20.0),
            13: 0.0,
            15: 0.0,
            17: rad(-50.0),
        }

        self.kp_map = {
            9: 50.0,
            10: 150.0,
            12: 200.0,
            14: 100.0,
            16: 150.0,
            11: 150.0,
            13: 200.0,
            15: 100.0,
            17: 150.0,
        }

        self.kd_map = {
            9:  2.0,
            10: 24.722,
            12: 26.387,
            14: 3.419,
            16: 8.654,
            11: 24.722,
            13: 26.387,
            15: 3.419,
            17: 8.654,
        }

        self.default_rsu_kp = 40.0
        self.default_rsu_kd = 0.99

        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"Started RSU HW gamepad test node @ {self.rate_hz:.1f} Hz. "
            f"Selected foot: {self.selected_foot}"
        )

    def load_motor_yaml(self, yaml_path=None):
        if yaml_path is None:
            try:
                from ament_index_python.packages import get_package_share_directory
                share_dir = get_package_share_directory("robstride_hardware_interface")
                yaml_path = os.path.join(share_dir, "config", "motor_setting.yaml")
            except Exception:
                yaml_path = os.path.expanduser(
                    "~/colcon_ws/src/robstride_hardware_interface/config/motor_setting.yaml"
                )

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        params = config["hardware_interface_node"]["ros__parameters"]
        can_interfaces = params["can_interfaces"]

        for can_name in can_interfaces:
            self.motor_ids += params[can_name]["motor_ids"]
            self.motor_types += params[can_name]["motor_type"]

        self.get_logger().info(f"Loaded motor ids: {self.motor_ids}")

    def on_motor_state(self, msg: MotorStateArray):
        for s in msg.states:
            self.latest_motor_pos[int(s.motor_id)] = float(s.position)

    def on_rsu_state(self, msg: RsuStateArray):
        # estimator q
        est_l_roll = float(msg.q.left_rsu_roll)
        est_l_pitch = float(msg.q.left_rsu_pitch)
        est_r_roll = float(msg.q.right_rsu_roll)
        est_r_pitch = float(msg.q.right_rsu_pitch)

        # 현재 HW test node가 publish 중인 target
        tgt_l_roll = float(self.left_roll)
        tgt_l_pitch = float(self.left_pitch)
        tgt_r_roll = float(self.right_roll)
        tgt_r_pitch = float(self.right_pitch)

        # target - estimator error
        err_l_roll = tgt_l_roll - est_l_roll
        err_l_pitch = tgt_l_pitch - est_l_pitch
        err_r_roll = tgt_r_roll - est_r_roll
        err_r_pitch = tgt_r_pitch - est_r_pitch

        # Check for large errors and warn
        error_threshold = 0.01
        large_errors = []

        if abs(err_l_roll) > error_threshold:
            large_errors.append(f"LEFT roll={err_l_roll:+.4f}")
        if abs(err_l_pitch) > error_threshold:
            large_errors.append(f"LEFT pitch={err_l_pitch:+.4f}")
        if abs(err_r_roll) > error_threshold:
            large_errors.append(f"RIGHT roll={err_r_roll:+.4f}")
        if abs(err_r_pitch) > error_threshold:
            large_errors.append(f"RIGHT pitch={err_r_pitch:+.4f}")

        if large_errors:
            self.get_logger().warn(
            f"[RSU ERROR THRESHOLD EXCEEDED] {', '.join(large_errors)}",
            throttle_duration_sec=0.5,
            )

    def on_rsu_solution(self, msg: RsuSolution):
        if not msg.feasible:
            self.get_logger().warn(
                "RSU solution infeasible. Holding previous actuator commands.",
                throttle_duration_sec=1.0,
            )
            return

        self.actuator_cmd[18] = float(msg.left_actuator_1)
        self.actuator_cmd[20] = float(msg.left_actuator_2)
        self.actuator_cmd[19] = float(msg.right_actuator_1)
        self.actuator_cmd[21] = float(msg.right_actuator_2)
        self.rsu_solution_ready = True

    def handle_gamepad_toggle(self, button_state: bool):
        if self.prev_button_state is None:
            self.prev_button_state = button_state
            return

        if button_state == self.prev_button_state:
            return

        self.prev_button_state = button_state

        if self.selected_foot == "left":
            self.selected_foot = "right"
            self.roll = self.right_roll
            self.pitch = self.right_pitch
        else:
            self.selected_foot = "left"
            self.roll = self.left_roll
            self.pitch = self.left_pitch

        self.prev_roll = self.roll
        self.prev_pitch = self.pitch

        self.get_logger().info(
            f"Selected foot toggled -> {self.selected_foot} | "
            f"roll={self.roll:.4f}, pitch={self.pitch:.4f}"
        )

    def update_rsu_target_from_gamepad(self):
        if not self.gamepad.is_running:
            self.get_logger().error(
                "Gamepad not running. Holding current RSU target.",
                throttle_duration_sec=2.0,
            )
            return

        now = time.time()
        dt = clamp(now - self.last_t, 0.0, 0.1)
        self.last_t = now

        cmd = self.gamepad.get_command()

        vx = float(cmd[0])  # pitch input
        vy = float(cmd[1])  # roll input

        if len(cmd) >= 4:
            button_state = bool(cmd[3])
            self.handle_gamepad_toggle(button_state)

        mag = max(abs(vx), abs(vy))

        if (not self.received_first_input) and mag > self.input_deadzone:
            self.received_first_input = True
            self.get_logger().info(f"First input detected. mag={mag:.3f}")

        if self.wait_first_input and (not self.received_first_input):
            return

        roll_dot = vy * self.roll_rate_scale
        pitch_dot = vx * self.pitch_rate_scale

        self.prev_roll = self.roll
        self.prev_pitch = self.pitch

        self.roll = clamp(self.roll + roll_dot * dt, self.roll_min, self.roll_max)
        self.pitch = clamp(self.pitch + pitch_dot * dt, self.pitch_min, self.pitch_max)

        if self.selected_foot == "left":
            self.left_roll = self.roll
            self.left_pitch = self.pitch
        else:
            self.right_roll = self.roll
            self.right_pitch = self.pitch

    def publish_rsu_target(self):
        msg = RsuTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.seq = 0

        msg.left_roll = float(self.left_roll)
        msg.left_pitch = float(self.left_pitch)
        msg.right_roll = float(self.right_roll)
        msg.right_pitch = float(self.right_pitch)

        self.pub_rsu_target.publish(msg)

    def get_command_position(self, motor_id: int):
        if motor_id in self.actuator_cmd:
            if self.actuator_cmd[motor_id] is not None:
                return self.actuator_cmd[motor_id]

            if motor_id in self.latest_motor_pos:
                return self.latest_motor_pos[motor_id]

            return None

        return self.initial_position.get(motor_id, 0.0)

    def publish_motor_command(self):
        msg = MotorCommandArray()

        for motor_id in self.motor_ids:
            pos = self.get_command_position(int(motor_id))

            if pos is None:
                self.get_logger().warn(
                    "RSU actuator position is not ready and no encoder hold value exists. "
                    "Skipping command publish.",
                    throttle_duration_sec=1.0,
                )
                return

            is_rsu = motor_id in [18, 20, 19, 21]

            msg.commands.append(
                MotorCommand(
                    motor_id=int(motor_id),
                    torque=0.0,
                    position=float(pos),
                    velocity=0.0,
                    kp=float(self.default_rsu_kp if is_rsu else self.kp_map.get(int(motor_id), 20.0)),
                    kd=float(self.default_rsu_kd if is_rsu else self.kd_map.get(int(motor_id), 0.99)),
                )
            )

        self.pub_motor_cmd.publish(msg)

    def on_timer(self):
        self.update_rsu_target_from_gamepad()
        self.publish_rsu_target()
        self.publish_motor_command()


def main(args=None):
    rclpy.init(args=args)
    node = RsuHwGamepadTestNode()

    try:
        rclpy.spin(node)
    finally:
        try:
            node.gamepad.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()