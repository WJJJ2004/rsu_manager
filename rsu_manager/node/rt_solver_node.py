#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from roa_interfaces.msg import RsuTarget, RsuSolution, MotorStateArray, RsuStateArray

from rsu_manager.util.core import (
    RSUCore,
    RSUEstimatorFactory,
    stamp_to_ns,
)


class RSURtSolverNode(Node):
    """
    실시간 제어 전용 RSU solver/state estimator 노드.

    입력:
      /rsu/target
      /hardware_interface/state

    출력:
      /rsu/solution
      /rsu/state
    """

    def __init__(self):
        super().__init__("rsu_rt_solver_node")

        self.core = RSUCore(self)
        self.solver = self.core.solver

        self.hold_alpha_on_infeasible = bool(
            self.declare_parameter("hold_alpha_on_infeasible", True).value
        )

        self._last_seq = None
        self._last_stamp = None

        self.prev_alpha_2d = np.zeros((2, 2), dtype=np.float64)

        left_ac1_id = self.declare_parameter("left_ac1_id", 18).value
        left_ac2_id = self.declare_parameter("left_ac2_id", 20).value
        right_ac1_id = self.declare_parameter("right_ac1_id", 19).value
        right_ac2_id = self.declare_parameter("right_ac2_id", 21).value

        self.motor_state = {
            "left_ac1": {"pos": None, "vel": None, "id": left_ac1_id},
            "left_ac2": {"pos": None, "vel": None, "id": left_ac2_id},
            "right_ac1": {"pos": None, "vel": None, "id": right_ac1_id},
            "right_ac2": {"pos": None, "vel": None, "id": right_ac2_id},
        }

        self.left_q_seed = np.array(
            self.declare_parameter("left_q_seed", [0.0, 0.0]).value,
            dtype=np.float64,
        )
        self.right_q_seed = np.array(
            self.declare_parameter("right_q_seed", [0.0, 0.0]).value,
            dtype=np.float64,
        )
        self.left_alpha_seed = np.array(
            self.declare_parameter("left_alpha_seed", [0.0, 0.0]).value,
            dtype=np.float64,
        )
        self.right_alpha_seed = np.array(
            self.declare_parameter("right_alpha_seed", [0.0, 0.0]).value,
            dtype=np.float64,
        )

        estimator_factory = RSUEstimatorFactory(self, self.solver)

        self.left_estimator = estimator_factory.make()
        self.right_estimator = estimator_factory.make()

        self.left_estimator.reset(
            q_init=self.left_q_seed,
            alpha_seed=self.left_alpha_seed,
            initialized=True,
        )
        self.right_estimator.reset(
            q_init=self.right_q_seed,
            alpha_seed=self.right_alpha_seed,
            initialized=True,
        )

        self._last_motor_state_stamp_ns = None
        self._rsu_state_seq = 0
        self.rsu_state_frame_id = str(
            self.declare_parameter("rsu_state_frame_id", "base_link").value
        )

        rsu_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        motor_status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.pub_both_foot_solution = self.create_publisher(
            RsuSolution,
            "/rsu/solution",
            rsu_qos,
        )
        self.sub_both_foot_request = self.create_subscription(
            RsuTarget,
            "/rsu/target",
            self._on_both_foot_request,
            rsu_qos,
        )
        self.sub_motor_status = self.create_subscription(
            MotorStateArray,
            "/hardware_interface/state",
            self._on_motor_state,
            motor_status_qos,
        )
        self.pub_rsu_state = self.create_publisher(
            RsuStateArray,
            "/rsu/state",
            rsu_qos,
        )

        self.get_logger().info(
            "RSU RT Solver started.\n"
            "Subscribing /rsu/target and /hardware_interface/state.\n"
            "Publishing /rsu/solution and /rsu/state."
        )

    def _accept_target_order(self, seq: int, stamp) -> bool:
        if seq != 0:
            if self._last_seq is None:
                self._last_seq = int(seq)
                return True

            if int(seq) <= int(self._last_seq):
                return False

            self._last_seq = int(seq)
            return True

        if self._last_stamp is None:
            self._last_stamp = stamp
            return True

        if stamp_to_ns(stamp) <= stamp_to_ns(self._last_stamp):
            return False

        self._last_stamp = stamp
        return True

    def _on_both_foot_request(self, msg: RsuTarget):
        if not self._accept_target_order(msg.seq, msg.header.stamp):
            self.get_logger().warn(
                "Received /rsu/target with non-increasing seq/stamp. Ignoring."
            )
            return

        l_roll = float(msg.left_roll)
        l_pitch = float(msg.left_pitch)

        # 오른발은 내부 solver convention에 맞추기 위해 부호 반전
        r_roll = float(msg.right_roll) * -1.0
        r_pitch = float(msg.right_pitch) * -1.0

        l_prev = self.prev_alpha_2d[0, :].copy()
        r_prev = self.prev_alpha_2d[1, :].copy()

        l_res = self.solver.solve(l_roll, l_pitch, l_prev)
        r_res = self.solver.solve(r_roll, r_pitch, r_prev)

        left_ok = bool(l_res.feasible)
        right_ok = bool(r_res.feasible)

        if left_ok:
            self.prev_alpha_2d[0, :] = np.array(l_res.alpha, dtype=np.float64).reshape(2,)
        elif not self.hold_alpha_on_infeasible:
            self.prev_alpha_2d[0, :] = 0.0

        if right_ok:
            self.prev_alpha_2d[1, :] = np.array(r_res.alpha, dtype=np.float64).reshape(2,)
        elif not self.hold_alpha_on_infeasible:
            self.prev_alpha_2d[1, :] = 0.0

        out = RsuSolution()
        out.header.stamp = msg.header.stamp
        out.seq = msg.seq

        out.left_actuator_1 = float(self.prev_alpha_2d[0, 0])
        out.left_actuator_2 = float(self.prev_alpha_2d[0, 1])

        # 외부 HW convention으로 되돌리기 위해 오른발 actuator 부호 반전
        out.right_actuator_1 = float(self.prev_alpha_2d[1, 0]) * -1.0
        out.right_actuator_2 = float(self.prev_alpha_2d[1, 1]) * -1.0

        out.feasible = bool(left_ok and right_ok)

        self.pub_both_foot_solution.publish(out)

    def _on_motor_state(self, msg: MotorStateArray):
        required_ids = {
            self.motor_state["left_ac1"]["id"],
            self.motor_state["left_ac2"]["id"],
            self.motor_state["right_ac1"]["id"],
            self.motor_state["right_ac2"]["id"],
        }

        msg_ids = {int(st.motor_id) for st in msg.states}
        missing_ids = required_ids - msg_ids

        if missing_ids:
            self.get_logger().error(
                f"Missing motor IDs in /hardware_interface/state: {missing_ids}. "
                f"Expected: {required_ids}, Got: {msg_ids}"
            )
            return

        id_to_state = {int(st.motor_id): st for st in msg.states}

        try:
            self.motor_state["left_ac1"]["pos"] = float(
                id_to_state[self.motor_state["left_ac1"]["id"]].position
            )
            self.motor_state["left_ac1"]["vel"] = float(
                id_to_state[self.motor_state["left_ac1"]["id"]].velocity
            )

            self.motor_state["left_ac2"]["pos"] = float(
                id_to_state[self.motor_state["left_ac2"]["id"]].position
            )
            self.motor_state["left_ac2"]["vel"] = float(
                id_to_state[self.motor_state["left_ac2"]["id"]].velocity
            )

            self.motor_state["right_ac1"]["pos"] = float(
                id_to_state[self.motor_state["right_ac1"]["id"]].position
            )
            self.motor_state["right_ac1"]["vel"] = float(
                id_to_state[self.motor_state["right_ac1"]["id"]].velocity
            )

            self.motor_state["right_ac2"]["pos"] = float(
                id_to_state[self.motor_state["right_ac2"]["id"]].position
            )
            self.motor_state["right_ac2"]["vel"] = float(
                id_to_state[self.motor_state["right_ac2"]["id"]].velocity
            )

        except (KeyError, IndexError, AttributeError) as e:
            self.get_logger().error(f"Failed to parse motor state: {e}")
            return

        stamp_ns = stamp_to_ns(msg.header.stamp)

        if self._last_motor_state_stamp_ns is None:
            self._last_motor_state_stamp_ns = stamp_ns
            return

        dt = (stamp_ns - self._last_motor_state_stamp_ns) * 1e-9
        self._last_motor_state_stamp_ns = stamp_ns

        if not np.isfinite(dt) or dt <= 0.0:
            self.get_logger().warn(
                f"Invalid dt from /hardware_interface/state: dt={dt}. Ignoring message."
            )
            return

        left_motor_pos = np.array([
            self.motor_state["left_ac1"]["pos"],
            self.motor_state["left_ac2"]["pos"],
        ], dtype=np.float64)

        left_motor_vel = np.array([
            self.motor_state["left_ac1"]["vel"],
            self.motor_state["left_ac2"]["vel"],
        ], dtype=np.float64)

        right_motor_pos = np.array([
            self.motor_state["right_ac1"]["pos"],
            self.motor_state["right_ac2"]["pos"],
        ], dtype=np.float64)

        right_motor_vel = np.array([
            self.motor_state["right_ac1"]["vel"],
            self.motor_state["right_ac2"]["vel"],
        ], dtype=np.float64)

        left_state, left_q, left_qd = self._estimate_one_foot(
            estimator=self.left_estimator,
            motor_pos=left_motor_pos,
            motor_vel=left_motor_vel,
            dt=dt,
            mirror=False,
        )

        right_state, right_q, right_qd = self._estimate_one_foot(
            estimator=self.right_estimator,
            motor_pos=right_motor_pos,
            motor_vel=right_motor_vel,
            dt=dt,
            mirror=True,
        )

        out = RsuStateArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.rsu_state_frame_id
        out.seq = self._rsu_state_seq
        self._rsu_state_seq += 1

        out.q_dot.left_rsu_roll = float(left_qd[0])
        out.q_dot.left_rsu_pitch = float(left_qd[1])
        out.q_dot.right_rsu_roll = float(right_qd[0])
        out.q_dot.right_rsu_pitch = float(right_qd[1])

        out.q.left_rsu_roll = float(left_q[0])
        out.q.left_rsu_pitch = float(left_q[1])
        out.q.right_rsu_roll = float(right_q[0])
        out.q.right_rsu_pitch = float(right_q[1])

        out.feasible = bool(
            left_state.feasible and right_state.feasible and
            left_state.valid and right_state.valid
        )

        self.pub_rsu_state.publish(out)

        if (not left_state.valid) or (not right_state.valid):
            self.get_logger().warn(
                "[RSU estimator] invalid state | "
                f"L(feasible={left_state.feasible}, valid={left_state.valid}, "
                f"res={left_state.residual_norm:.3e}, cond={left_state.condJ:.3f}, "
                f"sigma_min={left_state.sigma_min:.3e}) | "
                f"R(feasible={right_state.feasible}, valid={right_state.valid}, "
                f"res={right_state.residual_norm:.3e}, cond={right_state.condJ:.3f}, "
                f"sigma_min={right_state.sigma_min:.3e})"
            )
        elif left_state.degraded or right_state.degraded:
            self.get_logger().warn(
                "[RSU estimator] degraded state | "
                f"L(res={left_state.residual_norm:.3e}, cond={left_state.condJ:.3f}, "
                f"sigma_min={left_state.sigma_min:.3e}) | "
                f"R(res={right_state.residual_norm:.3e}, cond={right_state.condJ:.3f}, "
                f"sigma_min={right_state.sigma_min:.3e})"
            )

    def _estimate_one_foot(self, estimator, motor_pos, motor_vel, dt, mirror=False):
        motor_pos = np.array(motor_pos, dtype=np.float64)
        motor_vel = np.array(motor_vel, dtype=np.float64)

        if mirror:
            motor_pos = -motor_pos
            motor_vel = -motor_vel

        state = estimator.update(motor_pos, motor_vel, dt)

        q = np.array(state.q_rel, dtype=np.float64)
        qd = np.array(state.qd_rel, dtype=np.float64)

        if mirror:
            q = -q
            qd = -qd

        return state, q, qd


def main(args=None):
    rclpy.init(args=args)
    node = RSURtSolverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
