#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from rsu_manager.util.core import (
    RSUCore,
    stamp_to_ns,
    norm3,
    quat_to_R,
    angle_between,
    rad2deg,
)


class RSUDebugSolverNode(Node):
    """
    Debug 전용 RSU solver 노드.

    입력:
      /request_to_solver : geometry_msgs/Vector3Stamped
        vector.x = roll
        vector.y = pitch

    출력:
      /solver_answer : geometry_msgs/Vector3Stamped
        vector.x = alpha1
        vector.y = alpha2
        vector.z = feasible flag

      /joint_states : sensor_msgs/JointState
        RViz/TF 확인용 joint state
    """

    def __init__(self):
        super().__init__("rsu_debug_solver_node")

        self.core = RSUCore(self)
        self.solver = self.core.solver

        self.joint_ankle_pitch = str(self.declare_parameter("joint_ankle_pitch", "ankle_pitch").value)
        self.joint_ankle_roll = str(self.declare_parameter("joint_ankle_roll", "ankle_roll").value)
        self.joint_upper_crank = str(self.declare_parameter("joint_upper_crank", "upper_crank").value)
        self.joint_lower_crank = str(self.declare_parameter("joint_lower_crank", "lower_crank").value)

        self.hold_alpha_on_infeasible = bool(
            self.declare_parameter("hold_alpha_on_infeasible", True).value
        )

        self.prev_alpha_1d = np.array([0.0, 0.0], dtype=np.float64)

        self.roll = 0.0
        self.pitch = 0.0
        self.alpha1 = 0.0
        self.alpha2 = 0.0
        self._last_stamp = None
        self.last_debug_t = 0.0

        # TF-based hardware sanity check
        self.world_frame = str(self.declare_parameter("world_frame", "base_link").value)
        self.tf_timeout_sec = float(self.declare_parameter("tf_timeout_sec", 0.05).value)

        self.c1_frame = str(self.declare_parameter("c1_frame", "point_c1_1").value)
        self.c2_frame = str(self.declare_parameter("c2_frame", "point_c2_1").value)
        self.u1_frame = str(self.declare_parameter("u1_frame", "point_u1_1").value)
        self.u2_frame = str(self.declare_parameter("u2_frame", "point_u2_1").value)

        self.target_len_1_m = float(self.core.r[0]) / 1000.0
        self.target_len_2_m = float(self.core.r[1]) / 1000.0
        self.len_tol_m = float(self.declare_parameter("len_tol_m", 0.002).value)
        self.ang_min_deg = float(self.declare_parameter("ang_min_deg", 70.0).value)
        self.ang_max_deg = float(self.declare_parameter("ang_max_deg", 110.0).value)
        self.gate_publish_by_tf_check = bool(
            self.declare_parameter("gate_publish_by_tf_check", True).value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._tf_check_ok = False
        self._tf_check_diag = "not computed yet"

        self.pub_joint_state = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )
        self.pub_solver_respond = self.create_publisher(
            Vector3Stamped,
            "/solver_answer",
            10,
        )
        self.sub_solver_request = self.create_subscription(
            Vector3Stamped,
            "/request_to_solver",
            self._on_rpy,
            10,
        )

        self.get_logger().info(
            "RSU Debug Solver started.\n"
            "Subscribing /request_to_solver, publishing /solver_answer and /joint_states.\n"
            f"TF check: world={self.world_frame}, frames: "
            f"C1={self.c1_frame} U1={self.u1_frame} | "
            f"C2={self.c2_frame} U2={self.u2_frame}\n"
            f"Targets: L1={self.target_len_1_m * 1000:.1f}mm, "
            f"L2={self.target_len_2_m * 1000:.1f}mm, "
            f"tol={self.len_tol_m * 1000:.1f}mm, "
            f"ang=[{self.ang_min_deg:.1f},{self.ang_max_deg:.1f}]deg"
        )

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self._last_stamp
        msg.name = [
            self.joint_ankle_pitch,
            self.joint_ankle_roll,
            self.joint_upper_crank,
            self.joint_lower_crank,
        ]
        msg.position = [
            float(self.pitch),
            float(self.roll),
            float(self.alpha1),
            float(self.alpha2),
        ]
        self.pub_joint_state.publish(msg)

        now = time.time()
        if now - self.last_debug_t > 1.0:
            self.last_debug_t = now
            self.get_logger().info(
                f"RP=(roll={self.roll:+.3f}, pitch={self.pitch:+.3f}) rad | "
                f"alpha=({self.alpha1:+.3f}, {self.alpha2:+.3f}) rad | "
                f"TF_OK={self._tf_check_ok}"
            )

    def publish_solver_respond(self, feasible: bool):
        msg = Vector3Stamped()
        msg.header.stamp = self._last_stamp

        msg.vector.x = float(self.alpha1)
        msg.vector.y = float(self.alpha2)
        msg.vector.z = 1.0 if feasible else 0.0

        self.pub_solver_respond.publish(msg)

    def _lookup_tf(self, target_frame: str):
        try:
            return self.tf_buffer.lookup_transform(
                self.world_frame,
                target_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
        except Exception:
            return None

    def _tf_to_p_R(self, tfmsg):
        t = tfmsg.transform.translation
        q = tfmsg.transform.rotation
        p = np.array([t.x, t.y, t.z], dtype=np.float64)
        R = quat_to_R(q.x, q.y, q.z, q.w)
        return p, R

    def _in_ang_range(self, ang_deg: float) -> bool:
        return self.ang_min_deg <= ang_deg <= self.ang_max_deg

    def tf_hardware_safetycheck(self):
        tf_c1 = self._lookup_tf(self.c1_frame)
        tf_u1 = self._lookup_tf(self.u1_frame)
        tf_c2 = self._lookup_tf(self.c2_frame)
        tf_u2 = self._lookup_tf(self.u2_frame)

        if tf_c1 is None or tf_u1 is None or tf_c2 is None or tf_u2 is None:
            self._tf_check_ok = False
            self._tf_check_diag = "TF missing"
            raise RuntimeError(
                "TF lookup failed for frames: "
                f"C1={'OK' if tf_c1 else 'MISSING'} "
                f"U1={'OK' if tf_u1 else 'MISSING'} "
                f"C2={'OK' if tf_c2 else 'MISSING'} "
                f"U2={'OK' if tf_u2 else 'MISSING'}"
            )

        pc1, Rc1 = self._tf_to_p_R(tf_c1)
        pu1, Ru1 = self._tf_to_p_R(tf_u1)
        pc2, Rc2 = self._tf_to_p_R(tf_c2)
        pu2, Ru2 = self._tf_to_p_R(tf_u2)

        L1 = norm3(pc1 - pu1)
        L2 = norm3(pc2 - pu2)

        err1 = abs(L1 - self.target_len_1_m)
        err2 = abs(L2 - self.target_len_2_m)
        ok_len = err1 <= self.len_tol_m and err2 <= self.len_tol_m

        y_c1 = Rc1 @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_u1 = Ru1 @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_c2 = Rc2 @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_u2 = Ru2 @ np.array([0.0, 1.0, 0.0], dtype=np.float64)

        ang_c1_deg = rad2deg(angle_between(y_c1, pu1 - pc1))
        ang_u1_deg = rad2deg(angle_between(y_u1, pc1 - pu1))
        ang_c2_deg = rad2deg(angle_between(y_c2, pu2 - pc2))
        ang_u2_deg = rad2deg(angle_between(y_u2, pc2 - pu2))

        ok_ang = (
            self._in_ang_range(ang_c1_deg) and
            self._in_ang_range(ang_u1_deg) and
            self._in_ang_range(ang_c2_deg) and
            self._in_ang_range(ang_u2_deg)
        )

        if not ok_len:
            raise RuntimeError(
                "TF HW safety check failed: SHAFT LENGTH OUT OF RANGE\n"
                f"  L1 = {L1:.4f} m (target={self.target_len_1_m:.4f} m, err={err1 * 1000:.1f} mm)\n"
                f"  L2 = {L2:.4f} m (target={self.target_len_2_m:.4f} m, err={err2 * 1000:.1f} mm)\n"
                f"  pc1 = {pc1.tolist()}\n"
                f"  pu1 = {pu1.tolist()}\n"
                f"  pu1-pc1 = {(pu1 - pc1).tolist()}\n"
                f"  pc2 = {pc2.tolist()}\n"
                f"  pu2 = {pu2.tolist()}\n"
                f"  pu2-pc2 = {(pu2 - pc2).tolist()}\n"
                f"  angles (deg): C1={ang_c1_deg:.1f}, U1={ang_u1_deg:.1f}, "
                f"C2={ang_c2_deg:.1f}, U2={ang_u2_deg:.1f}"
            )

        if not ok_ang:
            raise RuntimeError(
                "TF HW safety check failed: BALL JOINT OUT OF SAFETY RANGE "
                f"(C1={ang_c1_deg:.1f}deg, U1={ang_u1_deg:.1f}deg, "
                f"C2={ang_c2_deg:.1f}deg, U2={ang_u2_deg:.1f}deg, "
                f"allowed=[{self.ang_min_deg:.1f}, {self.ang_max_deg:.1f}]deg)"
            )

        return True

    def _on_rpy(self, msg: Vector3Stamped):
        if self._last_stamp is None:
            self._last_stamp = msg.header.stamp
        elif stamp_to_ns(msg.header.stamp) <= stamp_to_ns(self._last_stamp):
            self.get_logger().warn(
                "Received /request_to_solver with timestamp older than or equal to last processed command. Ignoring."
            )
            return
        self._last_stamp = msg.header.stamp

        res = self.solver.solve(
            float(msg.vector.x),
            float(msg.vector.y),
            self.prev_alpha_1d,
        )

        if bool(res.feasible):
            a_solver = np.array(res.alpha, dtype=np.float64).reshape(2,)
            self.alpha1 = float(a_solver[0])
            self.alpha2 = float(a_solver[1])
            self.prev_alpha_1d[:] = a_solver
            self.roll = float(msg.vector.x)
            self.pitch = float(msg.vector.y)
        else:
            if self.hold_alpha_on_infeasible:
                self.get_logger().warn("[IK] infeasible -> hold previous state")
            else:
                self.get_logger().warn("[IK] infeasible -> no update")

        self.publish_joint_states()

        if self.gate_publish_by_tf_check:
            try:
                self._tf_check_ok = self.tf_hardware_safetycheck()
                if not self._tf_check_ok:
                    self._tf_check_diag = "TF hardware safety check failed"
            except Exception as e:
                self._tf_check_ok = False
                self.get_logger().error(f"TF hardware safety check FAILED: {e}")
        else:
            self._tf_check_ok = True

        self.publish_solver_respond(bool(res.feasible) and self._tf_check_ok)


def main(args=None):
    rclpy.init(args=args)
    node = RSUDebugSolverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
