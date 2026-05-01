#!/usr/bin/env python3

import math
import numpy as np

from roa_interfaces.msg import RsuSolution
from rsu_manager.util.rsu_solver import RSUParams, RSUSolver
from rsu_manager.util.rsu_state_estimator import RSUStateEstimator, RSUStateEstimatorConfig


def deg2rad(d):
    return d * math.pi / 180.0


def rad2deg(r):
    return r * 180.0 / math.pi


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm3(v) -> float:
    return float(np.linalg.norm(v))


def unit3(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = norm3(v)
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return v / n


def quat_to_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    uu = unit3(u, eps)
    vv = unit3(v, eps)
    d = float(np.dot(uu, vv))
    d = clamp(d, -1.0, 1.0)
    return math.acos(d)


class RSUCore:
    """
    공통 RSU 파라미터 선언 및 solver 생성 담당.
    Debug 노드와 RT 노드가 모두 이 클래스를 사용한다.
    """

    def __init__(self, node):
        self.node = node

        a_W_flat = node.declare_parameter(
            "a_W_mm_flat",
            [0.0, 36.0, 170.0,
             0.0, -36.0, 82.0],
        ).value

        b_F_flat = node.declare_parameter(
            "b_F_mm_flat",
            [-20.0, 36.0, 16.0,
             -20.0, -36.0, 16.0],
        ).value

        c_list = node.declare_parameter("c_mm", [30.0, -30.0]).value
        r_list = node.declare_parameter("r_mm", [154.0, 66.0]).value
        psi_list = node.declare_parameter(
            "psi_rad",
            [deg2rad(90.0), deg2rad(-90.0)],
        ).value

        if len(a_W_flat) != 6:
            raise RuntimeError(f"a_W_mm_flat must have length 6, got {len(a_W_flat)}")
        if len(b_F_flat) != 6:
            raise RuntimeError(f"b_F_mm_flat must have length 6, got {len(b_F_flat)}")
        if len(c_list) != 2 or len(r_list) != 2 or len(psi_list) != 2:
            raise RuntimeError(
                f"c_mm/r_mm/psi_rad must have length 2 "
                f"(got c={len(c_list)}, r={len(r_list)}, psi={len(psi_list)})"
            )

        self.a_W = np.array(a_W_flat, dtype=np.float64).reshape(2, 3)
        self.b_F = np.array(b_F_flat, dtype=np.float64).reshape(2, 3)
        self.c = np.array(c_list, dtype=np.float64).reshape(2,)
        self.r = np.array(r_list, dtype=np.float64).reshape(2,)
        self.psi = np.array(psi_list, dtype=np.float64).reshape(2,)

        params = RSUParams(
            a_W=self.a_W,
            b_F=self.b_F,
            c=self.c,
            r=self.r,
            psi=self.psi,
        )
        self.solver = RSUSolver(params)

        node.get_logger().info(
            "RSU Params:\n"
            f"  a_W_mm_flat: {a_W_flat}\n"
            f"  b_F_mm_flat: {b_F_flat}\n"
            f"  c_mm: {c_list}\n"
            f"  r_mm: {r_list}\n"
            f"  psi_rad: {psi_list}"
        )


class RSUEstimatorFactory:
    """
    RT 노드에서 좌/우 estimator를 만들기 위한 공통 팩토리.
    """

    def __init__(self, node, solver):
        self.node = node
        self.solver = solver

        self.jac_lambda = node.declare_parameter("jac_lambda", 1e-6).value
        self.jac_h = node.declare_parameter("jac_h", 5e-5).value
        self.beta_jac = node.declare_parameter("beta_jac", 1.0).value
        self.vel_lpf_tau = node.declare_parameter("vel_lpf_tau", 0.0).value
        self.motor_vel_lpf_tau = node.declare_parameter("motor_vel_lpf_tau", 0.0).value

        node.get_logger().info(
            "RSU Estimator Params:\n"
            f"  jac_lambda: {self.jac_lambda}\n"
            f"  jac_h: {self.jac_h}\n"
            f"  beta_jac: {self.beta_jac}\n"
            f"  vel_lpf_tau: {self.vel_lpf_tau}\n"
            f"  motor_vel_lpf_tau: {self.motor_vel_lpf_tau}"
        )

    def make(self):
        cfg = RSUStateEstimatorConfig(
            max_iter=5,
            lambda_pos=1e-6,
            jac_h=self.jac_h,
            jac_lambda=self.jac_lambda,
            beta_jac=self.beta_jac,
            vel_lpf_tau=self.vel_lpf_tau,
            motor_vel_lpf_tau=self.motor_vel_lpf_tau,
            cond_warn=50.0,
            cond_fail=200.0,
            sigma_min_thresh=1e-5,
            q_init=np.array([0.0, 0.0]),
        )

        return RSUStateEstimator(
            solver=self.solver,
            cfg=cfg,
        )
