#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener, TransformException


def to_point(x, y, z):
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p


def norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def quat_from_z_to_dir(dx, dy, dz):
    q = Quaternion()

    length = norm3(dx, dy, dz)
    if length < 1e-12:
        q.w = 1.0
        return q

    dx /= length
    dy /= length
    dz /= length

    vx = -dy
    vy = dx
    vz = 0.0

    s = math.sqrt((1.0 + dz) * 2.0)

    if s < 1e-12:
        q.w = 0.0
        q.x = 1.0
        q.y = 0.0
        q.z = 0.0
        return q

    q.w = s * 0.5
    inv_s = 1.0 / s
    q.x = vx * inv_s
    q.y = vy * inv_s
    q.z = vz * inv_s

    return q


class RGBA:
    def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self.r = float(r)
        self.g = float(g)
        self.b = float(b)
        self.a = float(a)


class RSULinkPlotter(Node):
    def __init__(self):
        super().__init__('rsu_link_plotter')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.world_frame = self.declare_parameter(
            'world_frame', 'base_link'
        ).value
        self.publish_rate_hz = self.declare_parameter(
            'publish_rate_hz', 60.0
        ).value
        self.radius = self.declare_parameter(
            'radius', 0.004
        ).value
        self.timeout_sec = self.declare_parameter(
            'tf_timeout_sec', 0.05
        ).value

        self.c1_frame = self.declare_parameter(
            'c1_frame', 'point_c1_1'
        ).value
        self.c2_frame = self.declare_parameter(
            'c2_frame', 'point_c2_1'
        ).value
        self.u1_frame = self.declare_parameter(
            'u1_frame', 'point_u1_1'
        ).value
        self.u2_frame = self.declare_parameter(
            'u2_frame', 'point_u2_1'
        ).value

        self.target_len_1_m = self.declare_parameter(
            'target_len_1_m', 0.1695
        ).value
        self.target_len_2_m = self.declare_parameter(
            'target_len_2_m', 0.0810
        ).value

        self.link1_color = RGBA(
            self.declare_parameter('link1_r', 1.0).value,
            self.declare_parameter('link1_g', 0.1).value,
            self.declare_parameter('link1_b', 0.1).value,
            self.declare_parameter('link1_a', 0.9).value,
        )

        self.link2_color = RGBA(
            self.declare_parameter('link2_r', 0.1).value,
            self.declare_parameter('link2_g', 0.4).value,
            self.declare_parameter('link2_b', 1.0).value,
            self.declare_parameter('link2_a', 0.9).value,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            'rsu_links',
            10
        )

        period = 1.0 / max(1.0, self.publish_rate_hz)
        self.timer = self.create_timer(period, self.on_timer)

        self.last_warn_time = self.get_clock().now()
        self.last_info_time = self.get_clock().now()

        self.get_logger().info(
            f'RSU Link Plotter started. world_frame={self.world_frame}'
        )
        self.get_logger().info(
            f'Frames: C1={self.c1_frame} U1={self.u1_frame} | '
            f'C2={self.c2_frame} U2={self.u2_frame}'
        )
        self.get_logger().info(
            f'Targets: L1={self.target_len_1_m * 1000.0:.1f} mm, '
            f'L2={self.target_len_2_m * 1000.0:.1f} mm'
        )

    def throttled_warn(self, msg, period_sec=2.0):
        now = self.get_clock().now()
        if (now - self.last_warn_time).nanoseconds * 1e-9 >= period_sec:
            self.get_logger().warn(msg)
            self.last_warn_time = now

    def throttled_info(self, msg, period_sec=1.0):
        now = self.get_clock().now()
        if (now - self.last_info_time).nanoseconds * 1e-9 >= period_sec:
            self.get_logger().info(msg)
            self.last_info_time = now

    def lookup_point(self, target_frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                target_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.timeout_sec)
            )

            p = Point()
            p.x = tf.transform.translation.x
            p.y = tf.transform.translation.y
            p.z = tf.transform.translation.z
            return True, p

        except TransformException as ex:
            self.throttled_warn(
                f'TF lookup failed: {self.world_frame} -> {target_frame} : {ex}'
            )
            return False, Point()

    def make_cylinder(self, marker_id, a, b, radius, ns, color):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD

        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        length = norm3(dx, dy, dz)

        m.pose.position = to_point(
            (a.x + b.x) * 0.5,
            (a.y + b.y) * 0.5,
            (a.z + b.z) * 0.5
        )
        m.pose.orientation = quat_from_z_to_dir(dx, dy, dz)

        m.scale.x = radius * 2.0
        m.scale.y = radius * 2.0
        m.scale.z = max(1e-6, length)

        m.color.r = color.r
        m.color.g = color.g
        m.color.b = color.b
        m.color.a = color.a

        return m

    def make_delete_marker(self, marker_id, ns):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.DELETE
        return m

    def on_timer(self):
        ok_c1, c1 = self.lookup_point(self.c1_frame)
        ok_c2, c2 = self.lookup_point(self.c2_frame)
        ok_u1, u1 = self.lookup_point(self.u1_frame)
        ok_u2, u2 = self.lookup_point(self.u2_frame)

        if not (ok_c1 and ok_c2 and ok_u1 and ok_u2):
            return

        l1 = norm3(c1.x - u1.x, c1.y - u1.y, c1.z - u1.z)
        l2 = norm3(c2.x - u2.x, c2.y - u2.y, c2.z - u2.z)

        self.throttled_info(
            f'L1={l1 * 1000.0:.1f} mm '
            f'(err {(l1 - self.target_len_1_m) * 1000.0:.1f}), '
            f'L2={l2 * 1000.0:.1f} mm '
            f'(err {(l2 - self.target_len_2_m) * 1000.0:.1f})'
        )

        arr = MarkerArray()
        marker_id = 0
        hide_threshold_m = 0.002

        err1 = abs(l1 - self.target_len_1_m)
        if err1 < hide_threshold_m:
            arr.markers.append(
                self.make_cylinder(
                    marker_id, c1, u1,
                    self.radius,
                    'rsu_link_1',
                    self.link1_color
                )
            )
        else:
            arr.markers.append(
                self.make_delete_marker(marker_id, 'rsu_link_1')
            )
        marker_id += 1

        err2 = abs(l2 - self.target_len_2_m)
        if err2 < hide_threshold_m:
            arr.markers.append(
                self.make_cylinder(
                    marker_id, c2, u2,
                    self.radius,
                    'rsu_link_2',
                    self.link2_color
                )
            )
        else:
            arr.markers.append(
                self.make_delete_marker(marker_id, 'rsu_link_2')
            )

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = RSULinkPlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()