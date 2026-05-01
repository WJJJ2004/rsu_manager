from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("rsu_manager")
    param_file = PathJoinSubstitution([pkg, "config", "rsu_rt.yaml"])

    return LaunchDescription([
        Node(
            package="rsu_manager",
            executable="rt_solver_node",
            name="rt_solver_node",
            output="screen",
            parameters=[
                param_file,
            ],
            arguments=["--ros-args", "--log-level", "info"],
        ),
    ])