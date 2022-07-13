from http.server import executable
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # package: package name
    # executable: name of executable file
    # name: arbitrary name give to node
    # remappings: remapping topic names, this MUST be an array of tuples
    # parameters: set parameters, the MUST be an array of dictionaries
    number_publisher_node = Node(
        package="ros2_pytorch",
        executable="ros2_pytorch",
        name="pytorch_node",
        parameters=[{"GPU": 1}]
    )

    ld.add_action(number_publisher_node)

    return ld