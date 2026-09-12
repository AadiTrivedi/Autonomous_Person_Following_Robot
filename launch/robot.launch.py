#Launch file for real hardware operation.
#Starts all nodes: camera → detector → re-ID → pose → controller → motors


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    conf_arg = DeclareLaunchArgument(
        "confidence", default_value="0.5",
        description="YOLO detection confidence threshold"
    )

    follow_dist_arg = DeclareLaunchArgument(
        "follow_distance", default_value="300.0",
        description="Target bounding box height in pixels at ideal follow distance"
    )

#Camera (Node 1)
    camera = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="camera",
        output="log",
        parameters=[{"video_device": "/dev/video0"}],
        remappings=[("image_raw", "/camera/image_raw")],
    )

#Detector (Node 2)
    detector = Node(
        package="follow_robot",
        executable="detector_node",
        name="detector",
        output="screen",
        parameters=[
            {
                "model_path": "yolov8n.pt",
                "confidence_threshold": LaunchConfiguration("confidence"),
                "person_class_id": 0,
            }
        ],
    )

#Re-ID (Node 3)
    reid = Node(
        package="follow_robot",
        executable="reid_node",
        name="reid",
        output="screen",
        parameters=[
            {
                "similarity_threshold": 0.6,
                "enrollment_duration": 5.0,
            }
        ],
    )

#Pose and Fall Detection (Node 4)
    pose = Node(
        package="follow_robot",
        executable="pose_node",
        name="pose",
        output="screen",
        parameters=[
            {
                "model_path": "/home/aadi/ros2_ws/movenet_lightning.tflite",
                "fall_angle_threshold": 60.0,
                "fall_confirm_seconds": 3.0,
            }
        ],
    )

#Follow Controller (Node 5)
    controller = Node(
        package="follow_robot",
        executable="follow_controller",
        name="follow_controller",
        output="screen",
        parameters=[
            {
                "frame_width": 640.0,
                "frame_height": 480.0,
                "max_linear_speed": 0.3,
                "max_angular_speed": 1.0,
                "target_bbox_height": LaunchConfiguration("follow_distance"),
                "dead_zone": 0.1,
                "obstacle_stop_distance": 0.4,
                "target_lost_timeout": 2.0,
            }
        ],
    )

#Motor Driver (Node 6)
    motors = Node(
        package="follow_robot",
        executable="motor_driver",
        name="motor_driver",
        output="screen",
        parameters=[
            {
                "left_enable_pin": 18,
                "left_forward_pin": 23,
                "left_reverse_pin": 24,
                "right_enable_pin": 12,
                "right_forward_pin": 25,
                "right_reverse_pin": 22,
                "wheel_base": 0.17,
                "max_motor_speed": 0.3,
                "pwm_frequency": 1000,
            }
        ],
    )

    return LaunchDescription([
        conf_arg,
        follow_dist_arg,
        camera,
        detector,
        reid,
        pose,
        controller,
        motors,
    ])