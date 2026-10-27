## Update Log: 
- the software environment is on the raspberry pi 5, with the dependencies for running the RPLidar.
- The robot is following correctly, nerfed the speed for safety precautions, correctly recognizing when the person has fallen down.

## Demo Video

[demo video1 taken in a lab showcases the robots following ability](./Followbot-Vid1.mp4)

[demo video2 showcases the ability to recognize when a person a has fallen](./Followbot-Vid2.mov)

# FollowBot

A robot that follows you around and carries your stuff. Also detects if you fall and texts someone for help.

Built with ROS2 Humble, C++, and a Raspberry Pi 5.

How it works:

You stand in front of it, it locks onto you, and then it follows you while maintaining a set distance. If you fall down, it detects the fall using pose estimation and sends an SMS alert after 3 seconds.

System flow: 

Camera -> YOLOv8n (person detection) -> Re-ID (lock onto target) -> EKF (fuse with lidar) -> PID controller -> motors

ROS2 Nodes

All written in python. 

detector_node: runs YOLOv8n on each camera frame, publishes bounding boxes
tracker_node: re-ID matching to lock onto target + EKF that works together with RPLIDAR distance
fall_detector_node: MoveNet pose estimation, reports a fall if keypoints stay horizontal for 3+ seconds
follow_controller_node: PID controller that outputs velocity commands to maintain distance and heading
safety_node: reads RPLIDAR scan, overrides velocity to zero if anything is within the stop zone
motor_driver_node: converts velocity commands to motor PWM signals over serial to the Arduino
Hardware
Raspberry Pi 5 (8GB): runs everything
Slamtec RPLIDAR A1M8: 360° 2D lidar for distance + obstacle avoidance
Camera
LeiwanSoul 4WD aluminum chassis with TT motors
L298N motor driver + Arduino as serial bridge
Anker zoro 20,000mAh power bank

# Acknowledgments

Built in the Sensors and Actuators Lab, University of Maryland, College Park.
