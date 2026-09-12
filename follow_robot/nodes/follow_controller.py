#!/usr/bin/env python3
#Node 5 Follow Controller

#Subscribes to: /tracked_target /fall_alert /scan

#Publishes to: /cmd_vel

#Node following logic:
#If Person left of frame center then turn left
#If Person right of frame center then turn right
#If Person's bounding box is smaller than threshold then drive faster
#If Person's bounding box is larger than threshold then slow down


import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan

import time
import math


class FollowController(Node):
    def __init__(self):
        super().__init__("follow_controller")
        self.declare_parameter("frame_width", 640.0)
        self.declare_parameter("frame_height", 480.0)
        self.declare_parameter("max_linear_speed", 0.3)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("target_bbox_height", 300.0)
        self.declare_parameter("min_bbox_height", 50.0)
        self.declare_parameter("dead_zone", 0.1)
        self.declare_parameter("obstacle_stop_distance", 0.4)
        self.declare_parameter("target_lost_timeout", 2.0)
        self.frame_w = self.get_parameter("frame_width").value
        self.frame_h = self.get_parameter("frame_height").value
        self.max_lin = self.get_parameter("max_linear_speed").value
        self.max_ang = self.get_parameter("max_angular_speed").value
        self.target_h = self.get_parameter("target_bbox_height").value
        self.min_h = self.get_parameter("min_bbox_height").value
        self.dead_zone = self.get_parameter("dead_zone").value
        self.obs_dist = self.get_parameter("obstacle_stop_distance").value
        self.lost_timeout = self.get_parameter("target_lost_timeout").value
        self.last_target_time = None
        self.fall_detected = False
        self.obstacle_ahead = False


        self.target_sub = self.create_subscription(
            Detection2D, "/tracked_target", self.on_target, 10
        )
        self.fall_sub = self.create_subscription(
            Bool, "/fall_alert", self.on_fall_alert, 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.on_scan, 10
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.latest_target = None

        self.get_logger().info("Follow controller ready")

    def on_target(self, msg: Detection2D):
        self.latest_target = msg
        self.last_target_time = time.time()

    def on_fall_alert(self, msg: Bool):
        self.fall_detected = msg.data
        if self.fall_detected:
            self.get_logger().error("FALL DETECTED — stopping robot")

    def on_scan(self, msg: LaserScan):
        num_readings = len(msg.ranges)
        if num_readings == 0:
            return

        angle_increment = msg.angle_increment

        half_cone_readings = int((math.pi / 6.0) / angle_increment)
        self.obstacle_ahead = False

        for i in range(half_cone_readings):
            if i < num_readings:
                r = msg.ranges[i]
                if msg.range_min < r < self.obs_dist:
                    self.obstacle_ahead = True
                    return

            j = num_readings - 1 - i
            if j >= 0 and j != i:
                r = msg.ranges[j]
                if msg.range_min < r < self.obs_dist:
                    self.obstacle_ahead = True
                    return

# Target fell: stop
    def control_loop(self):
        cmd = Twist()
        if self.fall_detected:
            self.cmd_pub.publish(cmd)
            return

# Target lost: stop
        if self.last_target_time is None:
            self.cmd_pub.publish(cmd)
            return

        elapsed = time.time() - self.last_target_time
        if elapsed > self.lost_timeout:
            self.get_logger().warn(
                f"Target lost for {elapsed:.1f}s — stopping",
                throttle_duration_sec=2.0,
            )
            self.cmd_pub.publish(cmd)
            return

        det = self.latest_target
        if det is None:
            self.cmd_pub.publish(cmd)
            return

        cx = det.bbox.center.position.x  
        bbox_h = det.bbox.size_y      

        frame_center_x = self.frame_w / 2.0
        error_x = (frame_center_x - cx) / frame_center_x
        if abs(error_x) < self.dead_zone:
            error_x = 0.0
        cmd.angular.z = error_x * self.max_ang
        if bbox_h < self.min_h:
            distance_factor = 1.0
        else:
            distance_factor = 1.0 - (bbox_h / self.target_h)
            distance_factor = max(0.0, min(1.0, distance_factor))

        cmd.linear.x = distance_factor * self.max_lin

        if self.obstacle_ahead:
            cmd.linear.x = 0.0
            self.get_logger().warn(
                "Obstacle detected, stopping forward motion",
                throttle_duration_sec=1.0,
            )

        self.cmd_pub.publish(cmd)

        self.get_logger().debug(
            f"cmd_vel: lin={cmd.linear.x:.2f} ang={cmd.angular.z:.2f} "
            f"| bbox_h={bbox_h:.0f} error_x={error_x:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = FollowController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()