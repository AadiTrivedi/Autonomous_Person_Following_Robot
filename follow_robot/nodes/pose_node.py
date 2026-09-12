#!/usr/bin/env python3
# Node 4 Pose Estimation & Fall Detection

#Subscribes to:/detector/image /tracked_target

#Publishes to: /person_state /fall_alert


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

import cv2
import numpy as np
import time

from ai_edge_litert.interpreter import Interpreter as tflite_Interpreter

#Keypoints outputted by Movenet
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class PoseNode(Node):
    def __init__(self):
        super().__init__("pose")

        self.declare_parameter("model_path", "movenet_lightning.tflite")
        self.declare_parameter("fall_angle_threshold", 60.0)
        self.declare_parameter("fall_confirm_seconds", 3.0)
        self.declare_parameter("min_keypoint_confidence", 0.3)

        model_path = self.get_parameter("model_path").value
        self.fall_angle_thresh = self.get_parameter("fall_angle_threshold").value
        self.fall_confirm_secs = self.get_parameter("fall_confirm_seconds").value
        self.min_kp_conf = self.get_parameter("min_keypoint_confidence").value
        self.interpreter = tflite_Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # MoveNet Lightning expects 192×192 input size.
        self.input_size = self.input_details[0]["shape"][1:3]  # (192, 192)

        self.bridge = CvBridge()
        self.prev_hip_y = None

        self.prev_frame_time = None
        self.fall_start_time = None
        self.alert_sent = False

# Subscribed to tracked target from re-ID
        self.img_sub = self.create_subscription(
            Image, "/detector/image", self.on_image, 10
        )

        self.target_sub = self.create_subscription(
            Detection2D, "/tracked_target", self.on_target, 10
        )

        self.state_pub = self.create_publisher(String, "/person_state", 10)
        self.alert_pub = self.create_publisher(Bool, "/fall_alert", 10)

        self.latest_frame = None

        self.get_logger().info("Pose and fall detection node ready")

    def on_image(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def on_target(self, msg):
        if self.latest_frame is None:
            return

        frame = self.latest_frame
        h, w = frame.shape[:2]

        cx = msg.bbox.center.position.x
        cy = msg.bbox.center.position.y
        bw = msg.bbox.size_x
        bh = msg.bbox.size_y

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))

        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]

        keypoints = self.run_movenet(crop)
        if keypoints is None:
            return

        state = self.classify_pose(keypoints)

        state_msg = String()
        state_msg.data = state
        self.state_pub.publish(state_msg)

        now = time.time()

        if state == "fallen":
            if self.fall_start_time is None:
                self.fall_start_time = now
                self.alert_sent = False
                self.get_logger().warn("Person appears to have fallen")

            elif (
                not self.alert_sent
                and (now - self.fall_start_time) >= self.fall_confirm_secs
            ):
                alert_msg = Bool()
                alert_msg.data = True
                self.alert_pub.publish(alert_msg)
                self.alert_sent = True
                self.get_logger().error(
                    f"FALL ALERT:  person has been down for "
                    f"{now - self.fall_start_time:.1f}s"
                )
        else:
            if self.fall_start_time is not None:
                self.get_logger().info("Person recovered from fallen state")
            self.fall_start_time = None
            self.alert_sent = False

    def run_movenet(self, crop):
        input_img = cv2.resize(crop, tuple(reversed(self.input_size)),
                               interpolation=cv2.INTER_LINEAR)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        input_tensor = np.expand_dims(input_img, axis=0)

        expected_dtype = self.input_details[0]["dtype"]
        if expected_dtype == np.float32:
            input_tensor = input_tensor.astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        keypoints = output[0][0]  # (17, 3) shape, each row is (y, x, confidence)

        return keypoints

    def classify_pose(self, keypoints):
        l_shoulder = keypoints[LEFT_SHOULDER]   # (y, x, conf)
        r_shoulder = keypoints[RIGHT_SHOULDER]
        l_hip = keypoints[LEFT_HIP]
        r_hip = keypoints[RIGHT_HIP]

        min_conf = min(
            l_shoulder[2], r_shoulder[2],
            l_hip[2], r_hip[2]
        )
        if min_conf < self.min_kp_conf:
            return "standing"  # if uncertain assume upright

        shoulder_mid_y = (l_shoulder[0] + r_shoulder[0]) / 2.0
        shoulder_mid_x = (l_shoulder[1] + r_shoulder[1]) / 2.0
        hip_mid_y = (l_hip[0] + r_hip[0]) / 2.0
        hip_mid_x = (l_hip[1] + r_hip[1]) / 2.0

        dy = hip_mid_y - shoulder_mid_y
        dx = hip_mid_x - shoulder_mid_x

        torso_angle_deg = abs(np.degrees(np.arctan2(dx, dy)))

        if torso_angle_deg > self.fall_angle_thresh:
            return "fallen"

        return "standing"


def main(args=None):
    rclpy.init(args=args)
    node = PoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()