#!/usr/bin/env python3
# Node 2 Person Detector
# Subscribes to /camera/image_raw, runs YOLOv8n, publishes detections to /detections


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    BoundingBox2D,
    ObjectHypothesisWithPose,
)
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
from ultralytics import YOLO


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("person_class_id", 0)  # COCO class 0 = person
        model_path = self.get_parameter("model_path").value
        self.conf_thresh = self.get_parameter("confidence_threshold").value
        self.person_class = self.get_parameter("person_class_id").value

        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self.on_frame, 10
        )
        self.det_pub = self.create_publisher(Detection2DArray, "/detections", 10)

        self.img_pub = self.create_publisher(Image, "/detector/image", 10)

        self.get_logger().info(
            f"Detector ready — model={model_path}, conf>={self.conf_thresh}"
        )

    def on_frame(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model(frame, verbose=False)[0]

        det_array = Detection2DArray()
        det_array.header = msg.header

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != self.person_class or conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            det = Detection2D()
            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.center.theta = 0.0
            det.bbox.size_x = w
            det.bbox.size_y = h

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls_id)
            hyp.hypothesis.score = conf
            det.results.append(hyp)

            det_array.detections.append(det)

        self.det_pub.publish(det_array)
        self.img_pub.publish(msg)  
        self.get_logger().debug(
            f"Published {len(det_array.detections)} person detections"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()