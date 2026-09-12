#!/usr/bin/env python3
#Draws bounding boxes 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer


class VizNode(Node):
    def __init__(self):
        super().__init__("viz")
        self.bridge = CvBridge()

        img_sub = Subscriber(self, Image, "/detector/image")
        det_sub = Subscriber(self, Detection2DArray, "/detections")

        self.sync = ApproximateTimeSynchronizer(
            [img_sub, det_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.on_sync)

    def on_sync(self, img_msg, det_msg):
        frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        for det in det_msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            conf = det.results[0].hypothesis.score if det.results else 0.0

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"person {conf:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2,
            )

        cv2.imshow("Detections", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()