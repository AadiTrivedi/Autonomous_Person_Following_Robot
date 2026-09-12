#!/usr/bin/env python3
#Node 3 — Person Re-Identification

#Subscribes to: /detector/image /detections
#Publishes to: /tracked_target 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import numpy as np
import time

import torch
from torchreid.reid.utils import FeatureExtractor


class ReIDNode(Node):
    def __init__(self):
        super().__init__("reid")

        self.declare_parameter("similarity_threshold", 0.6)

        self.declare_parameter("enrollment_duration", 5.0)

        self.sim_thresh = self.get_parameter("similarity_threshold").value
        self.enroll_duration = self.get_parameter("enrollment_duration").value

        self.extractor = FeatureExtractor(
            model_name="osnet_x0_25",
            model_path="",  
            device="cpu",       
        )

        self.bridge = CvBridge()
        self.template_embedding = None
        self.enrollment_embeddings = []

        self.enrollment_start = time.time()

        self.enrolled = False

        img_sub = Subscriber(self, Image, "/detector/image")
        det_sub = Subscriber(self, Detection2DArray, "/detections")

        self.sync = ApproximateTimeSynchronizer(
            [img_sub, det_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.on_sync)

        self.target_pub = self.create_publisher(Detection2D, "/tracked_target", 10)

        self.get_logger().info(
            f"Re-ID node started, stand in front of the camera for "
            f"{self.enroll_duration}s to enroll"
        )

    def crop_person(self, frame, det):
        h, w = frame.shape[:2]

        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        bw = det.bbox.size_x
        bh = det.bbox.size_y

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def compute_embedding(self, crop):
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        features = self.extractor(crop_rgb)

        return features[0].cpu().numpy()

    def cosine_similarity(self, vec_a, vec_b):
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def on_sync(self, img_msg, det_msg):
        frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        if not self.enrolled:
            self.handle_enrollment(frame, det_msg)
        else:
            self.handle_tracking(frame, det_msg)

    def handle_enrollment(self, frame, det_msg):
        elapsed = time.time() - self.enrollment_start

        detections = det_msg.detections

        if len(detections) == 1:
            crop = self.crop_person(frame, detections[0])
            if crop is not None and crop.size > 0:
                emb = self.compute_embedding(crop)
                self.enrollment_embeddings.append(emb)
                self.get_logger().info(
                    f"Enrollment: captured {len(self.enrollment_embeddings)} "
                    f"samples ({elapsed:.1f}s / {self.enroll_duration}s)"
                )
        elif len(detections) == 0:
            self.get_logger().warn(
                "Enrollment: no person detected",
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().warn(
                f"Enrollment: {len(detections)} people in frame — ",
                throttle_duration_sec=1.0,
            )

        if elapsed >= self.enroll_duration:
            if len(self.enrollment_embeddings) >= 5:
                template = np.mean(self.enrollment_embeddings, axis=0)

                norm = np.linalg.norm(template)
                if norm > 0:
                    template = template / norm

                self.template_embedding = template
                self.enrolled = True
                self.get_logger().info(
                    f"Enrollment complete template built from "
                    f"{len(self.enrollment_embeddings)} samples"
                )
            else:
                self.get_logger().warn(
                    f"Only {len(self.enrollment_embeddings)} samples "
                    f"collected extending enrollment by "
                    f"{self.enroll_duration}s. Stand in front of camera."
                )
                self.enrollment_start = time.time()

    def handle_tracking(self, frame, det_msg):
        best_det = None
        best_sim = -1.0

        for det in det_msg.detections:
            crop = self.crop_person(frame, det)
            if crop is None or crop.size == 0:
                continue

            emb = self.compute_embedding(crop)
            sim = self.cosine_similarity(emb, self.template_embedding)

            if sim > best_sim:
                best_sim = sim
                best_det = det

        if best_det is not None and best_sim >= self.sim_thresh:
            if best_det.results:
                best_det.results[0].hypothesis.score = float(best_sim)

            self.target_pub.publish(best_det)
            self.get_logger().debug(
                f"Target found similarity: {best_sim:.3f}"
            )
        else:
            self.get_logger().debug(
                f"Target not found (best similarity: {best_sim:.3f}, "
                f"threshold: {self.sim_thresh})",
                throttle_duration_sec=2.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = ReIDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()