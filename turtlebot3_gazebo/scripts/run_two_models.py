import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',  # TurtleBot3 camera topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        try:
            self.fire_detect_model = YOLO('~/argos_ws/src/fire_and_smoke_trained.pt')
            self.coco_model = YOLO('yolov8m.pt')
        except Exception as e:
            self.get_logger().error(f"Error loading YOLO models: {e}")
        self.fire_confidence_threshold = 0.3
        self.coco_confidence_threshold = 0.5

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        try:
            coco_results = self.coco_model.predict(frame, conf=self.coco_confidence_threshold, iou=0.45)
            fire_results = self.fire_detect_model.predict(frame, conf=self.fire_confidence_threshold, iou=0.45)
        except Exception as e:
            self.get_logger().error(f"Error during prediction: {e}")
            return

        annotated_frame = frame.copy()

        for box in coco_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence
            cls = self.coco_model.names[int(box.cls[0])]  # Class name
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(annotated_frame, f'{cls} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for box in fire_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence
            cls = self.fire_detect_model.names[int(box.cls[0])]  # Class name
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
            cv2.putText(annotated_frame, f'{cls} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('YOLOv8 Detection', annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    yolov8_node = YOLOv8Node()
    rclpy.spin(yolov8_node)
    yolov8_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
