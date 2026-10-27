from .haar import HaarCascadeClassifier
from .squeezenet import SqueezeNet
import cv2
import os

DEFAULT_SQUEEZENET_CLF_PATH = "pretrained/model_20-epochs_1740585416.1928105.pt"

class EnsembleClassifier(HaarCascadeClassifier):

    def __init__(self, squeezenet_clf_path=DEFAULT_SQUEEZENET_CLF_PATH):
        super().__init__()
        self.squeezenet_clf = SqueezeNet()
        self.squeezenet_clf.load_custom_model(self.parse_model_path(squeezenet_clf_path))
        self.prev_bbox = None  # Stores [x, y, w, h]
        self.alpha = 0.4     # Smoothing factor (0.1 = slow/smooth, 0.5 = fast/jittery)
        self.bbox_colors = {"bird": (0, 255, 0),
                            "cinnamon": (255, 0, 0),
                            "lutino": (0, 0, 255),
                            "pearl": (0, 255, 255),
                            "pied": (255, 255, 0),
                            "whiteface": (255, 0, 255)}

    def parse_model_path(self, model_path):
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, '..', model_path)
        return filepath

    # override
    def classify(self, image, display=True, as_matlike=False, print_results=True):
        """
        Runs the entire object detection + classification process with centered ROI.
        """
        frame = image
        result_label = ""

        if not as_matlike:
            frame = cv2.imread(image)

        # 1. Get actual frame dimensions
        h_frame, w_frame = frame.shape[:2]

        # 2. Define and Center the ROI (e.g., 400x400)
        # Using max/min ensures the ROI stays within the camera frame boundaries
        roi_w, roi_h = 450, 450
        self.roi_x = max(0, (w_frame - roi_w) // 2)
        self.roi_y = max(0, (h_frame - roi_h) // 2)
        self.roi_w = min(roi_w, w_frame)
        self.roi_h = min(roi_h, h_frame)

        # 3. Perform the crop safely
        roi_frame = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]
        
        if roi_frame.size == 0:
            return {"frame": frame, "result": "error", "full_result": None}

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Haar Cascade (detecting within the ROI)
        birds = self.detect(self.bird_clf, gray)
        cockatiels = self.detect(self.cockatiel_clf, gray)

        bird_det = self.get_detections("Bird", birds)
        cockatiel_det = self.get_detections("Cockatiel", cockatiels)

        haar_result = self.parse_result(bird_det, cockatiel_det)

        snet_result = None

        if haar_result["label"] == "cockatiel":
            if not as_matlike:
                snet_result = self.squeezenet_clf.test(image, print_results=print_results)
            else:
                snet_result = self.squeezenet_clf.test(frame, as_matlike=True, print_results=print_results)

            if print_results:
                print("[SqueezeNet Result] Label: {}, Probability: {}, Time: {}".format(
                    snet_result["label"], snet_result["probability"], snet_result["inference_time"]))
            
            # Use updated bounding box logic
            frame = self.attach_bounding_boxes(frame, haar_result["bboxes"], snet_result["label"])
            result_label = snet_result["label"]
        else:
            frame = self.attach_bounding_boxes(frame, haar_result["bboxes"], "bird")
            result_label = "bird"

        # Optional: Draw a subtle guide box for the user to see the detection area
        cv2.rectangle(frame, (self.roi_x, self.roi_y), 
                      (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (100, 100, 100), 1)

        if display:
            self.display_cv_image(frame)

        return {
            "frame": frame,
            "result": result_label,
            "full_result": snet_result
        }

    # override
    def attach_bounding_boxes(self, frame, bboxes, label):
    # Use the logic from the previous turn to handle a single smoothed box
        if not bboxes:
            return frame

        # 1. Pick the best box
        best_bbox = max(bboxes, key=lambda b: b[2] * b[3])
        curr_x, curr_y, curr_w, curr_h = best_bbox

        # 2. Map coordinates back to the full frame (IMPORTANT)
        abs_x = curr_x + self.roi_x
        abs_y = curr_y + self.roi_y

        if self.prev_bbox is None:
            self.prev_bbox = [abs_x, abs_y, curr_w, curr_h]
        else:
            for i, val in enumerate([abs_x, abs_y, curr_w, curr_h]):
                self.prev_bbox[i] = int((val * self.alpha) + (self.prev_bbox[i] * (1 - self.alpha)))

        x, y, w, h = self.prev_bbox
        color = self.bbox_colors.get(label, (255, 0, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
        
        return frame
    
    def select_and_process_static(self):
    # 1. Open File Dialog
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd(), "Images (*.jpg *.jpeg *.png)")
        
        if filename:
            # 2. Read Image
            image = cv2.imread(filename)
            if image is None:
                return

            # 3. Run Detection (Re-using your existing Ensemble class)
            # self.vthread.clf is initialized in VideoThread; for static, we can use it or a local instance
            results = self.vthread.clf.classify(image, display=False, as_matlike=True)

            # 4. Update UI with the Annotated Image
            pixmap = self.cv2pixmap(results['frame'])
            self.static_image_label.setPixmap(pixmap.scaled(self.static_image_label.size(), Qt.KeepAspectRatio))

            # 5. Show Classification Data
            if results['full_result']:
                label = results['full_result']['label']
                prob = results['full_result']['probability']
                self.static_result_text.setText(f"Detected Mutation: {label} ({prob})")
            else:
                self.static_result_text.setText("No Cockatiel Detected in Image.")

    
    # override
    def classify_live(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            self.classify(frame, display=False, as_matlike=True)
            cv2.imshow("Cockatiel Variety Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
