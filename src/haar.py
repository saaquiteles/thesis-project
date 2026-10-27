import cv2
import os

DEFAULT_BIRD_CLF_PATH = "../pretrained/bird15stages.xml"
DEFAULT_COCKATIEL_CLF_PATH = "../pretrained/cockatiel12stages.xml"

class HaarCascadeClassifier:

    def __init__(self, bird_clf_path=DEFAULT_BIRD_CLF_PATH, cockatiel_clf_path=DEFAULT_COCKATIEL_CLF_PATH):
        self.bird_clf = self.import_model(bird_clf_path)
        self.cockatiel_clf = self.import_model(cockatiel_clf_path)
        roi_width = 400
        roi_height = 400
      
        self.bird_threshold = 2.5e-05
        self.cockatiel_threshold = 2.0e-05

        self.roi_x = (640 - roi_width) // 2
        self.roi_y = (480 - roi_height) // 2
        self.roi_w = roi_width
        self.roi_h = roi_height

        self.scaleFactor = 1.12
        self.minNeighbors = 8
        self.minSize = (224, 224)

        self.bird_bbox_color = (0, 255, 0)
        self.cockatiel_bbox_color = (0, 0, 255)

    
    def import_model(self, model_path):
        """
        Method for importing a Cascade Classifier model from XML
        """
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, model_path)
        return cv2.CascadeClassifier(filepath)

    
    def get_roi(self, image, x, y, w, h):
        """
        Helper method for returning the Region of Interest (ROI)
        """
        return image[y:y+h, x:x+w]


    def detect(self, classifier, frame):
        """
        Method which simply performs a call to cv2.CascadeClassifier.detectMultiScale().
        In short, it performs the detection process using Haar Cascade.
        """
        return classifier.detectMultiScale(frame,
                                           scaleFactor=self.scaleFactor,
                                           minNeighbors=self.minNeighbors,
                                           minSize=self.minSize)


    def get_detections(self, label, outputs):
        """
        Calculates the confidence/scores for each detected instance.
        """
        confidence_scores = []
        bbox_regions = []
        labels = []

        for (x, y, w, h) in outputs:
            confidence = len(outputs) / (w * h)
            threshold = self.cockatiel_threshold if label.lower() == "cockatiel" else self.bird_threshold

            if confidence >= threshold:
                confidence_scores.append(confidence)
                bbox_regions.append((x, y, w, h))
                labels.append(label.lower())

        return {
            "scores": confidence_scores,
            "bboxes": bbox_regions,
            "labels": labels
        }


    def attach_bounding_boxes(self, frame, bboxes, label):
        """
        Method for attaching bounding boxes based on provided lists
        of bounding box coordinates and labels.
        """
        for (x, y, w, h) in bboxes:
            bbox_color = self.cockatiel_bbox_color if label.lower() == "cockatiel" else self.bird_bbox_color
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), bbox_color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.68, bbox_color, 2)

        return frame


    def mean(self, numbers):
        """
        Helper method for calculating Mean.
        """
        return sum(numbers) / len(numbers)


    def parse_result(self, bird_det, cockatiel_det): 
        """
        Determines the final detected object based on input scores.
        """
        bird_scores = bird_det["scores"]
        cockatiel_scores = cockatiel_det["scores"]

        result = {
            "score": 0,
            "label": "",
            "bboxes": []
        }

        def update_result(score, label, bboxes):
            result["score"] = score[0]
            result["label"] = label
            result["bboxes"] = bboxes

        if len(bird_scores) == 0 and len(cockatiel_scores) == 0:
            # immediately return if no instances are detected.
            # this is to also speed up live mode execution.
            return result
        elif len(bird_scores) > len(cockatiel_scores):
            update_result(bird_scores, "bird", bird_det["bboxes"])
        elif len(cockatiel_scores) > len(bird_scores):
            update_result(cockatiel_scores, "cockatiel", cockatiel_det["bboxes"])
        else:
            if self.mean(cockatiel_scores) < self.mean(bird_scores):
                update_result(cockatiel_scores, "cockatiel", cockatiel_det["bboxes"])
            else:
                update_result(bird_scores, "bird", bird_det["bboxes"])

        return result


    def classify(self, image, display=True, as_matlike=False):
        """
        Runs the entire object detection/classification process.
        """
        frame = image

        if not as_matlike:
            frame = cv2.imread(image)

        roi_frame = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        birds = self.detect(self.bird_clf, gray)
        cockatiels = self.detect(self.cockatiel_clf, gray)

        bird_det = self.get_detections("Bird", birds)
        cockatiel_det = self.get_detections("Cockatiel", cockatiels)

        result = self.parse_result(bird_det, cockatiel_det)
        frame = self.attach_bounding_boxes(frame, result["bboxes"], result["label"])

        if display:
            self.display_cv_image(frame)

        return {
            "frame": frame,
            "bird_scores": bird_det["scores"],
            "cockatiel_scores": cockatiel_det["scores"],
            "result": result
        }


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


    def display_cv_image(self, image_mat):
        """
        Method to display resulting image matrices using OpenCV
        """
        cv2.imshow('image', image_mat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
