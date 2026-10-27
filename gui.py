#!/usr/bin/python

import sys
import os
import io
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pyparsing import results
import qdarktheme

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGridLayout, QHBoxLayout, QComboBox,
    QLineEdit, QLabel, QPushButton, QStackedLayout, QTabWidget,
    QVBoxLayout, QWidget, QMessageBox
)

# Classifier classes
from src.ensemble import EnsembleClassifier
from src.haar import HaarCascadeClassifier
from src.colors import retrieve_dominant_colors_canny, get_color_luminance
from ultralytics import YOLO

YOLOV8_WEIGHTS_PATH = "./pretrained/best_latest.pt"

def letterbox_image(image, target_size=(640, 480)):
    """
    Resizes image to target_size while maintaining aspect ratio using padding.
    """
    ih, iw = image.shape[:2]
    th, tw = target_size
    
    # Calculate the scaling factor to fit the image within target_size
    scale = min(tw / iw, th / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    # Resize the image using the scale factor
    # INTER_AREA is best for shrinking; INTER_CUBIC is better for enlarging
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Create a solid black canvas of the target size
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    
    # Calculate offsets to center the image on the canvas
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    
    # Paste the resized image onto the center of the canvas
    canvas[dy:dy+nh, dx:dx+nw] = image_resized
    
    return canvas

class YOLOv8Thread(QObject):
    result = pyqtSignal(object)
    annotated = pyqtSignal(np.ndarray)
    done = pyqtSignal()

    def __init__(
        self,
        image_path,
        model_path=YOLOV8_WEIGHTS_PATH,
        parent=None,
        color_retrieval_strategy="canny"
    ):
        QThread.__init__(self, parent)

        self.image_path = image_path
        self.model_path = model_path
        self.model = YOLO(self.model_path)

        self.color_retrieval_strategy = color_retrieval_strategy

    def run(self):
        raw_image = cv2.imread(self.image_path)
        if raw_image is None:
            self.done.emit()
            return

        # Standardize resolution here so indicators in colors.py scale correctly
        image = letterbox_image(raw_image, (640, 480))

        results = self.model.predict(image)[0]
        final_dominant_colors = []
        display_image = image.copy()

        if results.boxes:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_bird = image[y1:y2, x1:x2]

                # COLOR ANALYSIS
                colors, shaded_crop = retrieve_dominant_colors_canny(
                    cropped_bird,
                    k=6
                )

                # Replace region with shaded result
                display_image[y1:y2, x1:x2] = shaded_crop

                if colors:
                    final_dominant_colors.append(colors)

        # EMIT RESULTS
        self.result.emit(final_dominant_colors)
        self.annotated.emit(display_image)
        self.done.emit()
        
class VideoThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray)
    update_results_signal = pyqtSignal(object)
    done_averaging = pyqtSignal(bool)
    averaging_results = pyqtSignal(dict)

    def __init__(self, camera_index=0, target_classifier="ensemble", averaging_mode=False):
        super().__init__()
        self.camera_index = camera_index
        self.target_classifier = target_classifier
        self.averaging_mode = averaging_mode
        self.clf = None
        self.haar_clf = None

    def switch_classifier(self, target_classifier):
        self.target_classifier = target_classifier

    def toggle_averaging(self, value: bool):
        self.averaging_mode = value

    def is_raspberry_pi(self):
        try:
            if os.name != 'posix': return False
            chips = ('BCM2708', 'BCM2709', 'BCM2711', 'BCM2835', 'BCM2836')
            with io.open('/proc/cpuinfo', 'r') as cpuinfo:
                for line in cpuinfo:
                    if line.startswith('Hardware'):
                        _, value = line.strip().split(':', 1)
                        if value.strip() in chips: return True
        except Exception: pass
        return False
    
    def run(self):
        is_rpi = self.is_raspberry_pi()
        self.clf = EnsembleClassifier()
        self.haar_clf = HaarCascadeClassifier()
        
        if is_rpi:
            from picamera2 import Picamera2
            from libcamera import Transform
            picam2 = Picamera2()
            config = {"format": "RGB888", "size": (640, 480)}
            picam2.configure(picam2.create_preview_configuration(main=config, transform=Transform(vflip=1, hflip=1)))
            picam2.start()
            while True:
                if self.averaging_mode: self.run_averaging_mode(picam2, is_rpi)
                else:
                    frame = picam2.capture_array()
                    self.detect(self.clf, self.haar_clf, frame, frame.any())
        else:
            cap = cv2.VideoCapture(self.camera_index)
            try:
                while True:
                    if self.averaging_mode: self.run_averaging_mode(cap, is_rpi)
                    else:
                        ret, frame = cap.read()
                        if ret: self.detect(self.clf, self.haar_clf, frame, ret)
            finally:
                cap.release()

    def run_averaging_mode(self, cap, is_rpi: bool):
        labels, probs, ctr = [], [], 0
        while ctr < 60:
            if is_rpi: frame = cap.capture_array(); ret = frame.any()
            else: ret, frame = cap.read()
            results = self.detect(self.clf, self.haar_clf, frame, ret)
            if results:
                labels.append(results['label'])
                probs.append(results['probability'].item())
            else:
                labels.append(None); probs.append(0)
            ctr += 1
            if not self.averaging_mode: break

        self.averaging_mode = False
        self.done_averaging.emit(True)
        overalls = self.get_overalls(labels, probs)
        highest_scoring = max(overalls, key=overalls.get)
        self.averaging_results.emit({highest_scoring: overalls[highest_scoring]})

    def get_overalls(self, labels, probabilities):
        results = {}
        for label in set(labels):
            probs = [p for idx, p in enumerate(probabilities) if labels[idx] == label]
            results[label] = sum(probs) / 60
        return results

    def detect(self, ensemble_clf, haar_clf, frame, signal):
        if signal and self.target_classifier == "ensemble":
            clf_output = ensemble_clf.classify(frame, display=False, as_matlike=True, print_results=False)
            self.update_frame_signal.emit(clf_output['frame'])
            self.update_results_signal.emit(clf_output['full_result'])
            return clf_output['full_result']

        if signal and self.target_classifier == "haar":
            clf_output = haar_clf.classify(frame, display=False, as_matlike=True)
            self.update_frame_signal.emit(clf_output['frame'])
            self.update_results_signal.emit(clf_output['result'])
            return clf_output['result']

class App(QWidget):
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.current_frame = None
        self.active_classifier = "ensemble"
        self.color_retrieval_strategy = "canny"
        self.averaging_mode = False
        self.setWindowTitle("Cockatiel Species Classifier")
        self.setup_interface()
        self.vthread = self.create_video_thread(camera_index)
        self.vthread.start()

    def setup_interface(self):
        main = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.detector_ui(), "Detector")
        self.tabs.addTab(self.color_density_analyzer_ui(), "Color Density")
        self.tabs.addTab(self.static_detector_ui(), "Static Image")
        main.addWidget(self.tabs)
        self.show()

    def static_detector_ui(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.static_image_label = QLabel("No Image Selected")
        self.static_image_label.setAlignment(Qt.AlignCenter)
        self.static_image_label.setMinimumSize(400, 400)
        layout.addWidget(self.static_image_label)
            
        upload_btn = QPushButton("Upload Cockatiel Image")
        upload_btn.clicked.connect(self.select_and_process_static)
        layout.addWidget(upload_btn)

        self.static_result_text = QLabel("Results will appear here...")
        self.static_result_text.setFont(QFont("Manrope", 12))
        layout.addWidget(self.static_result_text, alignment=Qt.AlignCenter)
        return tab

    def select_and_process_static(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd(), "Images (*.jpg *.jpeg *.png)")
        if filename:
            temp_img = cv2.imread(filename)
            if temp_img is None: return
            
            # Use Letterboxing instead of simple cv2.resize
            standard_img = letterbox_image(temp_img, (640, 480))

            # Run Classification
            snet_result = self.vthread.clf.squeezenet_clf.test(standard_img, as_matlike=True)
            results = self.vthread.clf.classify(standard_img, display=False, as_matlike=True)
            
            # Update UI
            pixmap = self.cv2pixmap(results['frame'])
            self.static_image_label.setPixmap(pixmap.scaled(self.static_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            if snet_result:
                self.static_result_text.setText(f"Mutation Identified: {snet_result['label']} ({snet_result['probability']})")
            else:
                self.static_result_text.setText("Classification failed.")
    
    def detector_ui(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.active_frame = QLabel()
        layout.addWidget(self.active_frame, alignment=Qt.AlignCenter)
        layout.addLayout(self.setup_classifier_switches_view())
        
        self.results_stack = QStackedLayout()
        self.results_stack.addWidget(self.convert_layout_to_widget(self.setup_ensemble_results_view()))
        self.results_stack.addWidget(self.convert_layout_to_widget(self.setup_haar_results_view()))
        layout.addLayout(self.results_stack)
        return tab
    
    def pass_to_output_box(self, results):
        """
        Displays the original image and color measurements.
        No bounding boxes or mutation labels are processed here.
        """
        if results.get("annotated") is not None:
            pixmap = self.cv2pixmap(results["annotated"])
            self.color_preview_box.setPixmap(pixmap.scaled(
                self.color_preview_box.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        try:
            if self.output_box is not None:
                self.clear_layout(self.output_box)
        except RuntimeError:
            return

        if results.get("results") and len(results["results"]) > 0:
            for color_data in results["results"][0]:
                hex_code, percentage = color_data
                
                color_block = QWidget()
                block_layout = QHBoxLayout(color_block)
                
                luminance = get_color_luminance(hex_code)
                text_color = "black" if luminance > 186 else "white"

                color_block.setStyleSheet(f"""
                    background-color: {hex_code}; 
                    border: 1px solid #555;
                    border-radius: 5px;
                    margin-bottom: 2px;
                """)
                
                dot = QLabel()

                dot.setFixedSize(20, 20)

                dot_border = "white" if text_color == "white" else "black"

                dot.setStyleSheet(f"""
                    background-color: {hex_code};
                    border: 2px solid {dot_border};
                    border-radius: 10px;
                    min-width: 20px;
                    max-width: 20px;
                """)
                
                text_label = QLabel(f"{hex_code} ({percentage:.2f}%)")
                text_label.setStyleSheet(f"color: {text_color}; font-weight: bold; border: none;")
                
                block_layout.addWidget(dot)
                block_layout.addWidget(text_label, 1)
                block_layout.setContentsMargins(10, 5, 10, 5)
                block_layout.setSpacing(10)
                
                self.output_box.addWidget(color_block)
            
            self.color_process_label.setText("Analysis Complete")

    def color_density_analyzer_ui(self):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        io_row = QHBoxLayout()
        
        # 1. Preview Box (Stays on the left)
        self.color_preview_box = QLabel("No Image Selected")
        self.color_preview_box.setAlignment(Qt.AlignCenter)
        self.color_preview_box.setMinimumSize(400, 400)
        self.color_preview_box.setStyleSheet("border: 1px solid #333;")
        
        # 2. Results Panel (The vertical lock)
        self.results_container = QWidget()
        # SET A FIXED HEIGHT: This prevents vertical resizing
        self.results_container.setFixedHeight(600) 
        self.results_container.setFixedWidth(250)
        
        # Layout inside the fixed container
        self.output_box = QVBoxLayout(self.results_container)
        self.output_box.setAlignment(Qt.AlignTop)
        self.output_box.setContentsMargins(5, 5, 5, 5)
        self.output_box.setSpacing(2) # Tighter spacing for better fit
        
        io_row.addWidget(self.color_preview_box, stretch=2)
        io_row.addWidget(self.results_container, stretch=1)

        # 3. Control Buttons
        self.color_upload_btn = QPushButton("Upload Image for Color Analysis")
        self.color_upload_btn.clicked.connect(self.select_and_analyze_color)

        vbox.addLayout(io_row)
        vbox.addWidget(self.color_upload_btn)
        
        self.color_process_label = QLabel("Ready for analysis")
        vbox.addWidget(self.color_process_label, alignment=Qt.AlignCenter)

        tab.setLayout(vbox)
        return tab

    def capture_frame(self):
        self.preview_box.hide(); self.capture_box.show()
        if self.current_frame is None: return
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        height, width, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, width, height, ch * width, QImage.Format_RGB888)
        qt_img.save("captured_image.jpg", "JPG")
        self.run_yolov8("captured_image.jpg", self.color_retrieval_strategy)
        self.process_label.setText("Frame captured successfully")

    def show_camera(self):
        self.process_label.setText("Showing camera feed...")
        self.preview_box.show(); self.capture_box.hide()
        self.clear_layout(self.output_box)

    def convert_layout_to_widget(self, layout):
        widget = QWidget(); widget.setLayout(layout)
        return widget

    def setup_ensemble_results_view(self):
        view = QVBoxLayout(); grid = QGridLayout()
        self.ensemble_results_label = QLabel("Unidentified"); self.ensemble_results_prob = QLabel("0.00")
        grid.addWidget(QLabel("Label"), 0, 0); grid.addWidget(QLabel("Prob"), 0, 1)
        grid.addWidget(self.ensemble_results_label, 1, 0); grid.addWidget(self.ensemble_results_prob, 1, 1)
        view.addLayout(grid)
        self.toggle_averaging_btn = QPushButton("Averaging Mode (60 frames)")
        self.toggle_averaging_btn.setCheckable(True); self.toggle_averaging_btn.clicked.connect(self.toggle_averaging)
        view.addWidget(self.toggle_averaging_btn)
        return view

    def toggle_averaging(self):
        self.averaging_mode = not self.averaging_mode
        self.vthread.toggle_averaging(self.averaging_mode)

    def setup_haar_results_view(self):
        grid = QGridLayout()
        self.haar_results_label = QLabel("Unidentified Bird"); self.haar_results_prob = QLabel("0.00")
        grid.addWidget(QLabel("Label"), 0, 0); grid.addWidget(QLabel("Score"), 0, 1)
        grid.addWidget(self.haar_results_label, 1, 0); grid.addWidget(self.haar_results_prob, 1, 1)
        return grid

    def setup_classifier_switches_view(self):
        layout = QHBoxLayout()
        self.toggle_ensemble = QPushButton("Ensemble"); self.toggle_ensemble.setCheckable(True)
        self.toggle_ensemble.setChecked(True); self.toggle_ensemble.clicked.connect(lambda: self.switch_classifier("ensemble"))
        self.toggle_haar = QPushButton("Haar"); self.toggle_haar.setCheckable(True)
        self.toggle_haar.clicked.connect(lambda: self.switch_classifier("haar"))
        layout.addWidget(self.toggle_ensemble); layout.addWidget(self.toggle_haar)
        return layout

    def switch_classifier(self, target):
        self.active_classifier = target
        self.vthread.switch_classifier(target)
        self.results_stack.setCurrentIndex(0 if target == "ensemble" else 1)
        self.toggle_ensemble.setChecked(target == "ensemble")
        self.toggle_haar.setChecked(target == "haar")

    def create_video_thread(self, camera_index):
        v = VideoThread(camera_index)
        v.update_frame_signal.connect(self.update_frame)
        v.update_results_signal.connect(self.update_results)
        v.done_averaging.connect(self.done_averaging)
        v.averaging_results.connect(self.display_averaging_results)
        return v

    def run_yolov8(self, img_path, strategy):
        self.yolo_thread = QThread()
        self.yolo = YOLOv8Thread(img_path, color_retrieval_strategy=strategy)
        results = {"annotated": None, "results": None}
        self.yolo.result.connect(lambda v: results.update({"results": v}))
        self.yolo.annotated.connect(lambda v: results.update({"annotated": v}))
        self.yolo.done.connect(self.yolo_thread.quit)
        self.yolo_thread.started.connect(self.yolo.run)
        self.yolo_thread.finished.connect(lambda: self.pass_to_output_box(results))
        self.yolo_thread.start()

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        self.current_frame = frame
        self.update_active_frame(frame)

    @pyqtSlot(bool)
    def done_averaging(self, status):
        if status: self.toggle_averaging_btn.setChecked(False)

    @pyqtSlot(dict)
    def display_averaging_results(self, results):
        if results:
            res_cls = list(results.keys())[0]
            QMessageBox.information(self, "Result", f"The cockatiel is a {res_cls} ({results[res_cls]:.2%})")

    @pyqtSlot(object)
    def update_results(self, results):
        if results:
            if self.active_classifier == "ensemble":
                self.ensemble_results_label.setText(results.get('label', 'None'))
                self.ensemble_results_prob.setText(str(results.get('probability', '0.00')))
            else:
                self.haar_results_label.setText(results.get('label', 'None'))
                self.haar_results_prob.setText(str(results.get('score', '0.00')))

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def select_and_analyze_color(self):
        """
        Opens a file dialog and runs YOLOv8 + Color Extraction on a static image.
        """
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd(), "Images (*.jpg *.jpeg *.png)")
        
        if filename:
            try:
                if hasattr(self, 'output_box') and self.output_box is not None:
                    self.clear_layout(self.output_box)
            except RuntimeError:
                pass

            self.color_process_label.setText(f"Processing: {os.path.basename(filename)}...")
            
            pixmap = QPixmap(filename)
            if hasattr(self, 'color_preview_box'):
                self.color_preview_box.setPixmap(pixmap.scaled(
                    self.color_preview_box.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            self.run_yolov8(filename, self.color_retrieval_strategy)

    def pass_to_output_box(self, results):
        """
        Display color analyzer results to the output box. 
        Uses colors.get_color_luminance to ensure text readability.
        """
        # 1. Update the image display with the YOLO-annotated frame
        if results.get("annotated") is not None:
            # Scale the annotated image to fit your static display box
            pixmap = self.cv2pixmap(results["annotated"])
            self.color_preview_box.setPixmap(pixmap.scaled(
                self.color_preview_box.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))

        # 2. Safety check: Clear the previous results and verify current layout
        try:
            if self.output_box is not None:
                self.clear_layout(self.output_box)
            else:
                return
        except RuntimeError:
            return

        # 3. Process and display each dominant color found by YOLO + colors.py
        if results.get("results") and len(results["results"]) > 0:
            # results["results"][0] contains the list of (hex, percentage) tuples
            for color_data in results["results"][0]:
                hex_code, percentage = color_data
                
                # Create a visual block for the color
                color_block = QLabel()
                
                # Use your colors.py get_color_luminance to pick the best text color
                # If luminance > 186 (bright color), use black text; otherwise white
                luminance = get_color_luminance(hex_code)
                text_color = "black" if luminance > 186 else "white"

                # Apply styling: background color, border, and readable text
                color_block.setStyleSheet(f"""
                    background-color: {hex_code}; 
                    color: {text_color}; 
                    border: 1px solid #555;
                    border-radius: 5px;
                    padding: 12px;
                    font-weight: bold;
                    font-size: 14px;
                    margin-bottom: 2px;
                """)
                
                color_block.setText(f"{hex_code} ({percentage:.2f}%)")
                color_block.setAlignment(Qt.AlignCenter)
                
                # Add the block to your side-panel layout
                self.output_box.addWidget(color_block)
            
            self.color_process_label.setText("Analysis Complete")
        else:
            self.color_process_label.setText("No bird detected to analyze colors.")

    def on_select_strategy_change(self):
        strategies = ["canny"]
        self.color_retrieval_strategy = strategies[self.select_strategy.currentIndex()]

    def update_active_frame(self, frame):
        """
        Updates the Pixmap/Image displayed on the UI.
        Only updates the Detector tab to prevent crashing other static tabs.
        """
        pix = self.cv2pixmap(frame)

        # Only update the first tab (Detector) with the live camera feed
        if self.tabs.currentIndex() == 0: 
            self.active_frame.setPixmap(pix)

    def cv2pixmap(self, cv2_frame):
        f = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        h, w, c = f.shape
        return QPixmap.fromImage(QImage(f.data, w, h, c * w, QImage.Format_RGB888))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", '-c', type=int, default=0)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    win = App(camera_index=args.camera)
    sys.exit(app.exec_())