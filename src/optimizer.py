"""
Script for finding the optimal minimum threshold to
detect Cockatiels based on a target dataset.
"""
import os
from .haar import HaarCascadeClassifier

class HaarOptimizer:

    def __init__(self, dataset_path, label="cockatiel"):
        self.dataset_path = self.get_dataset_absolute_path(dataset_path)
        self.haar_clf = HaarCascadeClassifier()
        self.label = label


    def get_dataset_absolute_path(self, dataset_path):
        """
        Returns the absolute path to the target dataset path.
        """
        current_dir = os.getcwd()
        return os.path.join(current_dir, dataset_path)


    def run(self):
        """
        Runs the Haar Cascade Classifier against each image
        in the specified dataset path.
        """
        scores = []

        for filename in os.listdir(self.dataset_path):
            if filename.lower().endswith(('jpg', 'png', 'jpeg')):
                filepath = os.path.join(self.dataset_path, filename)
                output = self.haar_clf.classify(filepath, display=False)
                print("Bird Scores: " + str(output["bird_scores"]))
                print("Cockatiel Scores: " + str(output["cockatiel_scores"]))

                target_key = "".join([self.label.lower(), "_scores"])

                if len(output[target_key]) > 1:
                    scores.append(min(output[target_key]))

        optimal_value = min(scores)
        print("Optimal Value: " + optimal_value)

        return optimal_value
