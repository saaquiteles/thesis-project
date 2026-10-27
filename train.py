"""
Script for fine-tuning/custom-training SqueezeNet
"""

from src.squeezenet import SqueezeNet

if __name__ == "__main__":
    clf = SqueezeNet()
    
    clf.generate_labels()
    clf.create_dataset()
    
    trained_model = clf.train()
