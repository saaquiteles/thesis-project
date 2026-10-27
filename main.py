import argparse
from src.haar import HaarCascadeClassifier
from src.optimizer import HaarOptimizer
from src.ensemble import EnsembleClassifier 


def run_task(args):
    if args.task == "live":
        clf = HaarCascadeClassifier() if args.haar else EnsembleClassifier()
        clf.classify_live(camera_index=args.camera)
        return None

    if args.task == "detect":
        clf = HaarCascadeClassifier() if args.haar else EnsembleClassifier()

        if not args.haar:
            print("Using Ensemble Method")
        else:
            print("Using Haar Classifier")

        clf.classify(args.image)
        return None

    if args.task == "optimize":
        optimizer = HaarOptimizer(args.dataset)
        optimizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("task", nargs='?', type=str, default="live")
    parser.add_argument("--image", '-i', help="Path to an input image")
    parser.add_argument("--dataset", '-d', help="Path to a dataset directory")
    parser.add_argument("--camera", '-c', nargs='?', type=int, default=0, help="Index of the camera that will be used by cv2.VideoCapture()")
    parser.add_argument("--haar", default=False, action=argparse.BooleanOptionalAction, help="If provided, this will only use the Haar Classifier instead of Ensemble Method (Haar + SqueezeNet).")

    args = parser.parse_args()

    if not args.image and args.task == "detect":
        print("ERROR: Please provide an input image with the -i option")
        raise SystemExit(1)

    if not args.dataset and args.task == "optimize":
        print("ERROR: Please provide a path to a dataset directory")
        raise SystemExit(1)

    run_task(args)
