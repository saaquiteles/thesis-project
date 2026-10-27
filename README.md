# Haar Cascade + SqueezeNet Classifier

## Installation

**Setup and activate Virtual Environment**

```sh
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux
.venv/bin/activate
```

**Install dependencies**

```sh
pip install -r requirements.txt
```

## How does this work?

This project is designed to perform Ensemble Learning (i.e. it combines two or more techniques in performing a machine learning task). It uses Haar Cascade for performing object detection and retrieving bounding boxes, while using SqueezeNet for classification (i.e. classifying cockatiels by species).

The Haar Classifier in this project is trained to classify birds and cockatiels.

The SqueezeNet Classifier is trained to classify five (5) different cockatiel species: Cinnamon, Lutino, Pearl, Pied, and Whiteface.

## Published Article

Link to published article: https://ebooks.iospress.nl/volumearticle/75521

## Live Mode

By default, `main.py` will run in **live mode**. In this case, the classifier will test each frame that passes through the camera.

```sh
python main.py
```

You can also tell `main.py` to run explicitly in live mode:

```sh
python main.py live
```

`main.py` is designed to use the Ensemble Classifier - which will use the Haar Classifier for detection and SqueezeNet for classification.

If you want to use only the Haar Classifier in live mode:

```sh
python main.py --haar
```

### Switching Cameras

Live mode relies on `cv2.VideoCapture()` and the index of the camera that will be used by default is `0`.

If you are using an external USB camera, then you might need to specify a different index for `cv2.VideoCapture()` to identify the camera.

Fortunately, `main.py` is already designed to accept a different index for the camera. Simply pass a `-c` option and the index. For example:

```sh
python main.py live -c 1
```

## Static Inputs

The project also provides support for image detection.

### Image Inputs

To detect bird/cockatiel instances from an image:

```sh
python main.py detect -i cockatiel.jpg
```

By default, `main.py`'s **detect** feature runs in Ensemble Method (Haar Cascade + SqueezeNet).

If you want to run "detect" while using only Haar Cascade, include the `--haar` option:

```sh
python main.py detect -i cockatiel.jpg --haar
```

### SqueezeNet-only

To run the SqueezeNet classifier, you can run `test.py` and provide the path to an input image using the `-i` option:

```sh
# test against the provided image of a Cardinal
python test.py -i cardinal.jpg

# test against the provided image of a Cockatiel
python test.py -i cockatiel.jpg
```

You can also pass your own model using the `-m` option.

```sh
python test.py -i cockatiel.jpg -m model.pt
```

## Haar Optimization (Optional)

To find optimal values for a specific dataset class:

```sh
python main.py optimize -d dataset/cockatiel
```

Note that this wouldn't affect the classifier's existing state, unless it is modified to use the values.

## Troubleshooting

In some cases, `CascadeClassifier()`'s `detectMultiScale()` may not work properly in some Linux environments, causing the following issue when running `gui.py`:

```
free(): invalid pointer
```

By default, `detectMultiScale()` uses OpenCL, and a broken OpenCL driver can cause the issue. In such cases, you can disable the OpenCL runtime and force OpenCV to use the CPU:

```sh
OPENCV_OPENCL_RUNTIME=disabled python gui.py
```

You can confirm that a broken OpenCL driver causes the issue by running:

```sh
clinfo
```

If this is the case, you will need to manually reinstall drivers yourself.
