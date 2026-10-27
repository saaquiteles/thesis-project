"""
Here is a sample script for using SqueezeNet from PyTorch.

Note that this only currently supports ImageNet. The bird-cockatiel model will be implemented
in this script soon, once the API is complete.
"""

import torch
import torchvision.models as models
import urllib
from PIL import Image
from torchvision import transforms

class SqueezeNetImageNet():

    REPOSITORY = "pytorch/vision:v0.10.0"
    MODEL = "squeezenet1_0"

    def __init__(self):
        # note that 'pretrained' is already deprecated, even though the PyTorch docs state that
        # 'pretrained' is a valid option.
        # we have to explicitly state weights to be equal to IMAGENET1K_V1, which
        # is equivalent to using the ImageNet-pretrained SqueezeNet.
        self.weights = models.squeezenet.SqueezeNet1_0_Weights.IMAGENET1K_V1
        self.model = torch.hub.load(self.REPOSITORY, self.MODEL, weights=self.weights)
        self.model.eval()


    # returns a function which consists of unified transforms.
    # a transform can be thought of as an image processing task whose
    # goal is to perform data augmentation before passing input images
    # to the model (i.e. preprocess the image, and convert to tensor)
    def preprocess(self):
        """
        Returns a function which consists of unified 'transforms'. A 'transform' can be 
        thought of as an image processing task, whose goal is to perform data augmentation, 
        which is important before passing input images to the model (i.e. we're preprocessing 
        the image, and then convert it to tensor).
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    # method for performing classification against an image from a URL
    def classify_from_url(self, url, filename):
        """
        Performs classification against an image from a URL.
        """
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

        preprocessor = self.preprocess()

        input_image = Image.open(filename)
        input_tensor = preprocessor(input_image)
        input_batch = input_tensor.unsqueeze(0)
        
        output = self.get_output(input_batch)
        self.get_output_probabilities(output)

    
    def get_output(self, batch):
        """
        Runs the model/classifier against an input tensor.
        """
        output = None

        if torch.cuda.is_available():
            batch = batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(batch)

        return output

    
    def get_output_probabilities(self, output):
        """
        Displays classes with the highest probabilities, according to the
        ImageNet classes.

        Note that we're only supporting ImageNet here at the moment. We'll
        support custom classes and fine-tuned models soon.
        """
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        top5_prob, top5_catid = torch.topk(probabilities, 5)

        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    # declare test image
    TEST_IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    TEST_IMAGE_NAME = "dog.jpg"

    # import our custom SqueezeNet class
    squeezenet = SqueezeNetImageNet()

    # run the SqueezeNet model against the image from the URL we provided
    squeezenet.classify_from_url(TEST_IMAGE_URL, TEST_IMAGE_NAME)
