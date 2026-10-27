import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import Accuracy, F1Score, Precision, Recall

import pandas as pd
import numpy as np
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
import time

from typing import Literal

DEFAULT_CLASS_IDS = ["cinnamon", "lutino", "pearl", "pied", "whiteface"]

class CustomDataset(Dataset):

    def __init__(self, X, y, BatchSize, transform):
        super().__init__()
        self.BatchSize = BatchSize
        self.X = X
        self.y = y
        self.transform = transform


    def get_number_of_batches(self):
        return math.floor(len(self.list_IDs) / self.BatchSize)
    

    def __getitem__(self, idx):
        class_id = self.y[idx]
        img = Image.open(self.X[idx]).convert("RGB")
        img = self.transform(img)

        return img, torch.tensor(int(class_id))


    def __len__(self):
        return len(self.X)



class SqueezeNet:

    def __init__(self, 
                 path_to_dataset='dataset', 
                 class_ids=DEFAULT_CLASS_IDS, 
                 epochs=10, 
                 batch_size=4,
                 learning_rate = 0.01):
        self.path_to_dataset = path_to_dataset
        self.class_ids = class_ids

        self.labels = pd.DataFrame()
        self.stages = ['train', 'val', 'test']

        # percentage of dataset used for training, validation, and testing set
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        # higher batch size = higher memory consumption = faster training
        # lower batch size = less memory consumption = slower training = more detail
        self.batch_size = batch_size

        # number of iterations around the dataset
        self.epochs = epochs

        # learning rate value, which we will pass to the optimizer
        self.learning_rate = learning_rate
        
        # use SqueezeNet with pre-trained weights from the ImageNet dataset
        self.model = models.squeezenet1_1(pretrained=True)
        
        # use the CPU if this script is executed on a non-CUDA-enabled machine
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # default transforms for image augmentation
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


    def load_custom_model(self, path_to_pt):
        """
        Use a custom SqueezeNet model in form of a .pt file
        """
        #dirname = os.path.dirname(__file__)
        filepath = os.path.join(os.getcwd(), path_to_pt)
        self.model = torch.load(filepath, map_location=self.device)


    def export_model(self, path_to_pt):
        """
        Export the current SqueezeNet model as a .pt file
        """
        torch.save(self.model, path_to_pt)


    def generate_labels(self, display=False):
        """
        Create a label map for the dataset
        """
        X = []
        y = []
        filenames = []

        for class_id in self.class_ids:
            dirpath = os.path.join(os.getcwd(), self.path_to_dataset, class_id)

            for file in os.listdir(dirpath):
                if file.lower().endswith(('.png', '.jpeg', '.jpg')):
                    X.append(os.path.join(dirpath, file))
                    y.append(class_id)
                    filenames.append(file)
        
        self.labels = pd.DataFrame(list(zip(X, filenames, y)), columns=['fileloc', 'filename', 'classid'])
        self.labels['int_class_id'] = self.labels['classid'].astype('category').cat.codes

        # print the entire dataframe
        if display:
            print(self.labels.to_string())


    def create_dataset(self):
        """
        Prepare the training, validation, and testing datasets.
        """
        df = self.labels.sample(frac=1)
        X = df.iloc[:, 0]
        y = df.iloc[:, 3]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - self.train_ratio, stratify=y, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=self.test_ratio/(self.test_ratio + self.val_ratio), random_state=0)

        image_datasets = {
            'train': CustomDataset(X_train.values, y_train.values, self.batch_size, self.transform),
            'val': CustomDataset(X_val.values, y_val.values, self.batch_size, self.transform),
            'test': CustomDataset(X_test.values, y_test.values, self.batch_size, self.transform)
        }

        self.dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=image_datasets[x].BatchSize, shuffle=True, num_workers=0) for x in self.stages
        }

        self.dataset_sizes = {x: len(image_datasets[x]) for x in self.stages}


    def train(self, overwrite_model=False):
        """
        Method for training/fine-tuning the SqueezeNet model
        """
        # update the last 2D convolution layer, so that the number of
        # output channels will match the number of classes. This is so
        # that we'll get outputs that depend on the classes we defined,
        # since SqueezeNet does not have a linear layer as its last layer.
        self.model.classifier._modules["1"] = torch.nn.Conv2d(512, len(self.class_ids), kernel_size=(1, 1))
        self.model.num_classes = len(self.class_ids)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return self.__train_model(self.model.to(self.device), criterion, optimizer, scheduler, overwrite_model)


    def __train_model(self, model, criterion, optimizer, scheduler, overwrite_model=False):
        """
        Private method which consists of the primary model training logic.
        """
        start_time = time.time()
        num_epochs = self.epochs

        total_loss = []
        total_acc = []

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("-" * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                num_batches = 0
                outputs = None

                for inputs, labels in self.dataloaders[phase]:
                    if phase == 'train':
                        num_batches += 1
                        percentage_complete = ((num_batches * self.batch_size) / (self.dataset_sizes[phase])) * 100
                        percentage_complete = np.clip(percentage_complete, 0, 100)
                        print("{:0.2f}".format(percentage_complete), "% \\ complete", end="\r")

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # reset gradient of optimized tensors
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs.float(), labels)

                        # update weights when in training mode
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    total_loss.append(float(running_loss))

                    predicted = torch.max(outputs.data, 1)[1]
                    running_corrects += (predicted == labels).sum()

                    if phase == 'train':
                        total_loss.append(float(running_loss))
                        total_acc.append(float(running_corrects))
                    
                    print("Loss: {:.4f}, Accuracy: {:.4f}".format(float(running_loss), float(running_corrects)))

                # in training mode, after each epoch, allow
                # the learning rate to decay
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]

                print("Epoch Loss: {:.4f}, Epoch Accuracy: {:.4f}".format(epoch_loss, epoch_acc.item()))
        
        # calculate time it took for the entire training process to finish
        time_elapsed = time.time() - start_time
        print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        # export entire model
        timestamp = time.time()
        torch.save(self.model, f"model_{self.epochs}-epochs_{timestamp}.pt")
        # export as state dict
        torch.save(self.model.state_dict(), f"model_{self.epochs}-epochs_{timestamp}_state-dict.pt")
        
        # if true, the class instance's model property will be set to the newly trained model
        if overwrite_model:
            self.model = model

        # visualize metrics
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(np.array(total_loss))
        ax1.set_ylabel("Running Loss")

        ax2.plot(np.array(total_acc))
        ax2.set_ylabel("Running Accuracy")
        ax2.set_xlabel(f"Iteration No. ({self.epochs} epochs)")
        
        plt.show()

        return model


    def test(self, 
             image, 
             model=None, 
             class_ids=DEFAULT_CLASS_IDS, 
             show_probabilities=False, 
             as_matlike=False,
             print_results=True
    ):
        """
        Test the model against an input image.
        """
        input_image = Image.open(image) if not as_matlike else Image.fromarray(image)

        start_time = time.time()
        trained_model = self.model if model is None else model

        preprocess = self.transform

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            trained_model.to('cuda')

        with torch.no_grad():
            output = trained_model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        categories = self.class_ids if model is None else class_ids

        top_prob, top_category_id = torch.topk(probabilities, len(categories))
        time_elapsed = time.time() - start_time

        if print_results:
            print("Label: {}, Probability: {}, Inference Time: {:.0f}m {:.6f}s".format(categories[top_category_id[0]], top_prob[0], time_elapsed // 60, time_elapsed % 60))

        if show_probabilities:
            for i in range(top_prob.size(0)):
                print(categories[top_category_id[i]], top_prob[i].item())

        return {"label": categories[top_category_id[0]],
                "probability": top_prob[0],
                "inference_time": "{:.0f}m {:.6f}s".format(time_elapsed // 60, time_elapsed % 60)}


    def __display_confusion_matrix(self, 
                                   input_tensor, 
                                   class_ids: list[str] = DEFAULT_CLASS_IDS):
        """
        Provides a visual representation of a confusion matrix, given a valid
        tensor. The input tensor must have a shape of (m x m), where m is the
        number of labels/classes.
        """
        
        # convert input tensor to numpy array
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.numpy()

        disp = ConfusionMatrixDisplay(confusion_matrix=input_tensor,
                                      display_labels=class_ids)

        disp.plot()
        plt.show()


    def __get_precision(self,
                        actual: torch.Tensor,
                        predicted: torch.Tensor,
                        class_ids: list[str] = DEFAULT_CLASS_IDS, 
                        average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro'):
        """
        Get the precision from actual values and predicted values.
        """
        precision = Precision(task="multiclass", average=average, num_classes=len(class_ids))
        
        return precision(predicted, actual)

    
    def __get_recall(self,
                     actual: torch.Tensor,
                     predicted: torch.Tensor,
                     class_ids: list[str] = DEFAULT_CLASS_IDS,
                     average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro'):
        """
        Get the recall from actual values and predicted values.
        """
        recall = Recall(task="multiclass", average=average, num_classes=len(class_ids))

        return recall(predicted, actual)

    
    def __get_accuracy(self,
                       actual: torch.Tensor,
                       predicted: torch.Tensor,
                       class_ids: list[str] = DEFAULT_CLASS_IDS,
                       average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro'):
        """
        Get the accuracy based on actual and predicted values.
        """
        accuracy = Accuracy(task="multiclass", average=average, num_classes=len(class_ids))

        return accuracy(predicted, actual)


    def __get_f1_score(self,
                       actual: torch.Tensor,
                       predicted: torch.Tensor,
                       class_ids: list[str] = DEFAULT_CLASS_IDS,
                       average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro'):
        """
        Get the F-1 score.
        """
        f1_score = F1Score(task="multiclass", num_classes=len(class_ids), average=average)

        return f1_score(predicted, actual)

    
    def eval(self, 
             class_ids=DEFAULT_CLASS_IDS, 
             model=None,
             average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro'):
        """
        Evaluate the model and get relevant metrics (e.g. confusion matrix).
        """

        # if the label map is not yet defined, generate the label map
        # and prepare the dataset, based on the input path to dataset
        # that is specified when the SqueezeNet class is instantiated
        if self.labels.shape[0] == 0:
            self.generate_labels()
            self.create_dataset()

        # get the class IDs from the dataset's label map
        int_class_ids = pd.DataFrame(self.labels[["int_class_id"]])
        dataset_class_ids = int_class_ids.apply(lambda x: x.iloc[0], axis=1)
        
        # list of predicted indexes
        predicted = []

        # convert to a tensor that can be passed to PyTorch's confusion matrix function
        actual = torch.tensor(dataset_class_ids)

        # iterate through each file in the dataset
        filepaths = self.labels[["fileloc"]].apply(lambda x: x.iloc[0], axis=1).tolist()

        # import trained model
        trained_model = self.model if model is None else model

        # define image augmentation transforms
        preprocess = self.transform

        # check if CUDA is available
        cuda_is_available = torch.cuda.is_available()
        
        for filepath in filepaths:
            input_image = Image.open(filepath)
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)

            input_batch = input_batch.to("cuda" if cuda_is_available else "cpu")
            trained_model.to("cuda" if cuda_is_available else "cpu")

            with torch.no_grad():
                output = trained_model(input_batch)

            predicted.append(torch.argmax(output[0]).item())
        
        # convert predictions to tensor
        preds = torch.tensor(predicted)

        # conf matrix
        conf_mat = multiclass_confusion_matrix(preds, actual, len(class_ids))

        # get precision, recall, and accuracy
        precision = self.__get_precision(actual, preds, average=average, class_ids=class_ids)
        recall = self.__get_recall(actual, preds, average=average, class_ids=class_ids)
        accuracy = self.__get_accuracy(actual, preds, average=average, class_ids=class_ids)
        f1_score = self.__get_f1_score(actual, preds, average=average, class_ids=class_ids)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("Accuracy: {}".format(accuracy))
        print("F-1 Score: {}".format(f1_score))

        # display confusion matrix
        self.__display_confusion_matrix(conf_mat)

