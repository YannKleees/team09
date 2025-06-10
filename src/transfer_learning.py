import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import time
import os
from PyQt5.QtCore import QObject, pyqtSignal
from tempfile import TemporaryDirectory
from src.dataset import BrainTumorDataset
from src.transforms_config import create_transform
class transfer_learning(QObject):
    """This is the class that handles transfer learning
    
    This class inherits the QObject class from PyQt to allow us to display the training progress in the visual interface.
    """
    epoch_msg = pyqtSignal(str)
    def __init__(self,batch_size,num_epochs,lr,data_root,disable_augment,adv_aug):
        """This is a standard __init__ which takes in some variables to be later used in trainging
        
        batch_size: integer which defines how many images to use in one pass of training
        num_epochs: integer which defines how many training iterations there should be
        lr: float which defines the learning rate of the neural network
        data_root: string which defines the directory to the training material. The training material must be in the format as
        defined in the dataset.py file
        disable_augment: boolean which decides if augmentations should be used on the training data. The augmentations applied are described
        in the transforms_config.py
        adv_aug: boolean that decides if advanced augmentations should be done on the training dataset
        """
        super().__init__()
        if torch.accelerator.is_available():
            self.device = torch.accelerator.current_accelerator().type
        else:
            self.device = 'cpu'
        self.epoch_msg.emit(f'trianing on {self.device}')
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.lr = lr
        self.transform = create_transform(
            image_size=(224, 224),
            augment=(not disable_augment),
            adv_aug=adv_aug,
            strong_aug=False,
            one_ch_to_three_ch=True
        )
        train_set = BrainTumorDataset(
                root_dir=data_root,
                subset="Training",
                transform=self.transform,
            )
        val_set = BrainTumorDataset(
                root_dir=data_root,
                subset="Testing",
                transform=self.transform,
            )
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        self.dataloaders = {
            'Training': self.train_loader,
            'Testing': self.val_loader
        }
        self.train_dataset = train_set
        self.test_dataset = val_set
        self.dataset_size = {
            'Training': len(self.train_dataset),
            'Testing': len(self.test_dataset)
        }
        self.class_names = self.train_dataset.classes

    def train(self,model, criterion, optimizer, scheduler):
        """This is the training function of the class
        
        model: torchvision.model which will be trained on the data defined in the __init__ method
        criterion: torch.nn which will be used as the loss function of the training
        optimizer: torch.optim which will be used as the optimizer during the training
        scheduler: torch.optim.lr_scheduler which will be used to adjust the learning rate during the course of the training
        """
        since = time.time()
        with TemporaryDirectory() as tempdir:
            best_model_path = os.path.join(tempdir,'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            max_accuracy=0.0
            for epoch in range(self.num_epochs):
                self.epoch_msg.emit(f'Epoch {epoch}/{self.num_epochs - 1}')
                for phase in ['Training','Testing']:
                    if phase == 'Training':
                        model.train()
                    else:
                        model.eval()
                    current_loss = 0.0
                    current_corrects = 0
                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase=='Training'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs,1)
                            loss = criterion(outputs,labels)
                            if phase == 'Training':
                                loss.backward()
                                optimizer.step()
                        current_loss+=loss.item()*inputs.size(0)
                        current_corrects+=torch.sum(preds == labels.data)
                    if phase == 'Training':
                        scheduler.step()
                    epoch_loss=current_loss/self.dataset_size[phase]
                    epoch_acc=current_corrects.double()/self.dataset_size[phase]
                    self.epoch_msg.emit(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    if phase =='Testing' and epoch_acc>max_accuracy:
                        max_accuracy = epoch_acc
                        torch.save(model.state_dict(), best_model_path)
                time_elapsed = time.time() - since
                self.epoch_msg.emit(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\nBest val Acc: {max_accuracy:4f}')
                model.load_state_dict(torch.load(best_model_path,weights_only=True))
        return model
    def finetuning(self,mode):
        """This is the finetuning method that take a resnet18 model and using our dataset we finetune it to our task
        
        mode: string which if binary is inputed will make the model predict in a binary manner aka is there a tumor or not. Otherwise
        the program will be in the multi mode where it will try and predict the specific type of cancer"""
        model_pretrained = models.resnet18(weights='IMAGENET1K_V1')
        model_pretrained_features = model_pretrained.fc.in_features
        if mode == "binary":
            model_pretrained.fc = nn.Linear(model_pretrained_features, 2)
        else:
            model_pretrained.fc = nn.Linear(model_pretrained_features, 4)
        model_pretrained = model_pretrained.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer_pretrained=optim.SGD(model_pretrained.parameters(), self.lr, momentum=0.9)
        expo_learn_sched = lr_scheduler.StepLR(optimizer_pretrained, step_size=7,gamma=0.1)
        self.train(model_pretrained,criterion,optimizer_pretrained,expo_learn_sched,self.num_epochs)
    def feat_extractor(self,mode):
        """This is the feature extraction method that take a resnet18 model and using our dataset we try to extract features from it

        mode: string which if binary is inputed will make the model predict in a binary manner aka is there a tumor or not. Otherwise
        the program will be in the multi mode where it will try and predict the specific type of cancer"""
        model_pretrained = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        for param in model_pretrained.parameters():
            param.requires_grad = False
        model_pretrained_features = model_pretrained.fc.in_features
        if mode == "binary":
            model_pretrained.fc = nn.Linear(model_pretrained_features, 2)
        else:
            model_pretrained.fc = nn.Linear(model_pretrained_features, 4)
        model_pretrained = model_pretrained.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer_pretrained = optim.SGD(model_pretrained.fc.parameters(), self.lr, momentum=0.9)
        expo_learn_sched = lr_scheduler.StepLR(optimizer_pretrained, step_size=7, gamma=0.1)
        self.train(model_pretrained,criterion,optimizer_pretrained,expo_learn_sched,self.num_epochs)
