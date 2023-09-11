import os 
import csv
import torch
import logging
import numpy as np
import torch
from pathlib import Path
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import glob

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
import argparse

ROOT_PATH = str(Path(os.path.dirname(__file__)))



class SceneEmbedding:
    def __init__(self, model_path=ROOT_PATH+'/models/scene/resnet50_places365.pth.tar',
                        labels_file=ROOT_PATH+'/models/scene/categories_places365.txt',
                        hierarchy_file=ROOT_PATH+'/models/scene/scene_hierarchy_places365.csv',
                        arch='resnet50'):
        if model_path is not None:
            model = models.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval().to(self._device)
            self.model = model

            # method for centre crop
            self._centre_crop = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            logging.warning('No model built.')

        # load hierarchy
        if hierarchy_file is not None and os.path.isfile(hierarchy_file):
            self._load_hierarchy(hierarchy_file)
        else:
            logging.warning('Hierarchy file not specified.')

        # load the class label
        if labels_file is not None and os.path.isfile(labels_file):
            classes = list()
            with open(labels_file, 'r') as class_file:
                for line in class_file:
                    cls_name = line.strip().split(' ')[0][3:]
                    cls_name = cls_name.split('/')[0]
                    classes.append(cls_name)
            self.classes = tuple(classes)
        else:
            logging.warning('Labels file not specified.')

    def embed(self, image):
        try:
            input_img = Variable(self._centre_crop(image).unsqueeze(0)).to(self._device)

            # forward pass for feature extraction
            x = input_img
            i = 0
            for module in self.model._modules.values():
                if i == 9:
                    break
                x = module(x)
                i += 1

            return x.detach().cpu().numpy().squeeze()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            logging.error('Cannot create embedding for image')
            return []

    def _load_hierarchy(self, hierarchy_file):
        hierarchy_places3 = []
        hierarchy_places16 = []

        with open(hierarchy_file, 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            next(content)  # skip explanation line
            next(content)  # skip explanation line
            for line in content:
                hierarchy_places3.append(line[1:4])
                hierarchy_places16.append(line[4:])

        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=np.float64)
        hierarchy_places16 = np.asarray(hierarchy_places16, dtype=np.float64)

        # NORM: if places label belongs to multiple labels of a lower level --> normalization
        self._hierarchy_places3 = hierarchy_places3 / np.expand_dims(np.sum(hierarchy_places3, axis=1), axis=-1)
        self._hierarchy_places16 = hierarchy_places16 / np.expand_dims(np.sum(hierarchy_places16, axis=1), axis=-1)




