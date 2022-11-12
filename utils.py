import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from math import radians, sin, cos, acos
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def open_json(fileName):
    with open(fileName,encoding='utf8') as json_data:
            return json.load(json_data)

def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def save_checkpoint(state, path):
    filename = f'{path}/{state["epoch"]}.pth.tar'
    torch.save(state, filename)

# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target0):
        target = target0['class'].to(device)
        output =  self.criterion(output, target.squeeze(1))
        return output


class Reg_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output0, target0):
        # method GCD
        output = torch.deg2rad( output0 )
        # targets = torch.stack( target0['lat_lon'], 1)

        target = torch.deg2rad(target0['lat_lon'] )
        lat1 = output[:, 0].to(device)
        lon1 = output[:, 1].to(device)
        lat2 = target[:, 0].to(device)
        lon2 = target[:, 1].to(device)
        a = 6371 * ( torch.acos(torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) * torch.cos(lon1 - lon2))   )
        a = torch.mean( a )
        a.requires_grad_ = True
        return a


def classify(combined_features):
    # flatten lists
    le_combined_features = [p for pred in combined_features for p in pred]

    trg = [p[0].item() for p in le_combined_features]
    top1 = [p[1].item() for p in le_combined_features]

    return {
        'classification': {
            'Accuracy': format(accuracy_score(trg, top1), '.4f'),
            'Precision': format(precision_score(trg, top1, average='weighted'), '.4f'),
            'Recall': format(recall_score(trg, top1, average='weighted'), '.4f'),
            'F1 score': format(f1_score(trg, top1, average='weighted'), '.4f')
        }
    }


def gcd(lon1=None, lat1=None, lon2=None, lat2=None):
    if lon1 == lon2 and lat1 == lat2:
        return 0
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    return 6371 * ( acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))


def find_location(cls_in, path_loc_to_class):
    with open(path_loc_to_class, encoding='utf8') as json_data:
        all_classes = json.load(json_data)
    for loc in all_classes:
        if all_classes[loc] == cls_in:
             return loc

def calc_gcd_accuracy_tk(t_coord, p_coord):
    n_total = len(t_coord)

    error_levels_ref = { 'city': 25, 'region': 200, 'country': 750, 'continent': 2500}
    error_levels = {'city': 0, 'region': 0, 'country': 0, 'continent': 0}

    for t, p in zip(t_coord, p_coord):

            gcd_tp = gcd(float(t[0]),float(t[1]),float(p[0]),float(p[1]))

            if gcd_tp <= error_levels_ref['city']:
                error_levels['city'] +=1
            if gcd_tp <= error_levels_ref['region']:
                error_levels['region'] +=1
            if gcd_tp <= error_levels_ref['country']:
                error_levels['country'] +=1
            if gcd_tp <= error_levels_ref['continent']:
                error_levels['continent'] +=1
    
    # error_levels = {'city': np.round(error_levels['city']/n_total,3),'region': np.round(error_levels['region']/n_total,3),'country': np.round(error_levels['country']/n_total,3), 'continent': np.round(error_levels['continent']/n_total,3)}
    
    error_levels = {
    'city': np.round(error_levels['city']/n_total*100,1),
    'region': np.round(error_levels['region']/n_total*100,1),
    'country': np.round(error_levels['country']/n_total*100,1), 
    'continent': np.round(error_levels['continent']/n_total*100,1)
    }
    
    return error_levels


def classify_gcd(combined_target_pred, cls2coord):

    targets = np.array( combined_target_pred )[:,0]
    preds = np.array( combined_target_pred )[:,1]
    target_coords = [cls2coord[str(trg)] for trg in targets]
    pred_coords = [cls2coord[str(p)] for p in preds]
 
    gcd_acc = calc_gcd_accuracy_tk(target_coords, pred_coords)

    return  gcd_acc