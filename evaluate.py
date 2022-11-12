import os
import random
import torch
import logging
import numpy as np
from pathlib import Path
from args import get_parser
from models.m_t import Geo_base as geo_base_t
from models.m_v import Geo_base as geo_base_v
from models.m_vt import Geo_base as geo_base_vt
from data_loader import Data_Loader
from utils import *

ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

path_eval = f'{ROOT_PATH}/{args.evaluation_results}/{args.model_name}'
Path(f'{path_eval}').mkdir(parents=True, exist_ok=True)

# create directories for the experiments
logging_path = f'{args.path_results}'
checkpoint_path = f'{ROOT_PATH}/{args.snapshots}/{args.model_name}'
Path(logging_path).mkdir(parents=True, exist_ok=True)
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[ logging.FileHandler(f'{logging_path}/test.log', 'w'),  logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # set model
    model = {'v_clip':geo_base_v(), 'v_loc':geo_base_v(), 'v_scene':geo_base_v(), 'v_obj':geo_base_v(), 'v_clip_loc':geo_base_v(), 'v_clip_scene':geo_base_v(), 'v_loc_scene':geo_base_v(), 'v_loc_scene_obj':geo_base_v() , 'v_clip_loc_scene':geo_base_v(),'v_loc_obj':geo_base_v(), 'v_scene_obj':geo_base_v(),
             't_body':geo_base_t(), 't_entity':geo_base_t() ,'t_2bert':geo_base_t(),
             'm_body_clip':geo_base_vt(), 'm_entity_clip':geo_base_vt(), 'm_2bert_clip':geo_base_vt(),'m_2bert_clip_scene':geo_base_vt(), 'm_2bert_clip_loc':geo_base_vt() ,  'm_2bert_loc_scene':geo_base_vt(), 'm_2bert_clip_loc_scene':geo_base_vt(),
            #  'reg_v_clip':geo_base_v(), 'reg_v_scene':geo_base_v(), 'reg_v_clip_loc_scene':geo_base_v(),
            #  'reg_t': geo_base_t(),
            #  'reg_mm_clip':geo_base_vt(), 'reg_mm_loc_scene':geo_base_vt(), 'reg_mm_body_clip':geo_base_vt(), 'reg_mm_entity_clip':geo_base_vt(), 'reg_mm_clip_loc_scene':geo_base_vt()
    }[args.model_name]

    model.to(device)

    model_path = f'{checkpoint_path}/{args.test_check_point}'
    logger.info(f"=> loading checkpoint '{model_path}'")
    if device.type == 'cpu':
        checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(model_path, encoding='latin1')

    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']})")
    test_vers = {'T1':{}, 'T2':{}, 'T3':{} }

    cls2coord = open_json(f'{args.data_path}/mm-locate-news/cls_to_coord.json')
    
    for ver in [1,2,3]:
        logger.info(f'Test version {ver} |  Model {args.model_name}')

        data_loader_test = Data_Loader(data_path=f'{args.data_path}/{args.data_to_use}', partition=f'test{ver}')  
        
        test_loader = torch.utils.data.DataLoader( data_loader_test,  batch_size=args.batch_size,  shuffle=False, num_workers=args.workers, pin_memory=False)

        logger.info(f'Test loader prepared for checkpoint {args.test_check_point}')

        test_vers[f'T{ver}'] = test(test_loader, model, cls2coord)

        logger.info(test_vers[f'T{ver}'] )
    
    logger.info(f'{test_vers["T1"][0]} & {test_vers["T1"][1]}  & {test_vers["T1"][2]}  & {test_vers["T1"][3]} &' +
                    f'{test_vers["T2"][0]} & {test_vers["T2"][1]}  & {test_vers["T2"][2]}  & {test_vers["T2"][3]} &' +
                    f'{test_vers["T3"][0]} & {test_vers["T3"][1]}  & {test_vers["T3"][2]}  & {test_vers["T3"][3]} '                 
                    )

    save_file(f'{path_eval}/{args.test_check_point}.json', test_vers)


def test(test_loader, model, cls2coord):

    # switch to evaluate mode
    model.eval()
    output_top1 = []

    for batch in test_loader:
        
        output_classify = model(batch)

        target = batch['class']

        output_top1.extend([  [t.item(), torch.topk(o, k=1)[1].item()] for o, t in zip(output_classify, target)  ])
   
    result_gcd_top1 = classify_gcd(output_top1, cls2coord)

    return [result_gcd_top1['city'], result_gcd_top1['region'], result_gcd_top1['country'], result_gcd_top1['continent']]


if __name__ == '__main__':
    main()

