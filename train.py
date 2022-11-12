from args import get_parser
import time
import logging
from pathlib import Path
from data_loader import Data_Loader
from utils import *
from models.m_t import Geo_base as geo_base_t
from models.m_v import Geo_base as geo_base_v
from models.m_vt import Geo_base as geo_base_vt
import os
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# create directories for train experiments
logging_path = f'{args.path_results}'
checkpoint_path = f'{ROOT_PATH}/{args.snapshots}/{args.model_name}'
Path(logging_path).mkdir(parents=True, exist_ok=True)
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{logging_path}/train{args.model_name}.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # set model
    model ={'v_clip':geo_base_v(), 'v_loc':geo_base_v(), 'v_scene':geo_base_v(), 'v_obj':geo_base_v(), 'v_loc_obj':geo_base_v(),'v_clip_loc':geo_base_v(), 'v_clip_scene':geo_base_v(), 'v_loc_scene':geo_base_v(), 'v_loc_scene_obj':geo_base_v() , 'v_clip_loc_scene':geo_base_v(),
            't_body':geo_base_t(), 't_entity':geo_base_t() ,'t_2bert':geo_base_t(),
            'm_body_clip':geo_base_vt(), 'm_entity_clip':geo_base_vt(), 'm_2bert_clip':geo_base_vt(),'m_2bert_clip_scene':geo_base_vt(), 'm_2bert_clip_loc':geo_base_vt() ,  'm_2bert_loc_scene':geo_base_vt(), 'm_2bert_clip_loc_scene':geo_base_vt(),
            # 'reg_v_clip':geo_base_v(), 'reg_v_scene':geo_base_v(), 'reg_v_clip_loc_scene':geo_base_v(),
            # 'reg_t': geo_base_t(),
            # 'reg_mm_clip':geo_base_vt(), 'reg_mm_loc_scene':geo_base_vt(), 'reg_mm_body_clip':geo_base_vt(), 'reg_mm_entity_clip':geo_base_vt(), 'reg_mm_clip_loc_scene':geo_base_vt()
    }[args.model_name]

    
    criterion = Loss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load checkpoint
    resume_path = f'{checkpoint_path}/{args.resume}'
    if os.path.isfile(resume_path):
        logger.info(f"=> loading checkpoint '{args.resume}''")
        checkpoint = torch.load(resume_path)
        args.start_epoch = int( checkpoint['epoch'].replace('epoch_',''))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    if args.freeze_text:
        for p in model.learn_text.parameters():
            p.requires_grad = False
    if args.freeze_image:
        for p in model.learn_image.parameters():
            p.requires_grad = False

    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # prepare training loader
    data_loader_train = Data_Loader(data_path=f'{args.data_path}/{args.data_to_use}', partition='train')
    train_loader = torch.utils.data.DataLoader( data_loader_train,  batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=False)
    logger.info('Training loader prepared.')

    # prepare validation loader
    data_loader_val = Data_Loader(data_path=f'{args.data_path}/{args.data_to_use}', partition='val')
    val_loader = torch.utils.data.DataLoader( data_loader_val,  batch_size=args.batch_size,  shuffle=False, num_workers=args.workers, pin_memory=False)
    logger.info('Validation loader prepared.')

    cls2coord = open_json(f'{args.data_path}/mm-locate-news/cls_to_coord.json')

    # train
    best_gcd_validation = {'city': -1, 'region': -1, 'country': -1, 'continent': -1 , 'all':-1,
                            'city_region': -1, 'city_region_country': -1}
    Path(f'{checkpoint_path}/city').mkdir(parents=True, exist_ok=True)
    Path(f'{checkpoint_path}/city_region').mkdir(parents=True, exist_ok=True)
    Path(f'{checkpoint_path}/city_region_country').mkdir(parents=True, exist_ok=True)
    Path(f'{checkpoint_path}/region').mkdir(parents=True, exist_ok=True)
    Path(f'{checkpoint_path}/country').mkdir(parents=True, exist_ok=True)
    Path(f'{checkpoint_path}/continent').mkdir(parents=True, exist_ok=True)
    Path(f'{checkpoint_path}/all').mkdir(parents=True, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):

        train_result, batch_time = train(train_loader, model, criterion, optimizer, epoch)

        val_loss , gcd_validation = validate(val_loader, model, criterion, cls2coord)

        logger.info(f'Validation loss: {val_loss} | Mean validation gcd: { gcd_validation["all"] } ' )

        # show tensorboard
        if args.tensorboard == True:
            writer.add_scalars(f"{args.model_name}/ {args.data_to_use} / Loss ",{ 'TRAIN': train_result['loss'].data,
                                                                                    'VAL': val_loss.data}, epoch  )


        save_checkpoint({'data': args.data_path.split('/')[-1],
                        'epoch': f'{epoch + 1}',
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, path=f'{checkpoint_path}')  

        for key_granularity in gcd_validation:

            if best_gcd_validation[key_granularity] <= gcd_validation[key_granularity]:
                gcd_val = gcd_validation[key_granularity]
                best_gcd_validation[key_granularity]  = gcd_val
                save_name = f'epoch_{epoch + 1}_gcdVAL_{key_granularity}_{gcd_val}_loss_{np.round(val_loss.item(), 3)}'

                save_checkpoint({'data': args.data_path.split('/')[-1],
                                'epoch': save_name,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, path=f'{checkpoint_path}/{key_granularity}')     
   

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses =  AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):

        output = model(batch)

        # compute loss
        loss = criterion(output, batch)

        # compute gradient and do Adam step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.zero_grad()
        losses.update(loss.data, args.batch_size)
        log_loss = f'Loss: {losses.val} ({losses.avg})'
        loss.backward()
        optimizer.step()

        # track time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(f'{args.model_name} | Epoch: {epoch+1} - {log_loss} - Batch: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')
    
    results = {  'loss': losses.avg }

    return results, batch_time
    

def validate(val_loader, model, criterion, cls2coord):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_top1 = []

    for i, batch in enumerate(val_loader):
        
        output = model(batch)

        output_top1.extend([  [t.item(), torch.topk(o, k=1)[1].item()] for o, t in zip(output,  batch['class'])  ])

        loss = criterion(output, batch)

        losses.update(loss.data, args.batch_size)

    val_loss =  losses.avg 

    gcd_validation =  classify_gcd(output_top1, cls2coord)

    gcd_validation['all'] = np.mean( [gcd_validation['city'], gcd_validation['region'], gcd_validation['country'], gcd_validation['continent']] )
    gcd_validation['city_region'] = np.mean( [gcd_validation['city'], gcd_validation['region'] ] )
    gcd_validation['city_region_country'] = np.mean( [gcd_validation['city'], gcd_validation['region'], gcd_validation['country'] ] )

    return val_loss, gcd_validation


if __name__ == '__main__':
    if args.tensorboard == True:
        writer = SummaryWriter()
    main()
