import torch
import numpy as np
import h5py
import torch.utils.data as data
from models.m_t import Geo_base as geo_base_t
from models.m_v import Geo_base as geo_base_v
from models.m_vt import Geo_base as geo_base_vt
from cmath import inf
import sys
sys.path.insert(1, '/data/1/mm-locate-news') 
from utils import *
from breakingnews.bn_args import get_parser

parser = get_parser()
args = parser.parse_args()

checkpoint_name = args.check_point.split('/')[-1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def get_eval(topk):

    results_path = f'{args.results_path}/{args.mode}/{args.model_name}_{checkpoint_name}_predictions.json'

    with open(results_path) as f:
        preds = json.load(f)
    h5f = h5py.File(args.h5_path, 'r')
    ls_gcds = []

    for i, id in enumerate(preds):
        # print(i+1, len(preds))
        min_d = inf

        ground_truth_locs_bn =  h5f[id]['class'][()] 
        if ground_truth_locs_bn.shape[0] == 1:
            if len(ground_truth_locs_bn.shape) > 2:
                ground_truth_locs_bn = ground_truth_locs_bn[0]

        topk_preds=preds[id][:topk]

        err = 0
        for p_coords in topk_preds:
            
            for gt in ground_truth_locs_bn:
                
                d = gcd(gt, p_coords)
                    
                if d < min_d:
                    min_d = d
                    d=inf

        ls_gcds.append(min_d)

    mean_gcd = np.mean(ls_gcds)
    median_gcd = np.median(ls_gcds)

    print(f'top {topk} results:', 'MEAN:', mean_gcd/1000, ' | MEDIAN:', median_gcd/1000)
    
    results_path = f'{args.results_path}/{args.mode}/eval_{args.model_name}_{checkpoint_name}.json'
    
    with open(results_path, 'w') as outfile:
        json.dump({'Mean': mean_gcd/1000, 'Median': median_gcd/1000}, outfile)


class Data_Loader_BN(data.Dataset):
    def __init__(self):
        
        self.h5f = h5py.File(args.h5_path, 'r')

        self.ids = [str(id) for id in self.h5f]
        self.h5f_info = h5py.File(f'/data/1/datasets/breaking_news/h5_splits/mufl/test/textual.h5', 'r')

    def __getitem__(self, index):
        instanceId = self.ids[index]


        grp = self.h5f[instanceId]

        clip = grp['clip'][()]
        vloc = grp['loc'][()]
        obj = grp['obj'][()]
        scene = grp['scene'][()]
        body = grp['body'][()]
        entity = grp['entity'][()]
        loc = grp['class'][()][0]

        if len(loc.shape) == 1:
            loc = np.array([loc])

        d = 50 - loc.shape[0]
        
        for _ in range(d):
            t = np.array([[inf, inf]])
            loc = np.append(loc, t, 0)

        if len(entity) == 0:
            entity = body

        return {
            'id': instanceId,
            'loc': vloc,
            'obj': obj,
            'scene':scene,
            'body':body,
            'entity':entity,
            'clip':clip,
            'locs': loc,
            'all_ids':self.ids,
        }

    def __len__(self):
        return len(self.ids)

def gcd( L1, L2) : # L1: lat, lng
    lat1, lon1 = (L1[0], L1[1])
    lat2, lon2 = (L2[0], L2[1])

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    return 6371 * ( acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))


def d_range(inp_range, val):
    if val < 25:
        inp_range[25] +=1
    elif 25<= val < 200:
        inp_range[200] +=1
    elif 200 <= val < 750:
        inp_range[750] +=1
    elif 200 <= val < 2500:
        inp_range[2500] +=1
    elif 2500 <= val < 3000:
        inp_range[3000] +=1
    elif 3000 <= val < 4000:
        inp_range[4000] +=1
    elif 4000 <= val < 5000:
        inp_range[5000] +=1
    elif 5000 <= val < 10000:
        inp_range[10000] +=1
    elif 10000 <= val < 20000:
        inp_range[20000] +=1
    return inp_range


if args.model_name == 'reg_v_clip':
    model = geo_base_v()
elif args.model_name == 'reg_t_2bert':
    model = geo_base_t()
elif args.model_name == 'reg_mm_clip_loc_scene':
    model = geo_base_vt()

model.to(device)


if device.type == 'cpu':
    checkpoint = torch.load(args.check_point, encoding='latin1', map_location='cpu')
else:
    checkpoint = torch.load(args.check_point, encoding='latin1')

model.load_state_dict(checkpoint['state_dict'])
model.eval()

test_loader = torch.utils.data.DataLoader(
                Data_Loader_BN(),
                batch_size= 128,
                shuffle=False,
                num_workers=0,
                pin_memory=False)

ls_gcds = []    
rg = {25:0, 200:0, 750:0, 2500:0, 3000:0, 5000:0, 4000:0, 10000:0, 20000:0}

for i_counter, batch in enumerate(test_loader):

    print(i_counter + 1, int(10581/len(batch['clip'])), 'batches done! | ', rg)
    output = model( batch )
    

    for i_s, id in enumerate( batch['id']):
        
        pred_coords = ( output[i_s][0].item(), output[i_s][1].item() )
        
        min_d = inf

        for gt_coords0 in batch['locs'][i_s]:
            if gt_coords0[0].item() == inf:
                continue
            gt_coords = ( gt_coords0[0].item(), gt_coords0[1].item() )
                    
            d = gcd( gt_coords,  pred_coords )
                        
            if d < min_d:
                min_d = d
                d = inf
            # if d > 10000:
            #     print

        ls_gcds.append(min_d)
        rg = d_range(rg, min_d)

mean_gcd = np.mean(ls_gcds)
median_gcd = np.median(ls_gcds)

print('#samples', len(ls_gcds),'MEAN:', mean_gcd/1000, ' | MEDIAN:', median_gcd/1000)
    
results_path = f'{args.results_path}/eval_{args.model_name}_{checkpoint_name}.json'
    
with open(results_path, 'w') as outfile:
    json.dump({'#samples': len(ls_gcds), 'Mean': mean_gcd/1000, 'Median': median_gcd/1000}, outfile)

print(rg)