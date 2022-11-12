import torch.utils.data as data
import h5py
from utils import *
from args import get_parser
parser = get_parser()
args = parser.parse_args()


# get id-to-class from train data
# h5_train = h5py.File('/data/1/mm-locate-news/dataset/mm-locate-news/train.h5', 'r')
# id2cls = {}
# for id in h5_train:
#     grp_train = h5_train[id]
#     loc_wd_id = id.split('_')[0]
#     cls_train = grp_train['class'][()]
#     id2cls[loc_wd_id] = cls_train
# # 
# # #### rewrite h5 val
# h50 = h5py.File('/data/1/mm-locate-news/dataset/mm-locate-news/val_wrong_class.h5', 'r')
# h5f = h5py.File(f'/data/1/mm-locate-news/dataset/mm-locate-news/val.h5', 'w')
# err = {}
# count = 0
# for id in h50:   
#     grp0 = h50[id]
#     clip = grp0['clip'][()]
#     loc = grp0['loc'][()]
#     obj = grp0['obj'][()]
#     scene = grp0['scene'][()]
#     body = grp0['body'][()]
#     entity = grp0['entity'][()]
#     cls0 = grp0['class'][()]
#     loc_wd = id.split('_')[0]
#     try:
#         cls = id2cls[loc_wd]
#     except:
#         err[loc_wd] = err[loc_wd]+1 if loc_wd in err else 1
#     grp = h5f.create_group(str(id))
#     grp.create_dataset(name='loc', data = loc, compression="gzip", compression_opts=9) 
#     grp.create_dataset(name='scene', data = scene, compression="gzip", compression_opts=9) 
#     grp.create_dataset(name='obj', data = obj, compression="gzip", compression_opts=9) 
#     grp.create_dataset(name='body', data = body, compression="gzip", compression_opts=9) 
#     grp.create_dataset(name='entity', data = entity, compression="gzip", compression_opts=9) 
#     grp.create_dataset(name='clip', data = clip, compression="gzip", compression_opts=9)  
#     grp.create_dataset(name='class', data = cls, compression="gzip", compression_opts=9)  
# h5f.close()

#####
# locations_coords = open_json('/data/1/mm-locate-news/dataset/mm-locate-news/cls_to_coord.json')
# info = open_json('/data/1/mm-locate-news/dataset/mm-locate-news/all_locations_test.json')
# coords_vals = {}
# h50 = h5py.File(f'/data/1/mm-locate-news/dataset/mm-locate-news/val.h5', 'r')
# for id in h50:
#     coords_val = {'h5':[], 'info':[]}
#     sample = h50[id]
#     loc_wd = id.split('_')[0]
#     loc = info[loc_wd]
#     # if 'city' in list(loc.keys()):
#     cls = sample['class'][()]
#     cls2coords = locations_coords[str(int(cls))]
#     coords_val['info'] = [loc['coordinates']['latitude'], loc['coordinates']['longitude']]
#     coords_val['h5'] = [cls2coords[1], cls2coords[0]]
#     coords_vals[id] = coords_val
#     count +=1
# ##### 

class Data_Loader(data.Dataset):
    def __init__(self, data_path, partition):


        if data_path == None:
            raise Exception('No data path specified.')

        self.partition = partition
        
        self.h5f = h5py.File(f'{data_path}/{partition}.h5', 'r')

        self.ids = [str(id) for id in self.h5f]

        self.locations_coords = open_json(f'{data_path}/cls_to_coord.json')
        

    def __getitem__(self, index):
        instanceId = self.ids[index]
        grp = self.h5f[instanceId]

        clip = grp['clip'][()]
        loc = grp['loc'][()]
        obj = grp['obj'][()]
        scene = grp['scene'][()]
        body = grp['body'][()]
        entity = grp['entity'][()]
        cls = grp['class'][()]
        coords = self.locations_coords[str(int(cls))]

        return {
            'id': instanceId,
            'loc': loc,
            'obj': obj,
            'scene':scene,
            'body':body,
            'entity':entity,
            'clip':clip,
            'class': cls,
            'lat': coords[0],
            'lon': coords[1],
            'lat_lon': coords,
            'all_ids':self.ids
        }

    def __len__(self):
        return len(self.ids)

