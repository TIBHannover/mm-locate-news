import torch.utils.data as data
import h5py
from utils import *
from args import get_parser
parser = get_parser()
args = parser.parse_args()


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

