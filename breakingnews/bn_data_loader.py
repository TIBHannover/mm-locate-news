import torch.utils.data as data
import h5py


class Data_Loader_BN(data.Dataset):
    def __init__(self, data_path, partition):

        if data_path == None:  raise Exception('No data path specified.')

        self.partition = partition
        
        self.h5f = h5py.File(f'{data_path}/{partition}.h5', 'r')
        self.h5f_info = h5py.File(f'{data_path}/{partition}_info.h5', 'r')
        self.ids = [str(id) for id in self.h5f]
        

    def __getitem__(self, index):
        instanceId = self.ids[index]

        grp = self.h5f[instanceId]
        clip = grp['clip'][()]
        loc = grp['loc'][()]
        obj = grp['obj'][()]
        scene = grp['scene'][()]
        body = grp['body'][()]
        entity = grp['entity'][()]

        grp_info = self.h5f_info[instanceId]
        coords = grp_info['locs'][()]

        if coords.shape[0] == 1:
            if len(coords.shape) > 2:
                coords = coords[0]
        if len(coords.shape) > 1:
            coords = coords[0]
        cls = grp['class'][()]


        return {
            'id': instanceId,
            'loc': loc,
            'obj': obj,
            'scene':scene,
            'body':body,
            'entity':entity,
            'clip':clip,
            'all_ids':self.ids,
            'lat_lon': coords,'class': cls,
        }

    def __len__(self):
        return len(self.ids)
