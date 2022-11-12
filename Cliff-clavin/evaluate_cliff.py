from cmath import inf
import sys
sys.path.insert(1, '/data/1/mm-locate-news')
from utils import *
import h5py
from args import get_parser
from pathlib import Path
import os
import glob
import requests
from shutil import copy

# read parser
parser = get_parser()
args = parser.parse_args()
ROOT = Path(os.path.dirname(__file__))

# a = [[1,1], [2,2], [3,3]]
# b = [[4,4], [5,5], [6,6]]
# ab = a+b
# print(ab)

def get_cliff_outputs():
    output_file = {}
    files = glob.glob(f'{ROOT}/output/Cliff-clavin_predictions/*')
    for f in files:
        file = open_json(f)
        id = f.split('.json')[0].split('/')[-1]
        output_file[id] = file
    return output_file


def get_prediction(preds_in0, k):
    
    try:
        preds_in = preds_in0['results']['places']['focus'][k]
    except:
        return []

    results = {}
    max_n = 0

    for loc in preds_in:
        loc_id = loc['id']
        
        if loc_id not in results:
            results[loc_id] = {'count':0, 'coords': (None, None)}

        results[loc_id]['count'] += 1
        results[loc_id]['coords'] = ( loc['lon'], loc['lat'] )
        
        if results[loc_id]['count'] > max_n:
            max_n = results[loc_id]['count']
    
    out_coords = []

    for loc_id in results:
        if results[loc_id]['count'] == max_n:
            out_coords.append( results[loc_id]['coords'] )

    return out_coords


def calc_gcd_accuracy_tk(t_coords, ps_coords, err_level):
    n_total = len(t_coords)

    gcd_acc = 0

    for i in range(n_total):
        
        t = t_coords[i]
        ps = ps_coords[i]

        min_gcd = inf

        for p in ps:

            gcd_tp = gcd(float(t[0]),float(t[1]),float(p[0]),float(p[1]) )
            if gcd_tp < min_gcd:
                min_gcd = gcd_tp

        if min_gcd <= err_level:
            gcd_acc +=1

    return np.round(gcd_acc/n_total*100,1)


cls2coord = open_json(f'{args.data_path}/mm-locate-news/cls_to_coord.json')
cliff_output_file = get_cliff_outputs()
gcds = {}
error_levels_ref = { 'city': 25, 'region': 200, 'country': 750, 'continent': 2500}
print_table = ""

for v in [1,2,3]:
    gcds[f'T{v}'] = {}
    h5_test = h5py.File(f'{args.data_path}/mm-locate-news/test{v}.h5', 'r')
    v_ids = [str(id) for id in h5_test]

    targets_coords = []
    preds_coords = {'city':[], 'region':[] , 'country':[], 'continent': []}

    for id in v_ids:

        cls = h5_test[id]['class'][()][0]
        targets_coords.append( cls2coord[str(cls)] )

        pred_cities =  get_prediction(cliff_output_file[id], 'cities' )
        pred_states = get_prediction(cliff_output_file[id], 'states' )
        pred_countries = get_prediction(cliff_output_file[id], 'countries' )
    
        preds_coords['city'].append( pred_cities)
        preds_coords['region'].append( pred_cities + pred_states )
        preds_coords['country'].append( pred_countries)
        preds_coords['continent'].append( pred_countries)

    for k in preds_coords:
        gcd_output = calc_gcd_accuracy_tk(targets_coords, preds_coords[k], err_level=error_levels_ref[k])
        gcds[f'T{v}'][k] = gcd_output
        print_table += f'{gcd_output } & ' 
    
    print(f'T{v}', gcds[f'T{v}'])

print('Cliff-clavin', print_table)
save_file(f'{ROOT}/output/CLIFF-clavin_gcd_values.json', gcds)
