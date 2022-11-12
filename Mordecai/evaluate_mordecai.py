from cmath import inf
import sys
from numpy import save
sys.path.insert(1, '/data/1/mm-locate-news')
from utils import *
import h5py
from args import get_parser
from pathlib import Path
import os
import glob


# read parser
parser = get_parser()
args = parser.parse_args()
ROOT = Path(os.path.dirname(__file__))

def get_mordecai_outputs():
    output_file = {}
    files = glob.glob(f'{ROOT}/output/mordecai_predictions/*')
    for f in files:
        file = open_json(f)
        id = f.split('.json')[0].split('/')[-1]
        output_file[id] = file
    return output_file


def calc_gcd_accuracy_tk(t_coords, ps_coords):
    n_total = len(t_coords)

    error_levels_ref = {  'country': 750, 'continent': 2500}
    error_levels = { 'country': 0, 'continent': 0}
    
    for i in range(n_total):
        
        t = t_coords[i]
        ps = ps_coords[i]

        min_gcd = inf

        for p in ps:

            gcd_tp = gcd(float(t[0]),float(t[1]),float(p[0]),float(p[1]) )

            if gcd_tp < min_gcd:
                min_gcd = gcd_tp

        if min_gcd <= error_levels_ref['country']:
            error_levels['country'] +=1
        if min_gcd <= error_levels_ref['continent']:
            error_levels['continent'] +=1
    
    error_levels = {
    'country': np.round(error_levels['country']/len(t_coords)*100,1), 
    'continent': np.round(error_levels['continent']/len(t_coords)*100,1)
    }

    return error_levels


def get_prediction_by_freq(preds_in):
    results = {}
    max_n = 0

    for p in preds_in:
        if 'id' in p:
            geoid = 'id'
        elif 'geoid' in p:
            geoid = 'geoid'
        else: 
            continue

        loc_id = p[geoid]

        if loc_id not in results:
            results[ loc_id ] = {'count':0, 'coords': (None, None)}

        results[ loc_id]['count'] += 1
        results[ loc_id]['coords'] = ( p['lon'], p['lat'] )
        results[ loc_id]['conf'] =  float(p['conf'])
        
        if results[loc_id]['count'] > max_n:
            max_n = results[loc_id]['count']

    # get the max confidence score
    max_conf = -1
    for loc_id in results:
        if results[loc_id]['count'] == max_n and results[loc_id]['conf']  > max_conf:
            max_conf = results[loc_id]['conf']

    # get the predition with maximum confidence
    out_coords = []
    for loc_id in results: 
        if results[loc_id]['conf'] == max_conf:
            out_coords.append( results[loc_id]['coords'] )

    return out_coords


cls2coord = open_json(f'{args.data_path}/mm-locate-news/cls_to_coord.json')
gcds = {}
output_file = get_mordecai_outputs()

print_table = ''

for v in [1, 2, 3]: 

    h5_test = h5py.File(f'{args.data_path}/mm-locate-news/test{v}.h5', 'r')
    v_ids = [str(id) for id in h5_test]
   
    targets_coords = []
    preds_coords = []

    for id in v_ids:
        Mordecai_preds = get_prediction_by_freq( output_file[id] )

        cls = h5_test[id]['class'][()][0]
        target_coords = cls2coord[str(cls)]
        targets_coords.append(target_coords) 
        preds_coords.append(Mordecai_preds) 

    gcds[f'T{v}'] = calc_gcd_accuracy_tk(targets_coords, preds_coords)
    print(f'T{v}', gcds[f'T{v}'])
    print_table += f" - & - & {  gcds[f'T{v}']['country'] } & { gcds[f'T{v}']['continent'] } &"

print('Mordecai', print_table)
save_file(f'{ROOT}/output/Mordecai_gcd_values.json', gcds)
