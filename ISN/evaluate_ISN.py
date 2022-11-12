import sys
sys.path.insert(1, '/data/1/mm-locate-news')
from utils import *
import h5py
from args import get_parser
from pathlib import Path
import os

# read parser
parser = get_parser()
args = parser.parse_args()
ROOT = Path(os.path.dirname(__file__))

cls2coord = open_json(f'{args.data_path}/mm-locate-news/cls_to_coord.json')
gcds = {}
print_table = ""

for v in [1,2,3]:
    h5_test = h5py.File(f'{args.data_path}/mm-locate-news/test{v}.h5', 'r')
    ISN_preds = open_json(f'{ROOT}/output/ISNs_predictions.json')[str(v)]

    targets_coords = []
    preds_coords = []

    for id in ISN_preds:
        cls = h5_test[id]['class'][()][0]
        target_coords = cls2coord[str(cls)]
        targets_coords.append(target_coords)   
        preds_coords.append( [ISN_preds[id]['lng'], ISN_preds[id]['lat']] )

    gcds[f'T{v}'] = calc_gcd_accuracy_tk(targets_coords, preds_coords)
    print_table += f" { gcds[f'T{v}']['city'] } & {gcds[f'T{v}']['region'] } & {gcds[f'T{v}']['country'] } & { gcds[f'T{v}']['continent'] } &"
    print(f'T{v}', gcds[f'T{v}'])

save_file(f'{ROOT}/output/ISNs_gcd_values.json', gcds)
print(print_table)