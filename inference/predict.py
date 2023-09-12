import sys
sys.path.insert( 1, '../mm-locate-news' )
from inference.embed_image import *
from inference.embed_text import *
from args import get_parser
from models.m_t import Geo_base as geo_base_t
from models.m_v import Geo_base as geo_base_v
from models.m_vt import Geo_base as geo_base_vt
from utils import *
from pathlib import Path
import os

ROOT_PATH = Path(os.path.dirname(__file__)).parent
parser = get_parser()
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# read test input image-text pair
image_path = args.test_image_path
with open( args.text_input_path ) as f:
    text_input = f.read()

# read locations info
loc_info = open_json(f'{ROOT_PATH}/{args.data_path}/mm-locate-news/loc_info.json')
cls2coord = open_json(f'{ROOT_PATH}/{args.data_path}/mm-locate-news/cls_to_coord.json')
cls2wikidata = open_json(f'{ROOT_PATH}/{args.data_path}/mm-locate-news/location_to_class.json')

# get multimodal features
data_in = {}
if 'clip' in args.model_name:
    data_in['clip'] = get_clip_image_feature(image_path)
if 'loc' in args.model_name:
    data_in['loc'] = get_location_feature(image_path)
if 'obj' in args.model_name:
    data_in['obj'] = get_obj_feature(image_path)
if 'scene' in args.model_name:
    data_in['scene'] = get_scene_feature(image_path)
if 'body' in args.model_name or '2bert' in args.model_name:
    data_in['body'] = get_bert_body_feature(text_input)
if 'entity' in args.model_name or '2bert' in args.model_name:
    data_in['entity'] = get_bert_entity_feature(text_input) 

# set the model
model = {'v_clip':geo_base_v(), 'v_loc':geo_base_v(), 'v_scene':geo_base_v(), 'v_obj':geo_base_v(), 'v_clip_loc':geo_base_v(), 'v_clip_scene':geo_base_v(), 'v_loc_scene':geo_base_v(), 'v_loc_scene_obj':geo_base_v() , 'v_clip_loc_scene':geo_base_v(),'v_loc_obj':geo_base_v(), 'v_scene_obj':geo_base_v(),
         't_body':geo_base_t(), 't_entity':geo_base_t() ,'t_2bert':geo_base_t(),
         'm_body_clip':geo_base_vt(), 'm_entity_clip':geo_base_vt(), 'm_2bert_clip':geo_base_vt(),'m_2bert_clip_scene':geo_base_vt(), 'm_2bert_clip_loc':geo_base_vt() ,  'm_2bert_loc_scene':geo_base_vt(), 'm_2bert_clip_loc_scene':geo_base_vt(),
        }[args.model_name]

# get output predictions
checkpoint_path = f'{ROOT_PATH}/{args.snapshots}/{args.model_name}'
model.to(device)
model_path = f'{checkpoint_path}/{args.test_check_point}'
checkpoint = torch.load(model_path, encoding='latin1', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
[output_classify] = model(data_in)

# top-k results
p_cls = torch.topk(output_classify, k=10)[1]
p_vals = torch.topk(output_classify, k=10)[0]
vals = [ v.item() for v in p_vals ] 
topk_preds = [v.item() for v in p_cls]
topk_preds_wikidata = [list(cls2wikidata.keys())[topk_pred] for topk_pred in topk_preds]
topk_preds_coords = [cls2coord[str(topk_pred)] for topk_pred in topk_preds]


for k, topk_pred in enumerate( topk_preds_wikidata ):
    print(args.model_name, 'top', k+1, 'prediction: ', loc_info[topk_pred]['city']['label'], ' - ' , loc_info[topk_pred]['country']['label'],' - ' , loc_info[topk_pred]['continent']['label'] )
