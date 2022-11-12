import torch
import torch.nn as nn
from args import get_parser
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# read parser
parser = get_parser()
args = parser.parse_args()

img_dim = args.img_dim

if args.model_name in ['v_clip', 'reg_v_clip']:
    img_dim = args.clip_dim

if args.model_name in [ 'v_clip_loc', 'v_clip_scene']:
    img_dim = args.clip_dim + args.img_dim

if args.model_name in ['v_loc_scene', 'v_loc_obj', 'v_scene_obj']:
    img_dim = 2 * args.img_dim

if args.model_name in  ['v_loc_scene_obj']:
    img_dim = 3 * args.img_dim

if args.model_name in  ['v_clip_loc_scene' ]:
    img_dim = args.clip_dim + 2 * args.img_dim

class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)

# embed images
class LearnImages(nn.Module):
    def __init__(self,dropout=args.dropout):
        super(LearnImages, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(img_dim, args.emb_dim),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.Tanh(),
            Norm()
        )

    def forward(self, x):
        r = self.embedding(x)
        return r


# combine network
class CombineNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(CombineNet, self).__init__()
        self.combine_net = nn.Sequential(
            nn.Linear(args.emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, tags)

        )

    def forward(self, x):

        cn = self.combine_net(x)
        
        # cn2 = cn.squeeze(1)
        return cn


class Geo_base(nn.Module):
    def __init__(self):
        super(Geo_base, self).__init__()

        self.learn_image = LearnImages()

        if 'reg' in args.model_name:
            self.combine_net = CombineNet(tags = 2)
        else:
            self.combine_net = CombineNet(tags = args.n_classes)

    def forward(self, batch):
        # input embeddings
        if args.model_name in ['v_clip', 'reg_v_clip']:
            image_emb0 = batch['clip']
        elif args.model_name == 'v_scene':
            image_emb0 = batch['scene']
        elif args.model_name == 'v_loc':
            image_emb0 = batch['loc']
        elif args.model_name == 'v_obj':
            image_emb0 = batch['obj']

        elif args.model_name == 'v_clip_loc':
            image_emb0 = torch.cat( (batch['clip'], batch['loc']), 1)
        elif args.model_name == 'v_clip_scene':
            image_emb0 = torch.cat( (batch['clip'], batch['scene']), 1)
        elif args.model_name == 'v_loc_scene':
            image_emb0 = torch.cat( (batch['loc'], batch['scene']), 1)
        elif args.model_name == 'v_loc_obj':
            image_emb0 = torch.cat( (batch['loc'], batch['obj']), 1)
        elif args.model_name == 'v_scene_obj':
            image_emb0 = torch.cat( (batch['scene'], batch['obj']), 1)

        elif args.model_name == 'v_loc_scene_obj':
            image_emb1 = torch.cat( (batch['loc'], batch['scene']), 1)
            image_emb0 = torch.cat( (image_emb1, batch['obj']), 1)
        elif args.model_name == 'v_clip_loc_scene':
            image_emb1 = torch.cat( (batch['scene'], batch['loc']), 1)
            image_emb0 = torch.cat( (image_emb1, batch['clip']), 1)

        image_emb = self.learn_image(image_emb0.to(device))

        return self.combine_net(image_emb)
