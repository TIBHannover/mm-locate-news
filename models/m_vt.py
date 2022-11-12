import torch
import torch.nn as nn

# d = 512
# batch = 10
# m = nn.Conv1d(in_channels = d, out_channels = 2, kernel_size=1, stride=2)
# input = torch.randn(batch, d).unsqueeze(2)
# output = m(input)
from args import get_parser
# read parser



parser = get_parser()
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
txt_dim = args.txt_dim
img_dim = args.img_dim

if args.model_name in [ 'reg_mm', 'reg_mm_loc_scene', 'm_2bert_clip', 'm_2bert_clip_loc_scene','m_2bert_clip_scene', 'm_2bert_clip_loc', 'm_2bert_loc_scene' ]:
    txt_dim = 2 * args.txt_dim  

if args.model_name == 'm_2bert_clip_loc_scene' :
    img_dim = args.clip_dim + 2 * args.img_dim 

if args.model_name in ['m_2bert_loc_scene', 'reg_mm_loc_scene'] :
    img_dim = 2 * args.img_dim 

if args.model_name in ['m_2bert_clip', 'reg_mm', 'reg_mm_body_clip', 'reg_mm_entity_clip' ]:
    img_dim = args.clip_dim

if args.model_name in ['m_2bert_clip_loc', 'm_2bert_clip_scene' ]:
    img_dim = args.clip_dim + args.img_dim 

class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)


# embed images
class LearnImages(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(LearnImages, self).__init__()


        self.embedding = nn.Sequential(
            nn.Linear(img_dim, args.emb_dim),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, tags),
            nn.Tanh(),
            Norm()

        )

    def forward(self, x):
        return self.embedding(x.to(device))

# embed text
class LearnText(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(LearnText, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(txt_dim, args.emb_dim),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, tags),
            nn.Tanh(),
            Norm()
        )

    def forward(self, x):
        return self.embedding(x.float().to(device))

# MLM Baseline model
class Geo_base(nn.Module):
    def __init__(self):
        super(Geo_base, self).__init__()
        if 'reg' in args.model_name:
            self.learn_image = LearnImages(tags=2)
            self.learn_text = LearnText(tags=2)
            self.fc1 = torch.nn.Linear(img_dim + txt_dim, 2)
            # self.fc1 = torch.nn.Linear(img_dim + txt_dim, int( (img_dim + txt_dim)/2 ))
            # self.fc2 = torch.nn.Linear(int( (img_dim + txt_dim)/2 ), int( (img_dim + txt_dim)/4 ))
            # self.fc3 = torch.nn.Linear(int( (img_dim + txt_dim)/4 ), 2)
            self.conv1d = torch.nn.Conv1d(img_dim + txt_dim, 2, 1, stride=2) # input = (batch, d, 1)

        else:
            self.learn_image = LearnImages(tags=args.n_classes)
            self.learn_text = LearnText(tags=args.n_classes)
        

    def forward(self, batch):
        # input embeddings
        
        if args.model_name in ['m_body_clip', 'reg_mm_body_clip']:
            image_emb0 =  batch['clip']
            text_emb0 = batch['body']
        elif args.model_name in [ 'm_entity_clip', 'reg_mm_entity_clip']:
            image_emb0 =  batch['clip']
            text_emb0 = batch['entity']
        elif args.model_name in ['m_2bert_clip', 'reg_mm']:
            text_emb0 = torch.cat( (batch['body'], batch['entity']), 1)            
            image_emb0 =  batch['clip']
        elif args.model_name == 'm_2bert_clip_loc_scene':
            text_emb0 = torch.cat((batch['body'], batch['entity']), 1)
            img_emb1 =  torch.cat((batch['loc'], batch['scene']), 1)
            image_emb0 =  torch.cat((img_emb1, batch['clip']), 1)
        elif args.model_name in ['m_2bert_loc_scene', 'reg_mm_loc_scene']:
            text_emb0 = torch.cat((batch['body'], batch['entity']), 1)
            image_emb0 =  torch.cat((batch['loc'], batch['scene']), 1)
        elif args.model_name == 'm_2bert_clip_scene':
            text_emb0 = torch.cat((batch['body'], batch['entity']), 1)
            image_emb0 =  torch.cat((batch['scene'], batch['clip']), 1)
        elif args.model_name == 'm_2bert_clip_loc':
            text_emb0 = torch.cat((batch['body'], batch['entity']), 1)
            image_emb0 =  torch.cat((batch['loc'], batch['clip']), 1)


        if 'reg' in args.model_name:

            image_emb = self.learn_image(image_emb0)
            text_emb = self.learn_text(text_emb0)
            output = self.conv1d( torch.cat( (image_emb0.float().to(device), text_emb0.float().to(device) ), 1).unsqueeze(2) ).squeeze(2)

            return output

        else:
            image_emb = self.learn_image(image_emb0)
            text_emb = self.learn_text(text_emb0)  
            return torch.max(image_emb, text_emb)
