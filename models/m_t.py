import torch
import torch.nn as nn
from args import get_parser

parser = get_parser()
args = parser.parse_args()

txt_dim = args.txt_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_name in ['t_2bert', 'reg_t']:
    txt_dim = 2 * args.txt_dim

class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)

# embed text
class LearnText(nn.Module):
    def __init__(self,dropout=args.dropout):
        super(LearnText, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(txt_dim, args.emb_dim),
                nn.Flatten(),
                nn.LeakyReLU(),
                nn.Linear(args.emb_dim, args.emb_dim),
                nn.Tanh(),
                Norm()
        )

    def forward(self, x):
        return self.embedding(x.float().to(device))


# combine network
class CombineNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(CombineNet, self).__init__()
        self.combine_net = nn.Sequential(
            nn.Linear(args.emb_dim, 512),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, tags)

        )

    def forward(self, x):
        cn0= x.squeeze(1)
        cn = self.combine_net(cn0)
        # cn2 = cn.squeeze(1)
        return cn


class Geo_base(nn.Module):
    def __init__(self):
        super(Geo_base, self).__init__()

        self.learn_text = LearnText()

        # if 'reg' in args.model_name:
        #     self.combine_net = CombineNet(tags = 2)
        # else:
        self.combine_net = CombineNet(tags = args.n_classes)

    def forward(self, batch):
        # input embeddings
        if args.model_name == 't_body':
            text = batch['body']
        elif args.model_name == 't_entity':
            text = batch['entity']
        elif args.model_name in ['t_2bert', 'reg_t']:
            text = torch.cat( (batch['body'], batch['entity']), 1)

        text_emb = self.learn_text(text)
        
        return self.combine_net(text_emb)

