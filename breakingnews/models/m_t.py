import torch
import torch.nn as nn
import sys
sys.path.insert(1, '/data/1/mm-locate-news') 
from breakingnews.bn_args import get_parser

parser = get_parser()
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)

class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        out = input.view(-1,1, args.emb_dim)
        return out # (batch_size, *size)

class LearnText(nn.Module):
    def __init__(self,  dropout=args.dropout):
        super(LearnText, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(2 * args.txt_dim, out_channels = args.emb_dim, kernel_size=3, stride=1, padding=1),
            Flatten(),
            nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        return self.embedding(x.float().to(device))


class Geo_base(nn.Module):
    def __init__(self):
        super(Geo_base, self).__init__()

        self.learn_text = LearnText()

        self.fc1 = torch.nn.Linear(64, 2)

    def forward(self, batch):
        # input embeddings
        text_emb0 = torch.cat( (batch['body'], batch['entity']), 1)

        text_emb = self.learn_text(text_emb0.unsqueeze(-1) ).squeeze(1)
        
        return self.fc1(text_emb)

