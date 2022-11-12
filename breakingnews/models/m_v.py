import torch
import torch.nn as nn
from breakingnews.bn_args import get_parser
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# read parser
parser = get_parser()
args = parser.parse_args()


class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)

# embed images
class LearnImages(nn.Module):
    def __init__(self,  dropout=args.dropout):
        super(LearnImages, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(args.img_dim, args.emb_dim),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, 64),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.embedding(x.to(device))


class Geo_base(nn.Module):
    def __init__(self):
        super(Geo_base, self).__init__()

        self.learn_image = LearnImages()
        self.fc1 = torch.nn.Linear(64, 2)

    def forward(self, batch):
        # input embeddings
        image_emb0 = batch['clip']

        image_emb = self.learn_image(image_emb0.to(device))
        output = self.fc1(image_emb)

        return output
