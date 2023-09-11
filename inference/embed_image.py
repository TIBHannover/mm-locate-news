from PIL import Image
import clip
import sys
sys.path.insert(1, '../mm-locate-news')
from inference.image_scene_features import SceneEmbedding
from inference.image_location_features import LocationEmbedding
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
device = 'cpu'

def get_scene_feature(img_path):
    embedder = SceneEmbedding()
    im_pil = Image.open(img_path).convert('RGB')

    return embedder.embed(im_pil)

def get_location_feature(img_path):

    embedder = LocationEmbedding()
    im_pil = Image.open(img_path).convert('RGB')
    return embedder.embed(im_pil)

def get_obj_feature(img_path):
    preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    ])
    resnet152 = models.resnet152(pretrained=True)
    resnet152 = nn.Sequential(*list(resnet152.children())[:-1])
    im_pil = Image.open(img_path).convert('RGB')
    image_tensor = preprocess(im_pil)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        resnet152.eval()  # Set the model to evaluation mode
        image_emb = resnet152(image_tensor)
    
    return image_emb.squeeze()

def get_clip_image_feature(img_path):
    image = Image.open(img_path)
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    image = preprocess(image).unsqueeze(0).to(device)
    outputs = model.encode_image(image)
    image_emb = outputs.squeeze().detach().cpu().numpy()
    return image_emb

# image_path = '/data/1/mmm_test/mm-locate-news/inference/Q7890669_5.jpg'
# print( get_obj_feature(image_path).shape )