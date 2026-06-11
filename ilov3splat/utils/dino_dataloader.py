import typing

import torch
from ilov3splat.utils.dino_extractor import ViTExtractor
from ilov3splat.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from torchvision import transforms
from typing import Tuple
import numpy as np

#usually 1260 max size, 1050 for vit-L with ROI
MAX_DINO_SIZE = 1260
def get_img_resolution(H, W, p=14):
    if H<W:
        new_W = MAX_DINO_SIZE
        new_H = (int((H/W)*MAX_DINO_SIZE)//p)*p
    else:
        new_H = MAX_DINO_SIZE
        new_W = (int((W/H)*MAX_DINO_SIZE)//p)*p
    return new_H, new_W

class DinoDataloader(FeatureDataloader):
    dino_model_type = "dinov2_vits14"
    dino_stride = 14
    dino_layer = 11
    dino_facet = "key"
    dino_bin = False

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
        pca_dim: int = 16,
    ):
        assert "image_shape" in cfg
        self.extractor = ViTExtractor(self.dino_model_type, self.dino_stride)
        self.device = device
        self.pca_dim = pca_dim
        super().__init__(cfg, device, image_list, cache_path)
        print("DINO shape: ", self.data.shape)
        
        
           
    def create(self, image_list):
        self.data = self.get_dino_feats(image_list)
        data_shape = self.data.shape
        if self.pca_dim != self.data.shape[-1]:
            self.pca_matrix = torch.pca_lowrank(self.data.view(-1, data_shape[-1]), q=self.pca_dim,niter=20)[2]
            self.data = torch.matmul(self.data.view(-1, data_shape[-1]), self.pca_matrix).reshape((*data_shape[:-1], self.pca_dim))
        else:
            self.pca_matrix = torch.eye(data_shape[-1])

    def load(self):
        super().load()
        cache_pca_path = self.cache_path.parent / ("pca.npy")
        self.pca_matrix = torch.from_numpy(np.load(cache_pca_path)).to(self.device)

    def save(self):
        super().save()
        cache_pca_path = self.cache_path.parent / ("pca.npy")
        np.save(cache_pca_path, self.pca_matrix.cpu().numpy())

    def get_dino_feats(self,image_list, keep_cuda=False):
        h,w = get_img_resolution(image_list.shape[2], image_list.shape[3])
        preprocess = transforms.Compose([
                        transforms.Resize((h,w),antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
        preproc_image_lst = preprocess(image_list).to(self.device)
        dino_embeds = []

        for image in tqdm(preproc_image_lst, desc="dino", total=len(image_list), leave=False):
            with torch.no_grad():
                descriptors = self.extractor.model.get_intermediate_layers(image.unsqueeze(0),reshape=True)[0].squeeze().permute(1,2,0)/10
                if keep_cuda:
                    dino_embeds.append(descriptors)
                else:
                    dino_embeds.append(descriptors.cpu().detach())
                
        return torch.stack(dino_embeds, dim=0)
    
    def get_pca_feats(self,image_list, keep_cuda = False):
        feats = self.get_dino_feats(image_list, keep_cuda=keep_cuda)    # (B, H', W', C)
        data_shape = feats.shape
        pca_feats = torch.matmul(feats.view(-1, data_shape[-1]),        # (B * H' * W'), C
                                 self.pca_matrix.to(feats)              # (C, pca_dim)
                                 ).reshape((*data_shape[:-1], self.pca_dim))    # # (B, H', W', pca_dim)
        return pca_feats
    
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

    def get_full_img_feats(self, img_ind) -> torch.Tensor:
        """
        returns BxHxWxC
        """
        return self.data[img_ind].to(self.device)