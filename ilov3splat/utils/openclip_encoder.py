from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision

try:
    import open_clip
except ImportError as exc:
    raise ImportError("open-clip-torch is required for language training: pip install open-clip-torch") from exc

from ilov3splat.utils.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from nerfstudio.viewer.viewer_elements import ViewerText


@dataclass
class OpenCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)
    device: str = "cuda:0"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.clip_model_type, self.clip_model_pretrained)


class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
            device=self.config.device,
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to(self.config.device)
        self.clip_n_dims = self.config.clip_n_dims

        self.positive_input = ViewerText("Positives", "", cb_hook=self.gui_cb)
        self.positives = list(self.config.positives)

        self.negatives = self.config.negatives
        with torch.no_grad():
            if len(self.positives) > 0 and self.positives != [""]:
                tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.config.device)
                self.pos_embeds = model.encode_text(tok_phrases)
                self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
            else:
                self.pos_embeds = torch.empty((0, self.clip_n_dims), device=self.config.device)

            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(self.config.device)
            self.neg_embeds = model.encode_text(tok_phrases)
            self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = [t.strip() for t in text_list if t.strip()]
        if len(self.positives) == 0:
            self.pos_embeds = torch.empty((0, self.clip_n_dims), device=self.config.device)
            return

        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.config.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        if len(self.positives) == 0 or self.pos_embeds.shape[0] == 0:
            return torch.zeros(embed.shape[0], 2, device=embed.device)

        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0).to(self.config.device)
        p = phrases_embeds.to(embed.dtype)
        embed = embed.to(p.device)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
