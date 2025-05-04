from typing import Any, Callable, Dict
import random
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from models.clap_encoder import CLAP_Encoder
from huggingface_hub import PyTorchModelHubMixin


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        ss_model: nn.Module = None,
        waveform_mixer=None,
        query_encoder: nn.Module = CLAP_Encoder().eval(),
        loss_function=None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func=None,
        use_text_ratio: float = 1.0,
    ):
        super().__init__()
        self.ss_model = ss_model
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

    
    def training_step(self, batch_data_dict, batch_idx):
        """
        Args:
            batch_data_dict['audio_text'] contains:
              ├─ 'text'                       : List[str]  (positive caption)
              ├─ 'mixture_component_texts'    : List[List[str]]
              └─ 'waveform'                   : Tensor (B,1,T)
        """
        random.seed(batch_idx)  

        bdict = batch_data_dict["audio_text"]
        batch_text_pos = bdict["text"]                      # positive

      
        mix_lists = bdict["mixture_component_texts"]        # List[List[str]]
        batch_text_neg = [
            lst[1] if isinstance(lst, (list, tuple)) and len(lst) > 1 else ""
            for lst in mix_lists
        ]                                                 
        batch_audio = bdict["waveform"]                     

        mixtures, segments = self.waveform_mixer(waveforms=batch_audio)

        conditions = self.query_encoder.get_query_embed(   
            modality="hybird",
            text=batch_text_pos,
            text_neg=batch_text_neg,
            audio=segments.squeeze(1),
            use_text_ratio=self.use_text_ratio,
        )

        input_dict = {
            "mixture": mixtures[:, None, :].squeeze(1),
            "condition": conditions,
        }
        target_dict = {"segment": segments.squeeze(1)}

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)["waveform"].squeeze()

        loss = self.loss_function({"segment": sep_segment}, target_dict)
        self.log_dict({"train_loss": loss})
        return loss

  
    def forward(self, x):
        pass

    def test_step(self, batch, batch_idx):
        pass


    def configure_optimizers(self):
        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def get_model_class(model_type):
    if model_type == "ResUNet30":
        from models.resunet import ResUNet30

        return ResUNet30
    raise NotImplementedError

