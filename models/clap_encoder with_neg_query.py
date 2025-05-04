import random
import torch
import torch.nn as nn
import torchaudio
from models.CLAP.open_clip import create_model
from models.CLAP.training.data import get_audio_features
from transformers import RobertaTokenizer


class CLAP_Encoder(nn.Module):
    """
    Wrapper around the CLAP audio-text model that can now take a **positive
    caption** (`text`) and an optional **negative caption** (`text_neg`).

    Fusion rule (fixed-order concat → 512-D projection):

        embed = Linear([pos_emb, neg_emb])   # shape = (B,512)

    If `text_neg` is None the behaviour is identical to the original code.
    """

    def __init__(
        self,
        pretrained_path="checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt",
        sampling_rate=32000,
        amodel="HTSAT-base",
    ):
        super().__init__()
        self.device = "cpu"
        self.precision = "fp32"
        self.amodel = amodel
        self.tmodel = "roberta"
        self.enable_fusion = False
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.sampling_rate = sampling_rate
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")

        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.encoder_type = "CLAP"

    
    def batch_to_list(self, batch):
        return [batch[i] for i in range(batch.size(0))]

    def _get_audio_embed(self, batch):
        with torch.no_grad():
            audio_dict_list = []
            assert (
                self.sampling_rate == 32000
            ), "We only support 32000 sampling rate"

            batch = torchaudio.functional.resample(
                batch, orig_freq=self.sampling_rate, new_freq=48000
            )
            for waveform in self.batch_to_list(batch):
                audio_dict = get_audio_features(
                    {},
                    waveform,
                    480000,
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                audio_dict_list.append(audio_dict)
            embed = self.model.get_audio_embedding(audio_dict_list)
            return embed.detach()

    def _get_text_embed(self, texts):
        double_batch = False
        if len(texts) == 1:
            texts = texts * 2
            double_batch = True
        with torch.no_grad():
            text_data = self.tokenizer(texts)
            embed = self.model.get_text_embedding(text_data)
        return embed[0:1] if double_batch else embed.detach()

    
    def get_query_embed(
        self,
        modality,
        audio=None,
        text=None,
        text_neg=None,                # NEW
        use_text_ratio=0.5,
        device=None,
    ):
        """
        Args
        ----
        modality : 'audio' | 'text' | 'hybird'
        text     : positive caption(s)  – list[str] or tuple[str]
        text_neg : negative caption(s)  – same length list[str] (optional)
        """

        if modality == "audio":
            embed = self._get_audio_embed(audio)

        elif modality == "text":
            embed = self._fuse_texts(text, text_neg)

        elif modality == "hybird":
            # choose between audio or (pos,neg) caption each call
            if random.random() > use_text_ratio:
                embed = self._get_audio_embed(audio)
            else:
                embed = self._fuse_texts(text, text_neg)
        else:
            raise NotImplementedError("Unknown modality flag.")

        return embed.float()

    #fuse pos and neg queries
    def _fuse_texts(self, text_pos, text_neg):
        """
        Return a 512-D embedding.
        If `text_neg` is None this falls back to single-caption behaviour.
        """
        if text_neg is None:
            return self._get_text_embed(text_pos)

        pos_emb = self._get_text_embed(text_pos)     # (B,512)
        neg_emb = self._get_text_embed(text_neg)     # (B,512)
        fused = torch.cat([pos_emb, neg_emb], dim=-1)  # (B,1024)

        # lazy creation of projection layer (shared across calls)
        if not hasattr(self, "fuse"):
            self.fuse = nn.Linear(1024, 512, bias=False)
        return self.fuse(fused).detach()


    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}
