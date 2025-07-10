import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
# from utils import get_cosine_schedule_with_warmup
import numpy as np
 
class TransformerRawClassifier(pl.LightningModule):
    def __init__(self, config, optim_cfg, pre_process):
        super().__init__()
        self.config = config
        self.optim_cfg = optim_cfg
        self.pre_process = pre_process
        self.save_hyperparameters()
 
        self.ecg_len = 130
        self.gaze_len = 10
        self.aux_len = 1  # e.g., ECG mean
        self.seq_len = self.ecg_len + self.gaze_len + self.aux_len
 
        self.input_dim = config.get("input_dim")
        self.d_model = config.get("dim_model")
        self.nhead = config.get("num_heads")
        self.num_layers = config.get("num_layers")
        self.dim_feedforward = config.get("dim_feedforward")
        self.num_classes = config.get("num_classes")
        self.dropout = config.get("dropout")
        self.seq_len = config.get("max_len")
 
        self.input_proj = nn.Linear(1, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.seq_len)
 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
 
        self.classifier = nn.Linear(self.d_model, self.num_classes)
 
    def forward(self, a1, a2, pre_process=None):
        """
        a1: [B, *, 130] ECG
        a2: [B, *, 8] Gaze
        """
        ecg_mean = a1.mean(dim=-1, keepdim=True)  # [B, 1]
        x = torch.cat([a1, a2, ecg_mean], dim=-1)  # [B, 139]
        x = x.view(x.size(0), x.size(1), 1)        # [B, 139, 1]
        x = self.input_proj(x)  # [B, 139, d_model]
        x = self.pos_encoder(x)
 
        encoded = self.transformer_encoder(x)  # [B, 139, d_model]
        pooled = encoded.mean(dim=1)  # [B, d_model]
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits
 
    def training_step(self, batch, batch_idx):
        t1, t2, labels = batch
        logits = self.forward(t1, t2, self.pre_process)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss
 
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        t1, t2, labels = batch
        logits = self.forward(t1, t2, self.pre_process)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
 
    def configure_optimizers(self):
        lr = float(self.optim_cfg["lr"])
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=1e-3)
        warmup_steps = int(self.optim_cfg["warmup_steps"])
        total_steps = self.optim_cfg["num_steps_per_epoch"] * self.optim_cfg["num_epochs"]
        scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr=self.optim_cfg["min_lr"],
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
 
    def freeze_backbone(self):
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
 
    def freeze_astencoder_most(self):
        self.freeze_backbone()
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # x: [B, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
 