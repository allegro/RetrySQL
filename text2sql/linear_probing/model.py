import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score

from text2sql.llm_training.domain.model import LinearProbingConfiguration
from text2sql.commons.logging_utils import get_logger

logger = get_logger(__name__)


class LinearProbingModel(pl.LightningModule):
    def __init__(self, config: LinearProbingConfiguration) -> None:
        super(LinearProbingModel, self).__init__()

        self.config = config

        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.config.backbone_model_name_or_path, trust_remote_code=True
        )
        self.backbone.lm_head = torch.nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.linear_layer = torch.nn.Linear(self.backbone.config.hidden_size, 1)

        self.val_labels = []
        self.val_preds = []
        self.val_losses = []

    def extract_last_token_embeddings(self, input_ids, attention_mask) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.logits

        # Find the index of the last non-padding token for each sequence
        last_non_pad_indices = attention_mask.sum(dim=1) - 1

        # Gather the embeddings of the last non-padding token for each sequence
        batch_size = embeddings.size(0)

        return embeddings[torch.arange(batch_size), last_non_pad_indices]

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        last_token_embeddings = self.extract_last_token_embeddings(input_ids, attention_mask)
        logits = self.linear_layer(last_token_embeddings)
        return logits.squeeze(1)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        logits = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        logits = self(input_ids, attention_mask)
        preds = torch.sigmoid(logits) > 0.5

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.val_preds.append(preds.cpu())
        self.val_labels.append(labels.cpu())
        self.val_losses.append(loss.unsqueeze(0).cpu())

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        val_losses = torch.cat(self.val_losses)

        avg_loss = torch.mean(val_losses)
        positive_label_f1_score = f1_score(y_true=labels.numpy(), y_pred=preds.numpy())
        balanced_acc = balanced_accuracy_score(y_true=labels.numpy(), y_pred=preds.numpy())
        cm = confusion_matrix(y_true=labels.numpy(), y_pred=preds.numpy(), labels=["correct", "wrong"])

        self.log('val_loss', avg_loss)
        self.log('balanced_val_acc', balanced_acc, on_epoch=True)
        self.log("positive_label_val_f1", positive_label_f1_score, on_epoch=True)
        logger.info("Confusion Matrix: \n" + str(cm))

        self.val_preds.clear()
        self.val_labels.clear()
        self.val_losses.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.linear_layer.parameters(), lr=self.config.lr)