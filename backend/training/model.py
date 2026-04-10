import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel

class TruthGuardMultiTaskModel(PreTrainedModel):
    def __init__(self, config, model_name_or_path):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        hidden_size = config.hidden_size

        # Multi-head architecture
        # Truth Head (Regression 0-1)
        self.truth_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # AI Detection Head (Binary Classification)
        self.ai_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )

        # Bias Head (Multi-class Classification: e.g., low, medium, high - 3 categories)
        self.bias_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Use pooler_output (for models that have it) or CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]

        truth_logits = self.truth_head(pooled_output)
        ai_logits = self.ai_head(pooled_output)
        bias_logits = self.bias_head(pooled_output)

        return {
            "truth": truth_logits,
            "ai": ai_logits,
            "bias": bias_logits
        }
