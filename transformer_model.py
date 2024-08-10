import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, embedding_matrix, num_heads=5, num_layers=2):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        
        self.position_embedding = nn.Embedding(512, embed_size)  
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x, mask):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        
        embedded = self.embedding(x) + self.position_embedding(position_ids)
        embedded = embedded.permute(1, 0, 2)  # Transformer expects input shape: (seq_length, batch_size, embed_size)
        
        # Apply the mask
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        
        transformer_output = transformer_output.permute(1, 0, 2)  # Back to (batch_size, seq_length, embed_size)
        
        pooled_output = F.avg_pool1d(transformer_output.transpose(1, 2), kernel_size=transformer_output.size(1)).squeeze(2)  # (batch_size, embed_size)
        
        out = self.fc(pooled_output)  # (batch_size, num_classes)
        return out

