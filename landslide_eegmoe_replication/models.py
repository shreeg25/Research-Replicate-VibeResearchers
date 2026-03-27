import torch
import torch.nn as nn
import torch.nn.functional as F

class Geospatial_To_MoE_Encoder(nn.Module):
    """
    The 'Bridge'. Converts 2D multi-channel geospatial rasters 
    into a 1D sequence for the Transformer/MoE architecture.
    """
    def __init__(self, in_channels=4, embed_dim=512, patch_size=8):
        super().__init__()
        # Using a Conv2d to extract patches (like a Vision Transformer)
        # A 64x64 image with patch_size 8 yields an 8x8 grid = 64 sequence tokens
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (Batch, 4, 64, 64)
        x = self.patch_embed(x) # Shape: (Batch, 512, 8, 8)
        x = x.flatten(2)        # Shape: (Batch, 512, 64)
        x = x.transpose(1, 2)   # Shape: (Batch, 64, 512) - The exact format MoE needs
        return x

class Expert(nn.Module):
    """2-Layer MLP with GELU activation (As specified in EEGMoE)."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class SSMoE_Block(nn.Module):
    """The core Domain-Decoupled Mixture of Experts architecture."""
    def __init__(self, embed_dim=512, num_specific_experts=4, num_shared_experts=2, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k

        # Specific Experts (Top-K Routing)
        self.specific_experts = nn.ModuleList([Expert(embed_dim, embed_dim * 4) for _ in range(num_specific_experts)])
        self.specific_router = nn.Linear(embed_dim, num_specific_experts)

        # Shared Experts (Soft Routing)
        self.shared_experts = nn.ModuleList([Expert(embed_dim, embed_dim * 4) for _ in range(num_shared_experts)])
        self.shared_router = nn.Linear(embed_dim, num_shared_experts)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.embed_dim) 

        # --- 1. Specific MoE (Top-K Routing) ---
        specific_logits = self.specific_router(x_flat)
        specific_probs = F.softmax(specific_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(specific_probs, self.top_k, dim=-1)

        # Calculate Load-Balancing Auxiliary Loss (L_aux)
        tokens_per_expert = torch.zeros(self.num_specific_experts, device=x.device)
        tokens_per_expert.scatter_add_(0, topk_indices.view(-1), torch.ones_like(topk_indices.view(-1), dtype=torch.float))
        fraction_tokens = tokens_per_expert / (x_flat.size(0) * self.top_k)
        prob_per_expert = specific_probs.mean(dim=0)
        aux_loss = self.num_specific_experts * torch.sum(fraction_tokens * prob_per_expert)

        specific_out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_indices = topk_indices[:, i]
            expert_probs = topk_probs[:, i].unsqueeze(-1)
            for j, expert in enumerate(self.specific_experts):
                mask = (expert_indices == j)
                if mask.any():
                    specific_out[mask] += expert_probs[mask] * expert(x_flat[mask])

        # --- 2. Shared MoE (Soft Routing) ---
        shared_logits = self.shared_router(x_flat)
        shared_probs = F.softmax(shared_logits, dim=-1)
        
        shared_out = torch.zeros_like(x_flat)
        for j, expert in enumerate(self.shared_experts):
            prob = shared_probs[:, j].unsqueeze(-1)
            shared_out += prob * expert(x_flat)

        # --- 3. Additive Fusion ---
        final_out = specific_out + shared_out
        return final_out.reshape(batch_size, seq_len, self.embed_dim), aux_loss

class Landslide_EEGMoE(nn.Module):
    """The master model wrapping the encoder, MoE, and classifier."""
    def __init__(self, in_channels=4, embed_dim=512, num_classes=1):
        super().__init__()
        self.encoder = Geospatial_To_MoE_Encoder(in_channels, embed_dim)
        self.moe_block = SSMoE_Block(embed_dim=embed_dim)
        
        # Binary Classification Head (Landslide vs No Landslide)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # 1. Flatten spatial grid into sequence
        x_seq = self.encoder(x) 
        
        # 2. Pass through MoE block
        moe_out, l_aux = self.moe_block(x_seq) 
        
        # 3. Global Average Pooling across the sequence length
        pooled_out = moe_out.mean(dim=1) 
        
        # 4. Predict probability
        prediction = self.classifier(pooled_out)
        
        return prediction, l_aux