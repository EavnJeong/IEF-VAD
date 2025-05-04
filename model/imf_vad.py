import torch
import torch.nn as nn


class MMFMIL(nn.Module):
    def __init__(self,
            num_class: int,
            embed_dim: int,
            visual_length: int,
            visual_width: int,
            visual_head: int,
            visual_layers: int,
            attn_window: int,
            prompt_prefix: int,
            prompt_postfix: int,
            device,
            args
        ):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = MultiModal_Fusion_Attn_Iter(
            embed_dim,
            num_layers=args.visual_layers,
            num_heads=args.visual_head,
            num_refinement_steps=args.num_refinement_steps,
            lambda_ref=args.lambda_ref,
            noise_model=args.noise_model,
            nu=args.nu
        )

    def forward(self, img_visual, ev_visual, padding_mask, text, lengths, return_attn=False):
        images = img_visual.to(torch.float)
        events = ev_visual.to(torch.float)

        return self.temporal(images, events)


class MultiModal_Fusion_Attn_Iter(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            num_layers=2, 
            num_heads=8, 
            dropout=0.1, 
            num_refinement_steps=3, 
            lambda_ref=0.5, 
            noise_model="StudentT", 
            nu=5, 
            epsilon=1e-8
        ):
        super(MultiModal_Fusion_Attn_Iter, self).__init__()
        self.num_layers = num_layers
        self.num_refinement_steps = num_refinement_steps
        self.lambda_ref = lambda_ref
        self.noise_model = noise_model
        self.nu = nu
        self.epsilon = epsilon
        
        # Image modality self-attention layers
        self.image_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.image_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Event modality self-attention layers
        self.event_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.event_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Whitening layers
        self.whiten_image = nn.LayerNorm(embed_dim)
        self.whiten_event = nn.LayerNorm(embed_dim)
        
        # Linear layers for predicting mean and log variance for each modality
        self.image_mu = nn.Linear(embed_dim, embed_dim)
        self.event_mu = nn.Linear(embed_dim, embed_dim)
        self.image_logvar = nn.Linear(embed_dim, embed_dim)
        self.event_logvar = nn.Linear(embed_dim, embed_dim)
        
        if num_refinement_steps == 0:
            self.refinement_blocks = nn.ModuleList([
                nn.Identity()
            ])
        else:
            self.refinement_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim)
                )
                for _ in range(num_refinement_steps)
            ])
        
        # Final binary classifier on the refined fused representation
        self.classifier = nn.Linear(embed_dim, 1)
    
    def forward(self, image_features, event_features):
        B, T, D = image_features.shape
        
        # 1. Temporal Encoding
        x_img = image_features  # (B, T, embed_dim)
        for i in range(self.num_layers):
            attn_out, _ = self.image_attn_layers[i](x_img, x_img, x_img)
            x_img = self.image_norms[i](x_img + attn_out)
        image_encoded = self.whiten_image(x_img)
        
        x_evt = event_features  # (B, T, embed_dim)
        for i in range(self.num_layers):
            attn_out, _ = self.event_attn_layers[i](x_evt, x_evt, x_evt)
            x_evt = self.event_norms[i](x_evt + attn_out)
        event_encoded = self.whiten_event(x_evt)
        
        image_mu = self.image_mu(image_encoded)         # (B, T, embed_dim)
        event_mu = self.event_mu(event_encoded)           # (B, T, embed_dim)
        image_logvar = self.image_logvar(image_encoded)   # (B, T, embed_dim)
        event_logvar = self.event_logvar(event_encoded)   # (B, T, embed_dim)
        
        if self.noise_model == "Gaussian":
            weight_image = torch.exp(-image_logvar)
            weight_event = torch.exp(-event_logvar)
        elif self.noise_model == "StudentT":
            factor = (self.nu + 1) / self.nu
            weight_image = factor * torch.exp(-image_logvar)
            weight_event = factor * torch.exp(-event_logvar)
        else:
            raise ValueError("Unsupported noise_model. Choose 'Gaussian' or 'StudentT'.")
        
        denom = weight_image + weight_event + self.epsilon
        norm_weight_image = weight_image / denom
        norm_weight_event = weight_event / denom
        
        fused = norm_weight_image * image_mu + norm_weight_event * event_mu  # (B, T, embed_dim)
        
        fused_refined = fused
        for i in range(self.num_refinement_steps):
            residual = self.refinement_blocks[i](fused_refined)  # (B, T, embed_dim)
            fused_refined = fused_refined - self.lambda_ref * residual
        logits = self.classifier(fused_refined)  # (B, T, 1)
        
        return {
            'fused': fused_refined,
            'logits': logits,
            'image_mu': image_mu,
            'event_mu': event_mu,
            'image_logvar': image_logvar,
            'event_logvar': event_logvar,
            'w_i': norm_weight_image,
            'w_e': norm_weight_event,
        }