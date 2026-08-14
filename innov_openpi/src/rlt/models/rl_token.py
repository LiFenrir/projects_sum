"""RL Token encoder-decoder model (Stage 1).

Compresses variable-length VLA prefix embeddings z_{1:M} into a single
fixed-size RL token z_rl via an information bottleneck, and reconstructs
the original embeddings to train the bottleneck via masked MSE loss.

Paper reference: "RL Token: Bootstrapping Online RL with VLA Models"
"""

import torch
import torch.nn as nn
from torch import Tensor


class RLTokenEncoder(nn.Module):
    """Encode VLA embeddings into a single RL token.

    Appends a learnable e_rl token to z_{1:M}, processes through a
    TransformerEncoder, and extracts the output at the RL position.

    Args:
        embedding_dim: Dimension of VLA embeddings (default 2048).
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.e_rl = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Encode VLA embeddings into z_rl.

        Args:
            z: VLA embeddings [B, M, D] (should be detached / stop-grad).
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            z_rl: RL token [B, D].
        """
        B = z.shape[0]

        # Append learnable e_rl token: [B, M+1, D]
        e_rl = self.e_rl.expand(B, -1, -1)
        tokens = torch.cat([z, e_rl], dim=1)

        # Extend pad_mask for the RL token (always valid)
        rl_mask = torch.ones(B, 1, dtype=torch.bool, device=z.device)
        extended_pad_mask = torch.cat([pad_mask, rl_mask], dim=1)

        # TransformerEncoder uses src_key_padding_mask where True = IGNORE
        # Our pad_mask has True = valid, so invert it
        ignore_mask = ~extended_pad_mask

        out = self.transformer(tokens, src_key_padding_mask=ignore_mask)

        # Extract RL token output (last position)
        z_rl = out[:, -1, :]  # [B, D]
        return z_rl


class RLTokenDecoder(nn.Module):
    """Reconstruct VLA embeddings from the RL token.

    Uses teacher-forced input [z_rl, z_1, ..., z_{M-1}] with causal masking.
    Cross-attends to z_rl as memory. Linear projection h_phi maps back to
    embedding space.

    Args:
        embedding_dim: Dimension of VLA embeddings (default 2048).
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.h_phi = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z_rl: Tensor, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Reconstruct VLA embeddings from z_rl.

        Teacher-forced input: [z_rl, z_1, ..., z_{M-1}] (shifted right).
        Output at position i predicts z_i.

        Args:
            z_rl: RL token [B, D].
            z: Original VLA embeddings [B, M, D] (stop-grad).
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            z_hat: Reconstructed embeddings [B, M, D].
        """
        # Teacher-forced input: [z_rl, z_1, ..., z_{M-1}]
        # Position 0 input = z_rl, position i input = z_{i-1}
        # Output at position i should reconstruct z_i
        tgt = torch.cat([z_rl.unsqueeze(1), z[:, :-1, :]], dim=1)  # [B, M, D]

        # Causal mask: position i can only attend to positions <= i
        M = tgt.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M,
            device=tgt.device,
        )

        # Memory = z_rl as a single token for cross-attention
        memory = z_rl.unsqueeze(1)  # [B, 1, D]

        # Padding mask for target: float mask (-inf = IGNORE, 0.0 = valid)
        # Must match type of tgt_mask (float) to avoid deprecated warning
        tgt_key_padding_mask = torch.where(pad_mask, 0.0, float("-inf"))

        out = self.transformer(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        z_hat = self.h_phi(out)  # [B, M, D]
        return z_hat


class RLTokenModel(nn.Module):
    """Combined RL token encoder-decoder for Stage 1 training.

    Training: forward(z, pad_mask) → (loss, z_rl, z_hat)
    Inference: encode(z, pad_mask) → z_rl

    Args:
        embedding_dim: Dimension of VLA embeddings.
        encoder_layers: Number of encoder transformer layers.
        encoder_heads: Number of encoder attention heads.
        decoder_layers: Number of decoder transformer layers.
        decoder_heads: Number of decoder attention heads.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = RLTokenEncoder(embedding_dim, encoder_layers, encoder_heads)
        self.decoder = RLTokenDecoder(embedding_dim, decoder_layers, decoder_heads)

    def forward(
        self,
        z: Tensor,
        pad_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Training forward pass: encode, decode, compute reconstruction loss.

        Args:
            z: VLA embeddings [B, M, D] (will be detached internally).
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            loss: Masked MSE reconstruction loss (scalar).
            z_rl: Encoded RL token [B, D].
            z_hat: Reconstructed embeddings [B, M, D].
        """
        # Stop gradient on VLA embeddings
        z = z.detach()

        z_rl = self.encoder(z, pad_mask)
        z_hat = self.decoder(z_rl, z, pad_mask)

        # Masked MSE: only compute loss on valid (non-padded) positions
        mse = (z_hat - z).pow(2).mean(dim=-1)  # [B, M]
        masked_mse = mse * pad_mask.float()  # zero out padded positions

        # Average over valid tokens
        num_valid = pad_mask.float().sum()
        loss = masked_mse.sum() / num_valid.clamp(min=1.0)

        return loss, z_rl, z_hat

    @torch.no_grad()
    def encode(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Inference-only: extract z_rl without decoding.

        Args:
            z: VLA embeddings [B, M, D].
            pad_mask: Boolean mask [B, M] (True = valid token).

        Returns:
            z_rl: RL token [B, D].
        """
        return self.encoder(z, pad_mask)
