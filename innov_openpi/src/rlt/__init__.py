"""RL Token: Bootstrapping Online RL with Vision-Language-Action Models.

Built on OpenPI. Provides:

- RL token encoder-decoder (Stage 1): compress VLA prefix embeddings into a
  single fixed-size RL token via masked MSE reconstruction.
- Online RL (Stage 2): TD3 + BC regularizer + reference action dropout,
  training a lightweight residual Actor and TwinQCritic on top of a frozen VLA.
"""
