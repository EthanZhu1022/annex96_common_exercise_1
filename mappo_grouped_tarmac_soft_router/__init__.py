"""
mappo_grouped_tarmac_soft_router - Grouped MAPPO with hybrid TarMAC
communication and state-conditioned soft actor routing.

The legacy training/evaluation pipeline mirrors `mappo_grouped_tarmac_hybrid`.
The `three_stage` schedule uses a shared actor encoder, static head pretraining,
router-only training, and fully dynamic actor fine-tuning.  Sharing the encoder
keeps every expert head in one latent coordinate system.

The `pretrained_full_expert` schedule instead loads the original grouped model
and routes complete encoder/communication/head expert branches.  All branches
reuse one TarMAC parameter set, so matched encoder/head latent spaces are kept
without discarding the pretrained checkpoint.
"""
