"""Budowa sieci z config.json (te same reguły co training/t2l/builders.py w repo treningowym)."""

from __future__ import annotations

from .modele.dp_network import DPNetwork
from .modele.latent_decoder import LatentDecoder1D
from .modele.text_encoder import TextEncoder
from .modele.uncond import UncondParams
from .modele.vf_estimator import VectorFieldEstimator
from .slownik import VOCAB_SIZE


def zbuduj(cfg: dict):
    """-> (text_encoder, vf_estimator, uncond, dp, dekoder) z sekcji ttl, dp i dekoder."""
    ttl = cfg["ttl"]
    te_cfg = ttl["text_encoder"]
    d_model = te_cfg["text_embedder"]["char_emb_dim"]
    se = ttl["style_encoder"]["style_token_layer"]
    spte = ttl["speech_prompted_text_encoder"]
    layer_scale_init = ttl.get("layer_scale_init", 1e-6)
    edge_fill = ttl.get("edge_fill", False)
    key_mode = spte.get("style_key_mode", "baked")

    text_encoder = TextEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_conv_layers=te_cfg["convnext"]["num_layers"],
        n_attn_layers=te_cfg["attn_encoder"]["n_layers"],
        expansion_factor=te_cfg["convnext"]["intermediate_dim"] // d_model,
        p_dropout=te_cfg["attn_encoder"]["p_dropout"],
        kernel_size=te_cfg["convnext"].get("ksz", 5),
        dilation_lst=te_cfg["convnext"].get("dilation_lst", [1] * te_cfg["convnext"]["num_layers"]),
        attn_n_heads=te_cfg["attn_encoder"]["n_heads"],
        attn_filter_channels=te_cfg["attn_encoder"]["filter_channels"],
        spte_n_heads=spte["n_heads"],
        spte_text_dim=spte.get("text_dim", 256),
        spte_style_dim=spte.get("style_dim", 256),
        spte_n_units=spte.get("n_units", 256),
        spte_n_style=se["n_style"],
        layer_scale_init=layer_scale_init,
        edge_fill=edge_fill,
        spte_key_mode=key_mode,
        context_chars=te_cfg.get("context_chars", 0),
        attn_xsa=bool(te_cfg["attn_encoder"].get("xsa", False)),
    )

    vf = ttl["vector_field"]
    mb = vf["main_blocks"]
    vf_estimator = VectorFieldEstimator(
        in_channels=vf["proj_in"]["ldim"] * vf["proj_in"]["chunk_compress_factor"],
        out_channels=vf["proj_out"]["ldim"] * vf["proj_out"]["chunk_compress_factor"],
        hidden_channels=vf["proj_in"]["odim"],
        text_dim=mb["text_cond_layer"]["text_dim"],
        style_dim=mb["style_cond_layer"]["style_dim"],
        num_style_tokens=se["n_style"],
        num_superblocks=mb["n_blocks"],
        time_embed_dim=vf["time_encoder"]["time_dim"],
        rope_gamma=float(mb["text_cond_layer"]["rotary_scale"]),
        main_blocks_cfg=mb,
        last_convnext_cfg=vf.get("last_convnext", {}),
        text_n_heads=mb["text_cond_layer"]["n_heads"],
        time_hdim=vf["time_encoder"].get("hdim", 256),
        use_residual=mb["text_cond_layer"].get("use_residual", True),
        rotary_base=mb["text_cond_layer"].get("rotary_base", 10000.0),
        text_n_units=mb["text_cond_layer"].get("n_units", 256),
        supertonic3_time_freqs=vf["time_encoder"].get("supertonic3_freqs", False),
        layer_scale_init=layer_scale_init,
        edge_fill=edge_fill,
    )
    if key_mode == "projected":
        vf_estimator.tile = text_encoder.speech_prompted_text_encoder.style_key   # jeden wspólny parametr

    um = ttl["uncond_masker"]
    uncond = UncondParams(text_dim=um["text_dim"], n_style=um["n_style"], style_value_dim=um["style_value_dim"],
                          init_std=um["std"], uncond_key=um.get("style_key_uncond", False),
                          style_key_dim=um.get("style_key_dim", um["style_value_dim"]))

    dp_cfg = cfg.get("dp", {})
    stl = dp_cfg.get("style_encoder", {}).get("style_token_layer", {})
    dp = DPNetwork(vocab_size=VOCAB_SIZE, style_dp=int(stl.get("n_style", 8)), style_dim=int(stl.get("style_value_dim", 16)),
                   sentence_encoder_cfg=dp_cfg.get("sentence_encoder"), style_encoder_cfg=dp_cfg.get("style_encoder"),
                   predictor_cfg=dp_cfg.get("predictor"), edge_fill=dp_cfg.get("edge_fill", False))
    dp.ref_encoder = None   # styl predyktora długości jest policzony z góry dla każdego głosu

    dekoder = LatentDecoder1D(cfg=dict(cfg["dekoder"]))
    return text_encoder, vf_estimator, uncond, dp, dekoder
