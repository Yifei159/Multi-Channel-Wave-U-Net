# -*- coding: utf-8 -*-
"""
Configuration file (Sacred Ingredient) – supports multi-channel training
"""

import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    """
    Base configuration.

    Important parameters
    --------------------
    mono_downmix : bool
        If True, all input audio will be down-mixed to mono before feeding
        into the network.

    num_channels : int
        Number of channels *after* any optional down-mixing step.
        • If mono_downmix=True, this must be 1.  
        • If mono_downmix=False, set it to the actual number of channels you
          wish to train on (usually 2 for stereo).
    """
    # ------------------------------------------------------------------ #
    #   Paths                                                            #
    # ------------------------------------------------------------------ #
    model_config = {
        "estimates_path": "estimates",   # folder must already exist
        "data_path": "data",

        # ------------------------------------------------------------------ #
        #   Training Hyper-parameters                                        #
        # ------------------------------------------------------------------ #
        "model_base_dir": "checkpoints",
        "log_dir": "logs",
        "batch_size": 8,
        "init_sup_sep_lr": 1e-5,
        "epoch_it": 100,
        "cache_size": 500,
        "num_workers": 4,
        "num_snippets_per_track": 1,

        # ------------------------------------------------------------------ #
        #   Wave-U-Net architecture                                          #
        # ------------------------------------------------------------------ #
        "num_layers": 8,
        "filter_size": 15,
        "merge_filter_size": 5,
        "input_filter_size": 15,
        "output_filter_size": 1,
        "num_initial_filters": 16,
        "num_frames": 88064,

        # ------------------------------------------------------------------ #
        #   Audio settings                                                   #
        # ------------------------------------------------------------------ #
        "expected_sr": 22050,
        "mono_downmix": True,     # Set False to keep original channel count
        "num_channels": 1,        # NEW – desired channel count after loading

        # ------------------------------------------------------------------ #
        #   Network behaviour & training                                     #
        # ------------------------------------------------------------------ #
        "output_type": "direct",
        "output_activation": "tanh",
        "context": False,
        "network": "unet",          # "unet" (time) | "unet_spectrogram" (freq)
        "upsampling": "linear",
        "task": "voice",            # 'voice' | 'multi_instrument' | 'custom'
        "augmentation": True,
        "raw_audio_loss": True,     # for unet_spectrogram: False -> L1 spec loss
        "worse_epochs": 15,
    }

    # Random experiment ID
    experiment_id = np.random.randint(0, 1_000_000)

    # ------------------------------------------------------------------ #
    #   Derive source names according to task                            #
    # ------------------------------------------------------------------ #
    if model_config["task"] == "multi_instrument":
        model_config["source_names"] = ["bass", "drums", "other", "vocals"]
    elif model_config["task"] == "voice":
        model_config["source_names"] = ["accompaniment", "vocals"]
    elif model_config["task"] == "custom":
        model_config["source_names"] = ["accompaniment", "vocals"]
    else:
        raise NotImplementedError(f"Unknown task: {model_config['task']}")

    model_config["num_sources"] = len(model_config["source_names"])

    # ------------------------------------------------------------------ #
    #   Sanity checks                                                    #
    # ------------------------------------------------------------------ #
    if model_config["mono_downmix"] and model_config["num_channels"] != 1:
        raise ValueError(
            "With mono_downmix=True, num_channels must be 1. "
            "Either set mono_downmix=False or change num_channels to 1."
        )

# ---------------------------------------------------------------------- #
#   Example of a named configuration                                     #
# ---------------------------------------------------------------------- #
@config_ingredient.named_config
def custom():
    """
    Template for a custom dataset / experiment.
    Override whatever you need here.
    """
    model_config = {
        "task": "custom",
        "data_path": "multi_yen_2-4/train",
        # For stereo custom dataset:
        # "mono_downmix": False,
        # "num_channels": 2,
    }