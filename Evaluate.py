import os
import glob
import json
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

import musdb
import museval

import Models.UnetAudioSeparator
import Models.UnetSpectrogramSeparator
import Utils

np.set_printoptions(precision=10, suppress=True)

tf.compat.v1.disable_eager_execution()     # graph mode


# ---------------------------------------------------------------------- #
#                           CORE PREDICTION                              #
# ---------------------------------------------------------------------- #
def predict(track, model_config, load_model, results_dir=None):
    """
    Predict source estimates for a single MUSDB track object (or Track-like).

    Parameters
    ----------
    track : musdb.Track
    model_config : dict
    load_model : str
        Checkpoint path
    results_dir : str or None
        If given, write museval JSON here.

    Returns
    -------
    dict
        {source_name: np.ndarray [T, C]}
    """
    # -------------------------------------------------------------- #
    #        1)  Build separator graph for *inference*               #
    # -------------------------------------------------------------- #
    ### NEW / CHANGED: channel-explicit input shape ---------------- #
    disc_input_shape = [
        1,                                   # batch     (fixed to 1)
        model_config["num_frames"],          # samples
        model_config["num_channels"]         # channels
    ]
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError("Unknown network type")

    sep_in_shape, sep_out_shape = separator_class.get_padding(np.array(disc_input_shape))
    sep_in_shape[0] = sep_out_shape[0] = 1   # batch=1 during inference
    mix_ph = tf.compat.v1.placeholder(tf.float32, sep_in_shape)

    separator_sources = separator_class.get_output(
        mix_ph,
        training=False,
        return_spectrogram=False,
        reuse=False
    )

    # -------------------------------------------------------------- #
    #        2)  Restore weights                                     #
    # -------------------------------------------------------------- #
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    restorer = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                        write_version=tf.compat.v1.train.SaverDef.V2)
    print("Variables in graph:", len(tf.compat.v1.global_variables()))
    restorer.restore(sess, load_model)
    print("Checkpoint loaded:", load_model)

    # -------------------------------------------------------------- #
    #        3)  Prepare mixture audio                               #
    # -------------------------------------------------------------- #
    mix_audio_raw = track.audio          # shape (T, C_in)
    orig_sr       = track.rate
    orig_ch       = mix_audio_raw.shape[1]

    # ---------- Match channel configuration ----------------------- #
    if model_config["mono_downmix"]:
        mix_audio = np.mean(mix_audio_raw, axis=1, keepdims=True)
        assert model_config["num_channels"] == 1
    else:
        desired_ch = model_config["num_channels"]
        mix_audio = mix_audio_raw
        if mix_audio.shape[1] < desired_ch:
            mix_audio = np.tile(mix_audio, [1, desired_ch])[:, :desired_ch]
        elif mix_audio.shape[1] > desired_ch:
            mix_audio = mix_audio[:, :desired_ch]

    # ---------- Resample to training SR --------------------------- #
    mix_audio = Utils.resample(mix_audio, orig_sr, model_config["expected_sr"])

    # -------------------------------------------------------------- #
    #        4)  Sliding-window prediction                           #
    # -------------------------------------------------------------- #
    preds = predict_track(
        model_config,
        sess,
        mix_audio,
        sep_in_shape,
        sep_out_shape,
        separator_sources,
        mix_ph
    )

    # Upsample back to original SR and trim to input length
    preds = {
        k: Utils.resample(v, model_config["expected_sr"], orig_sr)[: mix_audio_raw.shape[0], :]
        for k, v in preds.items()
    }

    # If we down-mixed but original audio has >1 ch, replicate predictions
    if model_config["mono_downmix"] and orig_ch > 1:
        preds = {k: np.tile(v, [1, orig_ch]) for k, v in preds.items()}

    # -------------------------------------------------------------- #
    #        5)  Optional museval evaluation                         #
    # -------------------------------------------------------------- #
    if results_dir is not None:
        museval.eval_mus_track(track, preds, output_dir=results_dir)

    sess.close()
    tf.compat.v1.reset_default_graph()
    return preds


def predict_track(model_config, sess, mix_audio, sep_in_shape,
                  sep_out_shape, separator_sources, mix_ph):
    """
    Sliding-window inference helper.

    mix_audio : np.ndarray [T, C]  (already resampled & channel-matched)
    """
    assert mix_audio.ndim == 2
    T_total, C = mix_audio.shape

    # Pad if shorter than input window
    extra_pad = 0
    if T_total < sep_in_shape[1]:
        extra_pad = sep_in_shape[1] - T_total
        mix_audio = np.pad(mix_audio, [(0, extra_pad), (0, 0)], mode="constant")

    # Allocate output buffers
    source_preds = {
        name: np.zeros_like(mix_audio, dtype=np.float32)
        for name in model_config["source_names"]
    }

    win_in   = sep_in_shape[1]
    win_out  = sep_out_shape[1]
    pad_ctx  = (win_in - win_out) // 2
    mix_pad  = np.pad(mix_audio, [(pad_ctx, pad_ctx), (0, 0)], mode="constant")

    for pos in range(0, T_total, win_out):
        if pos + win_out > T_total:
            pos = T_total - win_out

        snippet = mix_pad[pos: pos + win_in, :]
        snippet = snippet[np.newaxis, ...]                 # [1, win_in, C]
        out_dict = sess.run(separator_sources, feed_dict={mix_ph: snippet})

        for name in model_config["source_names"]:
            source_preds[name][pos: pos + win_out] = out_dict[name][0]

    if extra_pad > 0:
        source_preds = {k: v[:-extra_pad] for k, v in source_preds.items()}

    return source_preds


# ---------------------------------------------------------------------- #
#                     BATCH  PREDICTION  /  EVALUATION                   #
# ---------------------------------------------------------------------- #
def produce_musdb_source_estimates(model_config, load_model, musdb_path,
                                   output_path, subsets=None):
    """
    Predict and save source estimates for every song in MUSDB.

    Audio is saved to `output_path`; museval JSON is stored alongside.
    """
    print("Producing estimates for MUSDB – checkpoint:", load_model)
    mus = musdb.DB(musdb_path)
    if subsets is None:
        subsets = ["test"]

    for subset in subsets:
        tracks = mus.load_mus_tracks(subset)
        for track in tracks:
            print(f"Processing {track.name} ({subset})")
            preds = predict(track, model_config, load_model, results_dir=output_path)

            # Write wav files
            for src, audio in preds.items():
                out_dir = os.path.join(output_path, subset, track.name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{src}.wav")
                sf.write(out_path, audio, track.rate)


def produce_source_estimates(model_config, load_model, input_path,
                             output_path=None):
    """
    Predict sources for *any* mixture wav/flac; save stems next to file.
    """
    print("Predicting for file:", input_path)
    audio, sr = Utils.load(input_path, sr=None, mono=False)

    class TrackLike:
        def __init__(self, audio, rate):
            self.audio = audio
            self.rate  = rate
            self.name  = os.path.splitext(os.path.basename(input_path))[0]

    track = TrackLike(audio, sr)
    preds = predict(track, model_config, load_model)

    out_dir = output_path or os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)

    for src, audio in preds.items():
        sf.write(os.path.join(out_dir, f"{track.name}_{src}.wav"), audio, sr)


# ---------------------------------------------------------------------- #
#                      METRIC  SUMMARY  HELPERS                          #
# ---------------------------------------------------------------------- #
def compute_mean_metrics(json_folder, compute_averages=True, metric="SDR"):
    """
    Compute aggregate museval metrics for a folder of JSON files.
    """
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    inst_list = None
    for path in json_files:
        if path.endswith("test.json"):
            continue
        with open(path, "r") as f:
            js = json.load(f)
        if inst_list is None:
            inst_list = [[] for _ in js["targets"]]
        for i, tgt in enumerate(js["targets"]):
            inst_list[i].extend(float(frm["metrics"][metric]) for frm in tgt["frames"])

    inst_arrs = [np.array(lst) for lst in inst_list]
    if compute_averages:
        return [
            (
                np.nanmedian(a),
                np.nanmedian(np.abs(a - np.nanmedian(a))),
                np.nanmean(a),
                np.nanstd(a),
            )
            for a in inst_arrs
        ]
    return inst_arrs
