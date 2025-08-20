# -*- coding: utf-8 -*-
import glob
import os
import random
from multiprocessing import Process

import numpy as np
from lxml import etree
import librosa
import soundfile
import tensorflow as tf
import musdb

import Utils

# ---------------------------------------------------------------------- #
#                         Helper: snippet extraction                     #
# ---------------------------------------------------------------------- #
def take_random_snippets(sample, keys, input_shape, num_samples):
    """
    Randomly sample `num_samples` snippets of length input_shape[0] frames
    from the given track.
    """
    start_pos = tf.random.uniform(
        [num_samples],
        minval=0,
        maxval=sample["length"] - input_shape[0],
        dtype=tf.int64
    )
    return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)


def take_all_snippets(sample, keys, input_shape, output_shape):
    """
    Deterministically slide a window with stride = output_shape[0]
    over the whole track (for validation & test).
    """
    start_pos = tf.range(
        0,
        sample["length"] - input_shape[0],
        delta=output_shape[0],
        dtype=tf.int64
    )
    num_samples = start_pos.shape[0]
    return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)


def take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples):
    """
    Core slicing helper.
    input_shape = [frames, channels]
    """
    batch = {}
    for key in keys:
        batch[key] = tf.map_fn(
            lambda pos: sample[key][pos:pos + input_shape[0], :],
            start_pos,
            dtype=tf.float32
        )
        batch[key].set_shape([num_samples, input_shape[0], input_shape[1]])
    return tf.data.Dataset.from_tensor_slices(batch)


# ---------------------------------------------------------------------- #
#                         TFRecord helpers                               #
# ---------------------------------------------------------------------- #
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# ---------------------------------------------------------------------- #
#                         Writing TFRecords                              #
# ---------------------------------------------------------------------- #
def write_records(sample_list, model_config, input_shape, output_shape, records_path):
    """
    Multiprocess worker – convert a list of songs to .tfrecords
    """
    # Compute padding (context) frames
    if (input_shape[1] - output_shape[1]) % 2 != 0:
        print("WARNING: Required padding frames are not even!")
    pad_frames = (input_shape[1] - output_shape[1]) // 2

    num_writers = 1
    writers = [
        tf.io.TFRecordWriter(records_path + str(i) + ".tfrecords")
        for i in range(num_writers)
    ]

    all_keys = model_config["source_names"] + ["mix"]
    for sample in sample_list:
        print("Reading song:", sample["mix"])
        try:
            audio_tracks = {}
            for key in all_keys:
                # ------------------------------------------------------ #
                #   Load audio                                           #
                # ------------------------------------------------------ #
                audio, _ = Utils.load(
                    sample[key],
                    sr=model_config["expected_sr"],
                    mono=model_config["mono_downmix"]
                )

                # Make sure we always have shape (frames, channels)
                if audio.ndim == 1:
                    audio = np.expand_dims(audio, axis=1)

                ### NEW / CHANGED: enforce num_channels ---------------- #
                desired_ch = model_config["num_channels"]
                if audio.shape[1] != desired_ch:
                    if audio.shape[1] == 1 and desired_ch > 1:
                        # Replicate mono channel to stereo / multichannel
                        audio = np.tile(audio, [1, desired_ch])
                        print(f"Duplicated mono → {desired_ch}ch for {key}")
                    elif audio.shape[1] > desired_ch:
                        audio = audio[:, :desired_ch]
                        print(f"Clipped to first {desired_ch}ch for {key}")
                    else:
                        raise ValueError(
                            f"Audio ({sample[key]}) has {audio.shape[1]} channels, "
                            f"but num_channels={desired_ch}"
                        )
                ### ---------------------------------------------------- #

                audio_tracks[key] = audio
        except Exception as e:
            print("ERROR loading file – skipped.\n", e)
            continue

        # ---------------------------------------------------------- #
        #   Pad context frames                                       #
        # ---------------------------------------------------------- #
        audio_tracks = {
            k: np.pad(
                v,
                [(pad_frames, pad_frames), (0, 0)],
                mode="constant",
                constant_values=0.0
            )
            for k, v in audio_tracks.items()
        }

        # Sanity check: same length / channels for all stems
        length = audio_tracks["mix"].shape[0]
        channels = audio_tracks["mix"].shape[1]
        for audio in audio_tracks.values():
            assert audio.shape[0] == length
            assert audio.shape[1] == channels

        # Serialize
        feature = {k: _floats_feature(v) for k, v in audio_tracks.items()}
        feature["length"] = _int64_feature(length)
        feature["channels"] = _int64_feature(channels)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writers[random.randrange(num_writers)].write(example.SerializeToString())

    for w in writers:
        w.close()


# ---------------------------------------------------------------------- #
#                         Parsing TFRecords                              #
# ---------------------------------------------------------------------- #
def parse_record(example_proto, source_names, shape):
    """
    Deserialize TFRecord → dict with tensors
    """
    all_names = source_names + ["mix"]
    features = {
        key: tf.io.FixedLenSequenceFeature([], allow_missing=True, dtype=tf.float32)
        for key in all_names
    }
    features["length"] = tf.io.FixedLenFeature([], tf.int64)
    features["channels"] = tf.io.FixedLenFeature([], tf.int64)

    parsed = tf.io.parse_single_example(example_proto, features)

    length = tf.cast(parsed["length"], tf.int64)
    channels = tf.constant(shape[-1], tf.int64)   # EXPECTED channels

    sample = {}
    for key in all_names:
        sample[key] = tf.reshape(parsed[key], tf.stack([length, channels]))
    sample["length"] = length
    sample["channels"] = channels
    return sample


# ---------------------------------------------------------------------- #
#                         Dataset generators                             #
# ---------------------------------------------------------------------- #
def getCustomDatasetSamples(data_path):
    """
    Expect each song folder to contain mixture.wav / vocals.wav / accompaniment.wav
    """
    songs = [
        os.path.join(data_path, d)
        for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ]
    samples = []
    for song in songs:
        mix_path = os.path.join(song, "mixture.wav")
        vocals_path = os.path.join(song, "vocals.wav")
        acc_path = os.path.join(song, "accompaniment.wav")
        if all(map(os.path.exists, [mix_path, vocals_path, acc_path])):
            samples.append({
                "mix": mix_path,
                "vocals": vocals_path,
                "accompaniment": acc_path
            })
        else:
            print("Skipping (missing stems):", song)
    return samples


def get_dataset(model_config, input_shape, output_shape, partition):
    """
    Main entry – returns a tf.data.Dataset yielding dicts with:
        {"mix": [B, T, C], source_name: [B, T, C], ...}
    """
    ### NEW / CHANGED: dataset_name includes num_channels ------------- ###
    dataset_name = (
        f"task_{model_config['task']}_sr_{model_config['expected_sr']}_"
        f"mono_{model_config['mono_downmix']}_ch_{model_config['num_channels']}"
    )
    ### ---------------------------------------------------------------- ###
    main_folder = os.path.join(model_config["data_path"], dataset_name)

    # -------------------------------------------------------------- #
    #   Prepare .tfrecords if not present                            #
    # -------------------------------------------------------------- #
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
        if model_config["task"] == "custom":
            print("Preparing custom dataset... (may take a while)")
            samples = getCustomDatasetSamples(model_config["data_path"])
            random.shuffle(samples)
            n = len(samples)
            dataset = {
                "train": samples[: int(0.8 * n)],
                "valid": samples[int(0.8 * n): int(0.9 * n)],
                "test":  samples[int(0.9 * n):]
            }
        else:
            print("Preparing MUSDB dataset... (may take a while)")
            dsd_train, dsd_test = getMUSDB(model_config["musdb_path"])
            val_idx = np.random.choice(len(dsd_train), size=25, replace=False)
            train_idx = [i for i in range(len(dsd_train)) if i not in val_idx]
            dataset = {
                "train": [dsd_train[i] for i in train_idx],
                "valid": [dsd_train[i] for i in val_idx],
                "test":  dsd_test
            }
            if model_config["task"] == "voice":
                print("Adding CCMixter vocals dataset...")
                dataset["train"].extend(getCCMixter("CCMixter.xml"))

        # ------------- Write TFRecords (multi-processing) ------------- #
        num_cores = 8
        for curr in ["train", "valid", "test"]:
            print(f"Writing {curr} partition – {len(dataset[curr])} songs")
            random.shuffle(dataset[curr])
            partition_folder = os.path.join(main_folder, curr)
            os.makedirs(partition_folder, exist_ok=True)

            per_core = int(np.ceil(len(dataset[curr]) / float(num_cores)))
            procs = []
            for core in range(num_cores):
                filename_prefix = os.path.join(partition_folder, f"{core}_")
                subset = dataset[curr][core * per_core: (core + 1) * per_core]
                p = Process(
                    target=write_records,
                    args=(subset, model_config, input_shape, output_shape, filename_prefix)
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

    print("Dataset ready – loading TFRecords")
    # -------------------------------------------------------------- #
    #   Load TFRecords                                               #
    # -------------------------------------------------------------- #
    dataset_folder = os.path.join(main_folder, partition)
    record_files = glob.glob(os.path.join(dataset_folder, "*.tfrecords"))
    random.shuffle(record_files)

    dataset = tf.data.TFRecordDataset(record_files)
    dataset = dataset.map(
        lambda x: parse_record(x, model_config["source_names"], input_shape[1:]),
        num_parallel_calls=model_config["num_workers"]
    )
    dataset = dataset.prefetch(10)

    # -------------------------------------------------------------- #
    #   Convert tracks → snippets                                    #
    # -------------------------------------------------------------- #
    if partition == "train":
        dataset = dataset.flat_map(
            lambda x: take_random_snippets(
                x,
                model_config["source_names"] + ["mix"],
                input_shape[1:],
                model_config["num_snippets_per_track"]
            )
        )
    else:
        dataset = dataset.flat_map(
            lambda x: take_all_snippets(
                x,
                model_config["source_names"] + ["mix"],
                input_shape[1:],
                output_shape[1:]
            )
        )
    dataset = dataset.prefetch(100)

    # Data augmentation
    if partition == "train" and model_config["augmentation"]:
        dataset = dataset.map(
            Utils.random_amplify,
            num_parallel_calls=model_config["num_workers"]
        ).prefetch(100)

    # Center-crop (remove context frames)
    dataset = dataset.map(
        lambda x: Utils.crop_sample(
            x,
            (input_shape[1] - output_shape[1]) // 2
        )
    ).prefetch(100)

    # Repeat / shuffle / batch
    if partition == "train":
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=model_config["cache_size"])

    dataset = dataset.batch(model_config["batch_size"], drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


# ---------------------------------------------------------------------- #
#                        External dataset helpers                        #
# ---------------------------------------------------------------------- #
def getMUSDB(database_path):
    """Download / convert MUSDB to wav stems on first run."""
    mus = musdb.DB(database_path, is_wav=False)
    subsets = []
    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = []
        for track in tracks:
            track_path = track.path[:-4]
            mix_path = track_path + "_mix.wav"
            acc_path = track_path + "_accompaniment.wav"

            if os.path.exists(mix_path):
                print("WARNING: Skipping existing track", mix_path)
                paths = {"mix": mix_path, "accompaniment": acc_path}
                paths.update({k: track_path + "_" + k + ".wav"
                              for k in ["bass", "drums", "other", "vocals"]})
                samples.append(paths)
                continue

            rate = track.rate
            paths = {}
            stem_audio = {}
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + f"_{stem}.wav"
                audio = track.targets[stem].audio
                soundfile.write(path, audio, rate, "PCM_16")
                stem_audio[stem] = audio
                paths[stem] = path

            acc_audio = np.clip(
                sum(stem_audio[k] for k in stem_audio if k != "vocals"),
                -1.0, 1.0
            )
            soundfile.write(acc_path, acc_audio, rate, "PCM_16")
            paths["accompaniment"] = acc_path

            mix_audio = track.audio
            soundfile.write(mix_path, mix_audio, rate, "PCM_16")
            paths["mix"] = mix_path

            diff = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Max / mean additivity error:",
                  np.max(diff), "/", np.mean(diff))
            samples.append(paths)
        subsets.append(samples)
    return subsets  # [train, test]


def getCCMixter(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    def get_path(node):
        return db_path + os.sep + node.xpath("./relativeFilepath")[0].text

    samples = []
    for track in tracks:
        voice = get_path(
            track.xpath(".//instrument[instrumentName='Voice']")[0])
        mix = get_path(
            track.xpath(".//instrument[instrumentName='Mix']")[0])
        acc = get_path(
            track.xpath(".//instrument[instrumentName='Instrumental']")[0])
        samples.append({"mix": mix, "accompaniment": acc, "vocals": voice})
    return samples