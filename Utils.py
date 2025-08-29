import numpy as np
import tensorflow as tf
import librosa


def getTrainableVariables(tag=""):
    """Return trainable variables containing `tag` in their name."""
    return [v for v in tf.compat.v1.trainable_variables() if tag in v.name]


def getNumParams(tensors):
    """Count total number of scalar parameters across given tensors."""
    return int(np.sum([np.prod(t.get_shape().as_list()) for t in tensors]))


# ---------------------------------------------------------------------- #
#                  FEATURE MAP CROP & CONCAT (U-NET)                     #
# ---------------------------------------------------------------------- #
def crop_and_concat(x1, x2, match_feature_dim=True):
    if x2 is None:
        return x1
    x1 = crop(x1, x2.get_shape().as_list(), match_feature_dim)
    return tf.concat([x1, x2], axis=2)


# ---------------------------------------------------------------------- #
#                           DATA  AUGMENTATION                           #
# ---------------------------------------------------------------------- #
def random_amplify(sample):

    for key, val in sample.items():
        if key != "mix":
            sample[key] = tf.random.uniform([], 0.7, 1.0) * val
    sample["mix"] = tf.add_n([v for k, v in sample.items() if k != "mix"])
    return sample


def crop_sample(sample, crop_frames):
    """
    Remove `crop_frames` context frames from *sources* (not mix).

    Parameters
    ----------
    sample : dict of tensors
        {"mix": [T, C], source1: [T, C], ...}
    crop_frames : int
        Number of frames to border-crop.
    """
    if crop_frames <= 0:
        return sample
    for k, v in sample.items():
        if k != "mix":
            sample[k] = v[crop_frames:-crop_frames, :]
    return sample


# ---------------------------------------------------------------------- #
#                           AUDIO  HELPERS                               #
# ---------------------------------------------------------------------- #
def resample(audio, orig_sr, new_sr):
    """
    Librosa wrapper that keeps channel dimension last.
    `audio` must be (T, C) numpy array.
    """
    return librosa.resample(y=audio.T, orig_sr=orig_sr, target_sr=new_sr).T


def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    """
    Load an audio file and return a **T x C** matrix (always 2-D).

    • When `mono=True`, librosa down-mixes and we expand a channel dim.  
    • When `mono=False`, shape becomes (C, T) → transpose to (T, C).  
    """
    y, orig_sr = librosa.load(
        path,
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration,
        dtype=dtype,
    )
    if y.ndim == 1:                         # mono track
        y = y[np.newaxis, :]                # (1, T)
    return y.T.astype(dtype), orig_sr       # (T, C), sr


# ---------------------------------------------------------------------- #
#                           ACTIVATIONS                                  #
# ---------------------------------------------------------------------- #
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def AudioClip(x, training):
    """
    During inference, hard-clip to [-1, 1] to avoid exploding samples.
    """
    return x if training else tf.clip_by_value(x, -1.0, 1.0)


# ---------------------------------------------------------------------- #
#                              CROPPING                                  #
# ---------------------------------------------------------------------- #
def crop(tensor, target_shape, match_feature_dim=True):
    """
    Centre-crop a 3-D tensor [B, W, F] along the width axis until it matches
    `target_shape`.

    Only width (axis=1) may differ. If `match_feature_dim=True`, feature
    dim (axis=2) must already be equal.
    """
    shape = np.array(tensor.get_shape().as_list())
    diff  = shape - np.array(target_shape)

    # batch size must agree
    assert diff[0] == 0
    # feature dim mismatch allowed only when explicitly disabled
    assert diff[2] == 0 or not match_feature_dim

    if diff[1] < 0:
        raise ValueError("Target wider than tensor – cannot crop.")
    if diff[1] == 0:
        return tensor
    if diff[1] % 2 != 0:
        print("WARNING: Cropping with uneven number of frames")

    crop_left  = diff[1] // 2
    crop_right = diff[1] - crop_left
    return tensor[:, crop_left:-crop_right, :]


# ---------------------------------------------------------------------- #
#                     SPECTROGRAM  PAD / RECONSTRUCTION                  #
# ---------------------------------------------------------------------- #
def pad_freqs(tensor, target_shape):
    """
    Zero-pad (or no-op) along the frequency axis to match `target_shape`.
    Safely handles both numpy & tf.Tensor, 2-D or 4-D.
    """
    target_freqs = target_shape[1] if len(target_shape) == 4 else target_shape[0]

    if isinstance(tensor, tf.Tensor):
        in_shape = tensor.get_shape().as_list()
    else:
        in_shape = tensor.shape

    input_freqs = in_shape[1] if len(in_shape) == 4 else in_shape[0]
    diff = target_freqs - input_freqs
    if diff == 0:
        return tensor

    pad_top = diff // 2
    pad_bottom = diff - pad_top
    pad_spec = (
        [(0, 0), (pad_top, pad_bottom), (0, 0), (0, 0)]  # 4-D
        if len(target_shape) == 4
        else [(pad_top, pad_bottom), (0, 0)]              # 2-D
    )

    if isinstance(tensor, tf.Tensor):
        return tf.pad(tensor, pad_spec, mode="CONSTANT", constant_values=0.0)
    return np.pad(tensor, pad_spec, mode="constant", constant_values=0.0)


# ---------------------------------------------------------------------- #
#                    SPECTROGRAM → AUDIO (Griffin-Lim)                   #
# ---------------------------------------------------------------------- #
def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize,
                           phaseIterations=10, phase=None, length=None):
    """
    Griffin-Lim phase reconstruction helper.
    **Not used in the main training loop**, but handy for ad-hoc eval.
    """
    if phase is not None:
        if phaseIterations > 0:
            return reconPhase(magnitude, fftWindowSize, hopSize,
                               phaseIterations, phase, length)
        stftMatrix = magnitude * np.exp(phase * 1j)
        return librosa.istft(stftMatrix, hop_length=hopSize, length=length)
    return reconPhase(magnitude, fftWindowSize, hopSize,
                      phaseIterations, initPhase=None, length=length)


def reconPhase(magnitude, fftWindowSize, hopSize,
               phaseIterations=10, initPhase=None, length=None):
    """Iterative Griffin-Lim algorithm."""
    for i in range(phaseIterations):
        if i == 0:
            reconstruction = (
                np.exp(initPhase * 1j) if initPhase is not None
                else np.exp(1j * 2 * np.pi * np.random.rand(*magnitude.shape))
            )
        else:
            reconstruction = librosa.stft(audio, n_fft=fftWindowSize,
                                          hop_length=hopSize)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        audio = librosa.istft(
            spectrum, hop_length=hopSize,
            length=length if i == phaseIterations - 1 else None
        )
    return audio
