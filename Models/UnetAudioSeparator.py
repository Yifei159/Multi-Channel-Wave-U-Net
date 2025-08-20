import tensorflow as tf
import numpy as np

import Models.InterpolationLayer
import Models.OutputLayer
import Utils
from Utils import LeakyReLU


class UnetAudioSeparator:
    """
    Time-domain Wave-U-Net separator.

    The network uses *valid* convolutions when `context=True`, in which case
    the input must be larger than the output and `get_padding` returns the
    correct padded shapes.  Otherwise it defaults to *same* padding.
    """

    # ------------------------------------------------------------------ #
    #                           INITIALISATION                           #
    # ------------------------------------------------------------------ #
    def __init__(self, model_config):
        # Architecture
        self.num_layers            = model_config["num_layers"]
        self.num_initial_filters   = model_config["num_initial_filters"]
        self.filter_size           = model_config["filter_size"]
        self.merge_filter_size     = model_config["merge_filter_size"]
        self.input_filter_size     = model_config["input_filter_size"]
        self.output_filter_size    = model_config["output_filter_size"]
        self.upsampling            = model_config["upsampling"]

        # Training / inference
        self.output_type           = model_config["output_type"]
        self.context               = model_config["context"]
        self.padding               = "valid" if self.context else "same"
        self.output_activation     = model_config["output_activation"]

        # Sources / channels
        self.source_names          = model_config["source_names"]
        ### NEW / CHANGED --------------------------------------------- ###
        self.num_channels          = model_config["num_channels"]
        ### ----------------------------------------------------------- ###

    # ------------------------------------------------------------------ #
    #                       SHAPE / PADDING LOGIC                        #
    # ------------------------------------------------------------------ #
    def get_padding(self, shape):
        """
        Compute required input/output shapes given a desired output
        (batch, frames, channels) when `context=True`. For `context=False`
        it's just the same shape.

        Parameters
        ----------
        shape : list / np.ndarray
            Desired **output** shape [batch, frames, channels]

        Returns
        -------
        input_shape, output_shape : np.ndarray
            Corresponding input & output shapes.
        """
        if self.context:
            # Reverse compute minimal resolution
            rem = float(shape[1])                           # frames
            rem = rem - self.output_filter_size + 1

            for _ in range(self.num_layers):
                rem = rem + self.merge_filter_size - 1
                rem = (rem + 1.) / 2.

            x = np.asarray(np.ceil(rem), dtype=np.int64)
            assert x >= 2, "Too few frames – increase input length"

            output_shape = x
            input_shape  = x + self.filter_size - 1

            for i in range(self.num_layers):
                output_shape = 2 * output_shape - 1
                output_shape = output_shape - self.merge_filter_size + 1

                input_shape  = 2 * input_shape - 1
                if i < self.num_layers - 1:
                    input_shape += self.filter_size - 1
                else:
                    input_shape += self.input_filter_size - 1

            output_shape = output_shape - self.output_filter_size + 1

            input_shape  = np.concatenate([[shape[0]], [input_shape],  [self.num_channels]])
            output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])
            return input_shape, output_shape

        # Same-padding mode
        return [shape[0], shape[1], self.num_channels], [shape[0], shape[1], self.num_channels]

    # ------------------------------------------------------------------ #
    #                       FORWARD / GRAPH BUILD                        #
    # ------------------------------------------------------------------ #
    def get_output(self, inputs, training, return_spectrogram=False, reuse=True):
        """
        Build the computation graph and return source estimates.

        Parameters
        ----------
        inputs : tf.Tensor
            Mixture audio, shape [B, T, C]
        training : bool
            If True, apply training-specific layers (e.g., dropout)
        return_spectrogram : bool
            (Kept for compatibility; ignored in this implementation)
        reuse : bool
            Reuse variables (for weight sharing)

        Returns
        -------
        dict
            {source_name: tf.Tensor [B, T, C], ...}
        """
        with tf.compat.v1.variable_scope("separator", reuse=reuse):
            enc_outputs = []
            x = inputs                                           # [B, T, C]

            # ------------------------- ENCODER ------------------------ #
            for i in range(self.num_layers):
                x = tf.compat.v1.layers.conv1d(
                    x,
                    filters=self.num_initial_filters + self.num_initial_filters * i,
                    kernel_size=self.filter_size,
                    strides=1,
                    activation=LeakyReLU,
                    padding=self.padding
                )
                enc_outputs.append(x)
                x = x[:, ::2, :]                                 # Decimate (stride-2)

            # Bottleneck
            x = tf.compat.v1.layers.conv1d(
                x,
                filters=self.num_initial_filters + self.num_initial_filters * self.num_layers,
                kernel_size=self.filter_size,
                activation=LeakyReLU,
                padding=self.padding
            )

            # ------------------------- DECODER ------------------------ #
            for i in range(self.num_layers):
                # -> [B, 1, T', C] so we can use 2-D resize as upsampling
                x = tf.expand_dims(x, 1)

                if self.upsampling == "learned":
                    x = Models.InterpolationLayer.learned_interpolation_layer(x, self.padding, i)
                else:
                    cur_w = x.get_shape().as_list()[2]
                    if self.context:
                        new_w = cur_w * 2 - 1
                        x = tf.image.resize(
                            x, [1, new_w],
                            method=tf.image.ResizeMethod.BILINEAR,
                            align_corners=True
                        )
                    else:
                        new_w = cur_w * 2
                        x = tf.image.resize(
                            x, [1, new_w],
                            method=tf.image.ResizeMethod.BILINEAR
                        )

                x = tf.squeeze(x, 1)

                # Skip connection
                enc = enc_outputs[-i - 1]
                assert (
                    enc.get_shape().as_list()[1] == x.get_shape().as_list()[1]
                    or self.context
                )
                x = Utils.crop_and_concat(enc, x, match_feature_dim=False)

                # Merge conv
                x = tf.compat.v1.layers.conv1d(
                    x,
                    filters=self.num_initial_filters + self.num_initial_filters * (self.num_layers - i - 1),
                    kernel_size=self.merge_filter_size,
                    activation=LeakyReLU,
                    padding=self.padding
                )

            # Final skip connection with original input
            x = Utils.crop_and_concat(inputs, x, match_feature_dim=False)

            # --------------------- OUTPUT LAYER ----------------------- #
            if self.output_activation == "tanh":
                activation_fn = tf.tanh
            elif self.output_activation == "linear":
                activation_fn = lambda z: Utils.AudioClip(z, training)
            else:
                raise NotImplementedError("Unknown output_activation")

            if self.output_type == "direct":
                return Models.OutputLayer.independent_outputs(
                    x,
                    self.source_names,
                    self.num_channels,
                    self.output_filter_size,
                    self.padding,
                    activation_fn
                )

            if self.output_type == "difference":
                cropped_in = Utils.crop(
                    inputs,
                    x.get_shape().as_list(),
                    match_feature_dim=False
                )
                return Models.OutputLayer.difference_output(
                    cropped_in,
                    x,
                    self.source_names,
                    self.num_channels,
                    self.output_filter_size,
                    self.padding,
                    activation_fn,
                    training
                )

            raise NotImplementedError("Unknown output_type")
