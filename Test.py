import os, functools, numpy as np, tensorflow as tf
tf.compat.v1.disable_eager_execution()

import Datasets
import Models.UnetSpectrogramSeparator, Models.UnetAudioSeparator

np.set_printoptions(precision=10, suppress=True)


def test(model_config, partition, model_folder, load_model):
    """Evaluate a trained separator on `partition`; return mean loss."""
    # ------------- 1) graph --------------
    num_channels = model_config["num_channels"]
    disc_input_shape = [model_config["batch_size"],
                        model_config["num_frames"],
                        num_channels]

    if model_config["network"] == "unet":
        sep_cls = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        sep_cls = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError

    sep_in_shape, sep_out_shape = sep_cls.get_padding(np.array(disc_input_shape))
    sep_func = sep_cls.get_output

    dataset  = Datasets.get_dataset(model_config, sep_in_shape, sep_out_shape, partition=partition)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch    = iterator.get_next()

    # ------------- 2) forward -------------
    sep_out = sep_func(batch["mix"],
                       training=False,
                       return_spectrogram=not model_config["raw_audio_loss"],
                       reuse=False)

    # ------------- 3) loss --------------
    loss = 0.0
    for key in model_config["source_names"]:
        real = batch[key]          # [B,T,C]
        est  = sep_out[key]

        if model_config["network"] == "unet_spectrogram" and not model_config["raw_audio_loss"]:
            window = functools.partial(tf.signal.hann_window, periodic=True)
            real_flat = tf.reshape(tf.transpose(real, [0, 2, 1]), [-1, tf.shape(real)[1]])
            stfts     = tf.signal.stft(real_flat, 1024, 768, 1024, window_fn=window)
            real_mag  = tf.abs(stfts)                          # [B*C,F,Tʹ]

            if len(est.shape) == 4:    # est=[B,C,F,Tʹ]
                real_mag = tf.reshape(real_mag,
                                      [tf.shape(real)[0], num_channels,
                                       tf.shape(real_mag)[1], tf.shape(real_mag)[2]])
            elif len(est.shape) == 3:  # est=[B,F,Tʹ]
                real_mag = tf.reduce_mean(tf.reshape(real_mag,
                                                    [tf.shape(real)[0], num_channels,
                                                     tf.shape(real_mag)[1], tf.shape(real_mag)[2]]),
                                           axis=1)              # -> [B,F,Tʹ]

            loss += tf.reduce_mean(tf.abs(real_mag - est))
        else:
            loss += tf.reduce_mean(tf.square(real - est))

    loss /= float(model_config["num_sources"])

    # ------------- 4) session -------------
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
    sess        = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    tf.compat.v1.train.Saver(tf.compat.v1.global_variables()).restore(sess, load_model)
    print("Checkpoint restored – start evaluating")

    # ------------- 5) loop --------------
    total, n = 0.0, 0
    while True:
        try:
            lv = sess.run(loss)
            n += 1
            total += (lv - total) / n        
        except tf.errors.OutOfRangeError:
            break

    # TensorBoard
    writer = tf.compat.v1.summary.FileWriter(os.path.join(model_config["log_dir"], model_folder))
    writer.add_summary(tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(tag="test_loss", simple_value=total)
    ]), global_step=sess.run(global_step))
    writer.flush(); writer.close()

    print(f"Finished testing – Mean loss: {total:.6f}")
    sess.close(); tf.compat.v1.reset_default_graph()
    return total
