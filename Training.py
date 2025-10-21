import os
import functools
from sacred import Experiment
from Config import config_ingredient
import tensorflow as tf
import numpy as np
import Datasets
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import Test
import Evaluate

np.set_printoptions(precision=10, suppress=True)

tf.compat.v1.disable_eager_execution()

ex = Experiment('Waveunet Training', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1337

@config_ingredient.capture
def train(model_config, experiment_id, load_model=None):

    num_channels = int(model_config.get("num_channels", 1))
    if num_channels < 1:
        raise ValueError("model_config['num_channels'] must ≥ 1")

    disc_input_shape = [
        model_config["batch_size"],
        model_config["num_frames"],
        num_channels
    ]

    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError("Unknown network type: " + model_config["network"])
        

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    dataset = Datasets.get_dataset(
        model_config,
        sep_input_shape,
        sep_output_shape,
        partition="train"
    )
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch = iterator.get_next()

    print("Training...")

    separator_sources = separator_func(
        batch["mix"],
        training=True,
        return_spectrogram=(not model_config["raw_audio_loss"]),
        reuse=False
    )

    separator_loss = 0.0
    for key in model_config["source_names"]:
        real_source = batch[key]           # [B, T, C]
        sep_source  = separator_sources[key]

        if (model_config["network"] == "unet_spectrogram" and
                not model_config["raw_audio_loss"]):
            window = functools.partial(tf.signal.hann_window, periodic=True)

            # [B, T, C] -> [B*C, T]
            real_reshaped = tf.reshape(
                real_source,
                [-1, tf.shape(real_source)[1]]
            )
            stfts = tf.signal.stft(
                real_reshaped,
                frame_length=1024,
                frame_step=768,
                fft_length=1024,
                window_fn=window
            )                  
            real_mag = tf.abs(stfts)

            if len(sep_source.shape) == 4: 
                real_mag = tf.reshape(
                    real_mag,
                    [tf.shape(real_source)[0], num_channels,
                     tf.shape(real_mag)[1], tf.shape(real_mag)[2]]
                )

            separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source))
        else:
            separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))

    separator_loss = separator_loss / float(model_config["num_sources"])

    global_step = tf.compat.v1.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0),
        trainable=False, dtype=tf.int64
    )
    increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)

    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables: " + str(len(tf.compat.v1.global_variables())))

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.compat.v1.variable_scope("separator_solver"):
            separator_solver = tf.compat.v1.train.AdamOptimizer(
                learning_rate=model_config["init_sup_sep_lr"]
            ).minimize(separator_loss, var_list=separator_vars)

    tf.compat.v1.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.compat.v1.summary.merge_all(key='sup')


    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config_proto)
    sess.run(tf.compat.v1.global_variables_initializer())

    writer = tf.compat.v1.summary.FileWriter(
        os.path.join(model_config["log_dir"], str(experiment_id)),
        graph=sess.graph
    )

    if load_model is not None:
        restorer = tf.compat.v1.train.Saver(
            tf.compat.v1.global_variables(),
            write_version=tf.compat.v1.train.SaverDef.V2
        )
        print("Num of variables: " + str(len(tf.compat.v1.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from ' + load_model)

    saver = tf.compat.v1.train.Saver(
        tf.compat.v1.global_variables(),
        write_version=tf.compat.v1.train.SaverDef.V2,
        max_to_keep=3
    )

    _global_step = sess.run(global_step)
    _init_step = _global_step

    for _ in range(model_config["epoch_it"]):
        _, _sup_summaries = sess.run([separator_solver, sup_summaries])
        writer.add_summary(_sup_summaries, global_step=_global_step)
        _global_step = sess.run(increment_global_step)

    print("Finished epoch!")
    save_path = saver.save(
        sess,
        os.path.join(model_config["model_base_dir"], str(experiment_id), str(experiment_id)),
        global_step=int(_global_step)
    )

    writer.flush()
    writer.close()
    sess.close()
    tf.compat.v1.reset_default_graph()

    return save_path

@config_ingredient.capture
def optimise(model_config, experiment_id):

    local_config = dict(model_config)
    epoch = 0
    best_loss = 1e10
    model_path = None
    best_model_path = None

    for i in range(2):
        worse_epochs = 0
        if i == 1:
            print("=" * 50)
            print("  Finished first round, entering fine-tuning")
            print("=" * 50)
            local_config["batch_size"] *= 2
            local_config["init_sup_sep_lr"] = 1e-5

        while worse_epochs < local_config["worse_epochs"]:
            print("-" * 50)
            print("EPOCH:", epoch)
            print("-" * 50)

            model_path = train(
                model_config=local_config,
                load_model=model_path,
                experiment_id=experiment_id
            )

            curr_loss = Test.test(
                model_config=local_config,
                model_folder=str(experiment_id),
                partition="valid",
                load_model=model_path
            )
            epoch += 1

            if curr_loss < best_loss:
                worse_epochs = 0
                print(f"Validation improved {best_loss:.6f} → {curr_loss:.6f}")
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print(f"Validation worsened to {curr_loss:.6f}  (no.{worse_epochs})")

    print("=" * 50)
    print("TRAINING FINISHED – TESTING BEST MODEL", best_model_path)
    print("=" * 50)
    test_loss = Test.test(
        model_config=local_config,
        model_folder=str(experiment_id),
        partition="test",
        load_model=best_model_path
    )
    return best_model_path, test_loss

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]

    print("SCRIPT START")

    for d in [model_config["model_base_dir"], model_config["log_dir"]]:
        os.makedirs(d, exist_ok=True)

    sup_model_path, sup_loss = optimise(experiment_id=cfg["experiment_id"])
    print(f"Supervised training finished! Model: {sup_model_path}  |  Loss: {sup_loss:.6f}")

    if model_config.get("task", "") != "custom":
        Evaluate.produce_musdb_source_estimates(
            model_config,
            sup_model_path,
            model_config["musdb_path"],
            model_config["estimates_path"]
        )
    else:
        print("Custom training finished – implement your own evaluation if needed.")
