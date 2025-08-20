# -*- coding: utf-8 -*-
"""
Multi-channel Wave-U-Net training script (time-domain)
----------------------------------------------------
兼容单通道与多通道。请确保在 config 中添加
    "num_channels": 2            # 例如立体声
并让 Datasets、模型等返回 [B, T, C] 形状的数据。
"""

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

# 关闭 Eager Execution，以便继续使用 v1 图模式
tf.compat.v1.disable_eager_execution()

ex = Experiment('Waveunet Training', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1337
    # 可在 config 文件中覆盖

# ------------------------------------------------------------------------- #
#                              TRAINING LOOP                                #
# ------------------------------------------------------------------------- #
@config_ingredient.capture
def train(model_config, experiment_id, load_model=None):
    """
    训练一次（一个 epoch_it），返回保存模型的路径
    """
    ### NEW / CHANGED ----------------------------------------------------- ###
    # ----- 1. 支持多通道 -------------------------------------------------- #
    num_channels = int(model_config.get("num_channels", 1))
    if num_channels < 1:
        raise ValueError("model_config['num_channels'] 必须 ≥ 1")

    # 输入张量形状： [batch, frames, channels]
    disc_input_shape = [
        model_config["batch_size"],
        model_config["num_frames"],
        num_channels
    ]
    ### ------------------------------------------------------------------- ###

    # 构建分离器（Wave-U-Net 时域或频域）
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError("Unknown network type: " + model_config["network"])
        

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # ------------------------------------------------------------------ #
    #                            数据集                                   #
    # ------------------------------------------------------------------ #
    dataset = Datasets.get_dataset(
        model_config,
        sep_input_shape,
        sep_output_shape,
        partition="train"
    )
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch = iterator.get_next()            # dict，包含 "mix" 与各 source 名

    print("Training...")

    # ------------------------------------------------------------------ #
    #                        前向传播 + 损失                              #
    # ------------------------------------------------------------------ #
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

        # 频谱域损失（如果启用）
        if (model_config["network"] == "unet_spectrogram" and
                not model_config["raw_audio_loss"]):
            # ------ 计算每个通道的 STFT 并求幅度损失 ------------------ #
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
            )                              # [B*C, F,  T’]
            real_mag = tf.abs(stfts)

            # sep_source 需保证同样 [B, C, F, T’] 或 [B*C, F, T’]（由模型决定）
            # 若模型输出 [B, C, F, T’]，则 reshape real_mag 以对齐：
            if len(sep_source.shape) == 4:     # [B, C, F, T’]
                real_mag = tf.reshape(
                    real_mag,
                    [tf.shape(real_source)[0], num_channels,
                     tf.shape(real_mag)[1], tf.shape(real_mag)[2]]
                )

            separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source))
        else:
            # 时域 L2/L1
            separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))

    separator_loss = separator_loss / float(model_config["num_sources"])

    # ------------------------------------------------------------------ #
    #                       优化器 & 变量收集                              #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #                       Session & 日志                                #
    # ------------------------------------------------------------------ #
    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config_proto)
    sess.run(tf.compat.v1.global_variables_initializer())

    writer = tf.compat.v1.summary.FileWriter(
        os.path.join(model_config["log_dir"], str(experiment_id)),
        graph=sess.graph
    )

    # ------------------------------------------------------------------ #
    #                    载入已有模型（若指定）                            #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #                          训练循环                                   #
    # ------------------------------------------------------------------ #
    _global_step = sess.run(global_step)
    _init_step = _global_step

    for _ in range(model_config["epoch_it"]):
        _, _sup_summaries = sess.run([separator_solver, sup_summaries])
        writer.add_summary(_sup_summaries, global_step=_global_step)
        _global_step = sess.run(increment_global_step)

    # ------------------------------------------------------------------ #
    #                          保存模型                                   #
    # ------------------------------------------------------------------ #
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

# ------------------------------------------------------------------------- #
#                          HYPER-PARAMETER OPTIMISER                        #
# ------------------------------------------------------------------------- #
@config_ingredient.capture
def optimise(model_config, experiment_id):
    """
    两阶段训练（预训练 + 微调），返回最佳模型路径和测试损失
    """
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

# ------------------------------------------------------------------------- #
#                               ENTRY POINT                                 #
# ------------------------------------------------------------------------- #
@ex.automain
def run(cfg):
    """
    Sacred entry point
    """
    model_config = cfg["model_config"]

    print("SCRIPT START")

    # 创建目录
    for d in [model_config["model_base_dir"], model_config["log_dir"]]:
        os.makedirs(d, exist_ok=True)

    sup_model_path, sup_loss = optimise(experiment_id=cfg["experiment_id"])
    print(f"Supervised training finished! Model: {sup_model_path}  |  Loss: {sup_loss:.6f}")

    # MUSDB 或自定义评估
    if model_config.get("task", "") != "custom":
        Evaluate.produce_musdb_source_estimates(
            model_config,
            sup_model_path,
            model_config["musdb_path"],
            model_config["estimates_path"]
        )
    else:
        print("Custom dataset training finished – implement your own evaluation if needed.")