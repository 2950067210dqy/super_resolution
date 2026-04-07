from loguru import logger

import os
import time
from datetime import datetime

from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models import vgg19
from tqdm import tqdm

import wandb
from study.SRGAN.data_load import get_class_names, load_data, save_loaders_paths

from study.SRGAN.model.PIV_esrgan_RAFT.Module.model import Generator, Discriminator
from study.SRGAN.model.PIV_esrgan_RAFT.evaluate import evaluate, evaluate_all
from study.SRGAN.model.PIV_esrgan_RAFT.global_class import global_data
from study.SRGAN.model.PIV_esrgan_RAFT.train import image_pair_train, flow_train
from study.SRGAN.util.CSV_operator import CsvTable
from study.SRGAN.util.accumulator import Accumulator
from study.SRGAN.util.animator import Animator

def select_single_class(available_class_names, preset_name=None):
    """
    单类别训练时选择类别：
    - preset_name 非空且合法：直接用它
    - 否则：终端交互选择
    """
    if preset_name is not None:
        if preset_name not in available_class_names:
            logger.error(f"SINGLE_CLASS_NAME='{preset_name}' 不在可用类别中: {available_class_names}")
            raise ValueError(
                f"SINGLE_CLASS_NAME='{preset_name}' 不在可用类别中: {available_class_names}"
            )
        logger.error( f"SINGLE_CLASS_NAME='{preset_name}' 不在可用类别中: {available_class_names}")
        return preset_name

    logger.info("请选择单类别训练目标：")
    for idx, cname in enumerate(available_class_names):
        logger.info(f"  [{idx}] {cname}")

    while True:
        raw = input("输入类别序号: ").strip()
        if raw.isdigit():
            i = int(raw)
            if 0 <= i < len(available_class_names):
                return available_class_names[i]
        logger.warning("输入无效，请重新输入。")



def main():


    # 保存超参数
    global_data.esrgan.save_hyper_parameters_txt(f"{global_data.esrgan.OUT_PUT_DIR}/hyper_parameters.txt")
    # 获取类别名
    available_class_names = get_class_names(global_data.esrgan.GR_DATA_ROOT_DIR)

    logger.info(f"一共{len(available_class_names)}个类别：{available_class_names}")

    # 训练模式: all | single | mixed
    mode = global_data.esrgan.TRAIN_CLASS_MODE.lower().strip()
    if mode not in {"all", "single", "mixed"}:
        logger.error(f'TRAIN_CLASS_MODE 仅支持 all/single/mixed，当前为: {global_data.esrgan.TRAIN_CLASS_MODE}')
        raise ValueError(f"TRAIN_CLASS_MODE 仅支持 all/single/mixed，当前为: {global_data.esrgan.TRAIN_CLASS_MODE}")

    run_jobs = []
    if mode == "all":
        # 每个类别读取数据并且训练验证和保存模型
        for class_name in available_class_names:
            run_jobs.append({"run_class_name": class_name, "selected_classes": [class_name]})
    elif mode == "single":
        chosen = select_single_class(available_class_names, global_data.esrgan.SINGLE_CLASS_NAME)
        # 每个类别读取数据并且训练验证和保存模型
        run_jobs.append({"run_class_name": chosen, "selected_classes": [chosen]})
    else:
        # 每个类别读取数据并且训练验证和保存模型
        run_jobs.append({"run_class_name": global_data.esrgan.MIXED_CLASS_TAG, "selected_classes": None})

    for job in run_jobs:
        class_name = job["run_class_name"]
        selected_classes = job["selected_classes"]

        # 几倍上采样倍率来训练
        for SCALE in global_data.esrgan.SCALES:
            # 获取数据 自动根据类别划分数据集并读取，每个类别都安装比例划分训练集和验证集
            # 根据类别和上采样读取数据
            train_loader, validate_loader, test_loader, class_names, samples = load_data(
                gr_data_root_dir=global_data.esrgan.GR_DATA_ROOT_DIR,
                lr_data_root_dir=f"{global_data.esrgan.LR_DATA_ROOT_DIR}/x{int(SCALE * SCALE)}/data",
                batch_size=global_data.esrgan.BATCH_SIZE,
                shuffle=global_data.esrgan.SHUFFLE,
                target_size=global_data.esrgan.TARGET_SIZE,
                train_nums_rate=global_data.esrgan.Train_nums_rate,
                validate_nums_rate=global_data.esrgan.Validate_nums_rate,
                test_nums_rate=global_data.esrgan.Test_nums_rate,
                random_seed=global_data.esrgan.RANDOM_SEED,
                selected_classes=selected_classes,
                class_sample_ratio=global_data.esrgan.CLASS_SAMPLE_RATIO,
                return_test_loader=True
            )
            # 每个类别的图像对和flo文件分别训练验证和保存模型
            for data_type in global_data.esrgan.DATA_TYPES:
                # Start a new wandb run to track this script.
                wandb.init(
                    entity="2950067210-usst",
                    project="esrgan",
                    name=f"{global_data.esrgan.name}_{global_data.esrgan.DESCRIPTION}_{class_name}_{data_type}",
                    config={
                        "createTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "IS_AUTO_DL": global_data.esrgan.IS_AUTO_DL,
                        "epochs": global_data.esrgan.EPOCH_NUMS,
                        "batch_size": global_data.esrgan.BATCH_SIZE,
                        "lr_G":global_data.esrgan. G_LR,
                        "lr_D": global_data.esrgan.D_LR,
                        "RANDOM_SEED":global_data.esrgan. RANDOM_SEED,
                        "SCALE": SCALE,
                        "SHUFFLE": global_data.esrgan.SHUFFLE,
                        "LAMBDA_ADVERSARIAL": global_data.esrgan.LAMBDA_ADVERSARIAL,
                        "LAMBDA_regularization_loss": global_data.esrgan.LAMBDA_regularization_loss,
                        "LAMBDA_loss_pixel": global_data.esrgan.LAMBDA_loss_pixel,
                        "LAMBDA_PIXEL_L1": global_data.esrgan.LAMBDA_PIXEL_L1,
                        "LAMBDA_PIXEL_MSE": global_data.esrgan.LAMBDA_PIXEL_MSE,
                        "PIXEL_WHITE_ALPHA": global_data.esrgan.PIXEL_WHITE_ALPHA,
                        "LAMBDA_GRAY_CONS": global_data.esrgan.LAMBDA_GRAY_CONS,
                        "SAVE_AS_GRAY": global_data.esrgan.SAVE_AS_GRAY,
                        "weight_decay": global_data.esrgan.weight_decay,
                        "g_optimizer_betas": global_data.esrgan.g_optimizer_betas,
                        "d_optimizer_betas": global_data.esrgan.d_optimizer_betas,
                        "Train_nums_rate": global_data.esrgan.Train_nums_rate,
                        "Validate_nums_rate": global_data.esrgan.Validate_nums_rate,
                        "Test_nums_rate": global_data.esrgan.Test_nums_rate,
                        "train_mode": mode,
                        "selected_classes": selected_classes if selected_classes is not None else "ALL_MIXED",
                    },
                )

                # 创建文件夹
                Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}").mkdir(
                    parents=True, exist_ok=True)
                Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.TRAINING_DIR}").mkdir(
                    parents=True, exist_ok=True)
                Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}").mkdir(
                    parents=True, exist_ok=True)
                Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_DIR}").mkdir(
                    parents=True, exist_ok=True)
                Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_ALL_DIR}").mkdir(
                    parents=True, exist_ok=True)
                Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOG_DIR}").mkdir(
                    parents=True, exist_ok=True)

                #初始化训练日志
                logger.add(
                    f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOG_DIR}/training.log",
                    rotation="100 MB",
                    retention="30 days",
                    level="DEBUG",
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {process.name} | {thread.name} | {name}:{module}:{line} | {message}",
                    enqueue=True,
                    backtrace=True,
                    diagnose=True,

                )
                animator = Animator(xlabel='epoch', xlim=[1, global_data.esrgan.EPOCH_NUMS], ylim=[0, 0.5],
                                    legend=global_data.esrgan.loss_label + global_data.esrgan.validate_label)

                generator = Generator(inner_chanel=3).to(global_data.esrgan.device)
                discriminator = Discriminator(inner_chanel=3).to(global_data.esrgan.device)
                if global_data.esrgan.csvOperator is None:
                    global_data.esrgan.csvOperator = CsvTable(
                        file_path=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}/loss_{class_name} _{data_type}_scale_{int(SCALE * SCALE)}.csv",
                        columns=global_data.esrgan.CSV_COLUMNS)
                else:
                    global_data.esrgan.csvOperator.switch_file(
                        file_path=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}/loss_{class_name} _{data_type}_scale_{int(SCALE * SCALE)}.csv")
                if global_data.esrgan.IS_LOAD_EXISTS_MODEL:
                    generator_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/discriminator_{global_data.esrgan.name}.pth"
                    if os.path.exists(generator_save_path):
                        generator.load_state_dict(torch.load(generator_save_path, map_location=global_data.esrgan.device))
                        logger.info(f"Loaded pretrained model generator from {generator_save_path}")
                    else:
                        logger.info("No pretrained model generator found. Starting training from scratch.")

                    discriminator_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/generator_{global_data.esrgan.name}.pth"
                    if os.path.exists(discriminator_save_path):
                        discriminator.load_state_dict(torch.load(discriminator_save_path, map_location=global_data.esrgan.device))
                        logger.info(f"Loaded pretrained model discriminator from {discriminator_save_path}")
                    else:
                        logger.info("No pretrained model discriminator found. Starting training from scratch.")

                g_optimizer = torch.optim.Adam(generator.parameters(), lr=global_data.esrgan.G_LR, betas=global_data.esrgan.g_optimizer_betas,
                                               weight_decay=global_data.esrgan.weight_decay)
                d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=global_data.esrgan.D_LR, betas=global_data.esrgan.d_optimizer_betas,
                                               weight_decay=global_data.esrgan.weight_decay)

                global_data.esrgan.START_TIME = time.time()
                # 轮数
                """
                训练 start
                """
                for epoch in range(global_data.esrgan.EPOCH_NUMS):
                    #动态更新对抗损失
                    current_lambda_adversarial = global_data.esrgan.update_adversarial_weight(epoch)
                    logger.info(
                        f"[Train] Epoch {epoch + 1}: current adversarial weight = {current_lambda_adversarial:.6f}"
                    )
                    generator.train()  # 确保生成器在训练模式
                    discriminator.train()  # 确保判别器在训练模式

                    metric = Accumulator(len(global_data.esrgan.loss_label))
                    train_progress_bar = tqdm(train_loader,
                                              desc=f"Epoch [{epoch + 1}/{global_data.esrgan.EPOCH_NUMS}] {class_name} {data_type} scale_{int(SCALE * SCALE)} Training",
                                              unit="batch", dynamic_ncols=True,
                                            ascii=True,
                                            leave=True,

                                              )

                    for i, batch in enumerate(train_progress_bar):
                        """ 图片对训练"""
                        if data_type == "image_pair":
                            image_pair_train(
                                epoch=epoch,batch=batch, i=i, g_optimizer=g_optimizer,
                                d_optimizer=d_optimizer, generator=generator,
                                discriminator=discriminator, train_progress_bar=train_progress_bar,
                                metric=metric, data_type=data_type, device=global_data.esrgan.device, class_name=class_name, SCALE=SCALE
                            )
                        elif data_type == "flo":
                            """flo文件训练"""
                            flow_train(
                                epoch=epoch,batch=batch, i=i, g_optimizer=g_optimizer,
                                d_optimizer=d_optimizer, generator=generator,
                                discriminator=discriminator, train_progress_bar=train_progress_bar,
                                metric=metric, data_type=data_type, device=global_data.esrgan.device, class_name=class_name, SCALE=SCALE
                            )
                    # 每轮结束后评价一次 验证集只取一轮batch
                    evaluate(epoch=epoch, class_name=class_name, data_type=data_type, device=global_data.esrgan.device,
                             generator=generator, discriminator=discriminator, animator=animator, validate_loader=validate_loader,
                             loss_label=global_data.esrgan.loss_label,validate_label=global_data.esrgan. validate_label, SCALE=SCALE,
                             csvOperator=global_data.esrgan.csvOperator,metric=metric,train_loader_lens=len(train_loader))

                wandb.finish()
                """
                训练 end
                """

                """
                验证集全部验证一遍 start
                """
                evaluate_all(
                    generator=generator,
                    data_loader=validate_loader,
                    class_name=class_name,
                    data_type=data_type,
                    SCALE=SCALE,
                    output_root=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_ALL_DIR}",
                    metrics_csv_path=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_ALL_DIR}/metrics_all.csv",
                    stride=6,
                )
                """
                验证集全部验证一遍 end
                """
                """
                保存 训练集 验证集 测试集的引用地址json合集 方便查看用了哪些数据 而且也可以重新load 
                """
                save_loaders_paths(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/datas_splits.json", train_loader=train_loader,validate_loader=validate_loader, test_loader=test_loader)


    global_data.esrgan.END_TIME = time.time()
    logger.info(f"一共运行：{global_data.esrgan.END_TIME - global_data.esrgan.START_TIME}秒")

if __name__ =="__main__":
    main()
