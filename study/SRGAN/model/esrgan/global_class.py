from loguru import logger
import time

import wandb
from datetime import datetime
from pathlib import Path
import inspect
import torch
class global_data:
    class esrgan:
        #运行环境是否是autoDL
        IS_AUTO_DL = False
        AUTODL_DATA_PATH = rf"/root/autodl-tmp" if IS_AUTO_DL else r""
        # =========================
        # 训练任务标识
        # =========================
        name = "esrgan"  # 当前实验名（用于输出目录/模型名/wandb run名）
        DESCRIPTION = "v1"  # 实验补充描述（可写损失配置、数据版本等）
        name +=DESCRIPTION
        # 类别训练模式: "all" | "single" | "mixed"
        TRAIN_CLASS_MODE = "all"
        # 当 TRAIN_CLASS_MODE="single" 时可预设；为 None 则运行时让你输入选择
        SINGLE_CLASS_NAME = None
        # mixed 模式下的目录名/日志名
        MIXED_CLASS_TAG = "mixed_all_classes"

        # =========================
        # 设备与模型加载
        # =========================
        device = torch.device("cuda")  # 训练设备
        IS_LOAD_EXISTS_MODEL = False  # 是否从已保存模型断点继续训练
        # =========================
        # 可视化与保存相关
        # =========================
        SAVE_AS_GRAY = True  # True: 保存为灰度图(1通道)；False: 按原通道保存 只影响图片对，不影响flo文件,同时处理相关损失函数也会按照这个
        # =========================
        # 训练主超参数
        # =========================
        EPOCH_NUMS = 20  # 训练轮数
        BATCH_SIZE = 8  # batch 大小
        SHUFFLE = True  # 训练集是否打乱
        TARGET_SIZE = None  # 数据加载时是否统一 resize 到该尺寸
        RANDOM_SEED = 42  # 数据划分随机种子
        # SCALES = [2,math.sqrt(8),4] # 生成器上采样倍率（内部两次 PixelShuffle）
        SCALES = [2]  # 生成器上采样倍率（内部两次 PixelShuffle）

        # =========================
        # 损失项系数
        # =========================
        LAMBDA_ADVERSARIAL = 5e-4  # 感知损失中对抗项权重
        LAMBDA_regularization_loss = 2e-8  # 正则项权重（当前基本未启用）
        LAMBDA_loss_pixel = 1  # 像素损失总权重

        LAMBDA_PIXEL_L1 = 1e-2  # 像素L1权重
        LAMBDA_PIXEL_MSE = 1e-3  # 像素MSE权重
        PIXEL_WHITE_ALPHA = 1.0  # 灰度场白点区域加权系数
        LAMBDA_GRAY_CONS = 1e-2  # 灰度三通道一致性约束权重

        # =========================
        # 优化器超参数
        # =========================
        # 正则项
        weight_decay = 0
        # 优化器 betas
        g_optimizer_betas = (0.5, 0.999)
        d_optimizer_betas = (0.5, 0.999)
        # 学习率
        G_LR = 0.0001
        D_LR = 0.0001
        # =========================
        # 数据集划分比例
        # =========================
        # 训练数据集和验证集合比例 测试集  比例
        Train_nums_rate = 0.8
        Test_nums_rate = 0.0
        Validate_nums_rate = 1 - Train_nums_rate - Test_nums_rate

        # =========================
        # 数据路径与输出路径
        # =========================
        # 真实数据根路径
        GR_DATA_ROOT_DIR = rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_1/data"
        # 低分辨率数据根地址
        LR_DATA_ROOT_DIR = rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_1_lr"

        # 如果路径不存在则创建路径
        OUT_PUT_DIR = f"{AUTODL_DATA_PATH}/train_datas/{name}"  # 实验输出总目录
        TRAINING_DIR = "/training_data"#正在训练输出目录
        LOSS_DIR = "/train_loss"  # 损失曲线目录
        MODEL_DIR = "/train_model"  # 模型权重目录
        PREDICT_DIR = "/predict"  # 预测结果目录
        PREDICT_ALL_DIR = "/predict_all"  # 预测全部结果目录
        LOG_DIR = "/log" #日志目录
        use_gpu = torch.cuda.is_available()
        Path(OUT_PUT_DIR).mkdir(parents=True, exist_ok=True)

        # 需要训练的数据类型  # 参与训练的数据模态
        DATA_TYPES = ['image_pair', 'flo']
        # DATA_TYPES =['flo']
        IMAGE_PAIR_TYPES = ['previous', 'next']  # 图像对中的两个时刻/帧
        """
        超参数 end
        """
        loss_label = ['g_loss', 'g_perceptual_loss', "g_content_loss",
                      "g_adversarial_loss", 'g_regularization_loss', 'g_loss_pixel',
                      "g_loss_pixel_l1", "g_loss_pixel_mse",
                      'd_loss', 'd_real_loss', 'd_fake_loss']
        validate_label = ['Validation_Loss', 'Avg_PSNR']
        # 存储数据至csv的列名
        CSV_COLUMNS = ['EPOCH'] + loss_label + validate_label + ['time']
        # csv操作实例 CsvTable
        csvOperator = None
        # 使用wandb可视化训练过程
        # 初始化 WandB
        # 防止重复 login
        _wandb_logged_in = False
        wandb_key = "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"

        # 开始时间
        START_TIME = time.time()
        #结束时间
        END_TIME = time.time()
        @classmethod
        def ensure_wandb_login(cls):
            if not cls._wandb_logged_in:
                wandb.login(key=cls.wandb_key)
                cls._wandb_logged_in = True

        @classmethod
        def save_hyper_parameters_txt(cls, file_path="hyper_parameter.txt"):
            """
            自动收集当前类的超参数并保存到文本文件。
            规则：
            - 跳过私有属性(_开头)
            - 跳过方法/函数/callable
            - 可序列化常见类型（str/int/float/bool/list/tuple/dict/Path/torch.device）
            """


            def _to_text(v):
                if isinstance(v, torch.device):
                    return f'torch.device("{v.type}")'
                if isinstance(v, Path):
                    return f'r"{str(v)}"'
                if isinstance(v, str):
                    return f'"{v}"'
                return repr(v)

            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [f"# created_at = {created_at}"]

            # 按名字排序，输出稳定
            for name in sorted(dir(cls)):
                if name.startswith("_"):
                    continue
                value = getattr(cls, name)

                # 跳过方法、属性描述符、类/静态方法等
                if inspect.ismethod(value) or inspect.isfunction(value) or inspect.isbuiltin(value):
                    continue
                if isinstance(getattr(type(cls), name, None), (classmethod, staticmethod, property)):
                    continue
                if callable(value):
                    continue

                lines.append(f"{name} = {_to_text(value)}")

            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).write_text("\n".join(lines), encoding="utf-8")
            logger.info(f"hyper_parameter Saved to {file_path}")
        # @classmethod
        # def save_hyper_parameters_txt(cls,file_path="hyper_parameter.txt"):
        #     """将当前实验超参数写入文本文件，便于复现实验。"""
        #     created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #
        #     lines = [
        #         f"# created_at = {created_at}",
        #         f'name = "{cls.name}"',
        #         f'DESCRIPTION = "{cls.DESCRIPTION}"',
        #         f'TRAIN_CLASS_MODE = {cls.TRAIN_CLASS_MODE}',
        #         f'SINGLE_CLASS_NAME = {cls.SINGLE_CLASS_NAME}',
        #         f'MIXED_CLASS_TAG = {cls.MIXED_CLASS_TAG}',
        #         f'device = torch.device("{cls.device.type}")',
        #         f"IS_LOAD_EXISTS_MODEL = {cls.IS_LOAD_EXISTS_MODEL}",
        #         "",
        #         f"SAVE_AS_GRAY = {cls.SAVE_AS_GRAY}",
        #         "",
        #         f"EPOCH_NUMS = {cls.EPOCH_NUMS}",
        #         f"BATCH_SIZE = {cls.BATCH_SIZE}",
        #         f"SHUFFLE = {cls.SHUFFLE}",
        #         f"TARGET_SIZE = {cls.TARGET_SIZE}",
        #         f"RANDOM_SEED = {cls.RANDOM_SEED}",
        #         f"SCALE = {cls.SCALES}",
        #         "",
        #         f"LAMBDA_ADVERSARIAL = {cls.LAMBDA_ADVERSARIAL}",
        #         f"LAMBDA_regularization_loss = {cls.LAMBDA_regularization_loss}",
        #         f"LAMBDA_loss_pixel = {cls.LAMBDA_loss_pixel}",
        #         f"PIXEL_WHITE_ALPHA:{cls.PIXEL_WHITE_ALPHA}",
        #         f"LAMBDA_GRAY_CONS:{cls.LAMBDA_GRAY_CONS}",
        #         f"LAMBDA_PIXEL_L1 = {cls.LAMBDA_PIXEL_L1}",
        #         f"LAMBDA_PIXEL_MSE = {cls.LAMBDA_PIXEL_MSE}",
        #
        #         f"weight_decay = {cls.weight_decay}",
        #         f"g_optimizer_betas = {cls.g_optimizer_betas}",
        #         f"d_optimizer_betas = {cls.d_optimizer_betas}",
        #         f"G_LR = {cls.G_LR}",
        #         f"D_LR = {cls.D_LR}",
        #         "",
        #         f"Train_nums_rate = {cls.Train_nums_rate}",
        #         f"Test_nums_rate = {cls.Test_nums_rate}",
        #         f"Validate_nums_rate = {cls.Validate_nums_rate}",
        #         "",
        #         f'GR_DATA_ROOT_DIR = r"{cls.GR_DATA_ROOT_DIR}"',
        #         f'LR_DATA_ROOT_DIR = r"{cls.LR_DATA_ROOT_DIR}"',
        #         "",
        #         f'OUT_PUT_DIR = r"{cls.OUT_PUT_DIR}"',
        #         f'LOSS_DIR = r"{cls.LOSS_DIR}"',
        #         f'MODEL_DIR = r"{cls.MODEL_DIR}"',
        #         f'PREDICT_DIR = r"{cls.PREDICT_DIR}"',
        #         f'PREDICT_ALL_DIR = r"{cls.PREDICT_ALL_DIR}"',
        #         f"use_gpu = {cls.use_gpu}",
        #         f"DATA_TYPES = {cls.DATA_TYPES}",
        #         f"IMAGE_PAIR_TYPES = {cls.IMAGE_PAIR_TYPES}",
        #     ]
        #
        #     Path(file_path).write_text("\n".join(lines), encoding="utf-8")
        #     print(f"hyper_parameter Saved to {file_path}")
# 模块导入时只执行一次
global_data.esrgan.ensure_wandb_login()
