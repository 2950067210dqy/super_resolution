

import wandb
from datetime import datetime
from pathlib import Path
import inspect
import torch
class global_data:
    class srgan:
        # =========================
        # 训练任务标识
        # =========================
        name = "v3"
        DESCRIPTION = ""
        TRAIN_CLASS_MODE = "all"
        SINGLE_CLASS_NAME = None
        MIXED_CLASS_TAG = "mixed_all_classes"

        # =========================
        # 设备与模型加载
        # =========================
        device = torch.device("cuda")
        IS_LOAD_EXISTS_MODEL = False

        # =========================
        # 可视化与保存相关
        # =========================
        SAVE_AS_GRAY = True

        # =========================
        # 训练主超参数
        # =========================
        EPOCH_NUMS = 20
        BATCH_SIZE = 16
        SHUFFLE = True
        TARGET_SIZE = None
        RANDOM_SEED = 42
        SCALES = [2]

        # =========================
        # 损失项系数
        # =========================
        LAMBDA_PERCEPTION = 5e-4
        LAMBDA_regularization_loss = 2e-8
        LAMBDA_loss_pixel = 1
        LAMBDA_PIXEL_L1 = 1e-2
        LAMBDA_PIXEL_MSE = 1e-3
        PIXEL_WHITE_ALPHA = 1.0
        LAMBDA_GRAY_CONS = 1e-2

        # =========================
        # 优化器超参数
        # =========================
        weight_decay = 0
        g_optimizer_betas = (0.5, 0.999)
        d_optimizer_betas = (0.5, 0.999)
        G_LR = 1e-4
        D_LR = 1e-4

        # =========================
        # 数据集划分比例
        # =========================
        Train_nums_rate = 0.8
        Test_nums_rate = 0.0
        Validate_nums_rate = 1 - Train_nums_rate - Test_nums_rate

        # =========================
        # 数据路径与输出路径
        # =========================
        GR_DATA_ROOT_DIR = r"/study_datas/sr_dataset/class_1/data"
        LR_DATA_ROOT_DIR = r"/study_datas/sr_dataset/class_1_lr"

        OUT_PUT_DIR = f"./train_data/{name}"
        LOSS_DIR = "/train_loss"
        MODEL_DIR = "/train_model"
        PREDICT_DIR = "/predict"
        PREDICT_ALL_DIR = "/predict_all"

        use_gpu = torch.cuda.is_available()
        DATA_TYPES = ['image_pair', 'flo']
        IMAGE_PAIR_TYPES = ['previous', 'next']

        loss_label = [
            'g_loss', 'g_perceptual_loss', "g_content_loss",
            "g_adversarial_loss", 'g_regularization_loss', 'g_loss_pixel',
            "g_loss_pixel_l1", "g_loss_pixel_mse",
            'd_loss', 'd_real_loss', 'd_fake_loss'
        ]
        validate_label = ['Validation_Loss', 'Avg_PSNR']
        CSV_COLUMNS = ['EPOCH'] + loss_label + validate_label + ['time']
        csvOperator = None

        # 防止重复 login
        _wandb_logged_in = False
        wandb_key = "wandb_v1_xxx"

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
            print(f"hyper_parameter Saved to {file_path}")
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
        #         f"LAMBDA_PERCEPTION = {cls.LAMBDA_PERCEPTION}",
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
global_data.srgan.ensure_wandb_login()