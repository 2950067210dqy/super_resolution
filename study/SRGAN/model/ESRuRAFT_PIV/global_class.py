from loguru import logger
import time

import wandb
from datetime import datetime
from pathlib import Path
import inspect
import torch
class global_data:
    class esrgan:
        README = """
           v1-v8 是生成器颗粒损失的消融实验。
           将原始的esrgan 根据颗粒图像对进行优化：
         
           2） 重新调整判别器的梯度冻结和启用，并且对更新判别器时重新判别一次。 
               image_pair 的损失因为我生成了两次previous 和 next 所以计算相关损失要多除以2.
           3） 启用vgg 19 的14
           4） 结构loss 和 物理loss还需调参
           5） 上采样pixelshuffle改成upsample、
           6） 修改给p_loss传入的参数 之前是判别器判别之后的概率 应该是生成的图像和真实的图像
           7) 修改训练过程中 临时的验证的次数为1次batch 并且修改平均信噪比和验证损失的逻辑
           8) 添加_to_gray时图像归一化 
           9） 添加内容损失的权重 0.8 与对抗损失0.2
           10） 减少生成器的RRDB 层 由23层减到11层 可以的 速度提高 轻量化、
           11）将生成器的第一层的卷积层k9换成三个并行的卷积层k3 k5 k7 并联concat 后 1x1的卷积 -> 64
           12）将上采样部分改得更细一点  每次上采样后接一个小残差块 Upsample Conv LeakyReLU ResidualBlock ,上采样后的细节修复会更强，尤其对小颗粒边缘恢复更有帮助
           13）改成双帧式针对piv的esrgan
           14）将卷积的padding 0填充改成反射填充 reflect  replicate 
               zero padding 像这样： 图像外面全是黑洞卷积一到边缘就看到很多假黑像素
               reflect padding 像这样：图像边缘像镜子一样往外延伸 卷积在边缘看到的还是类似原图的纹理 解决padding artifact
           15）与RAFT联合
           16）将RAFT的epe损失补偿到生成器的g_loss中
           ......
           ESRuRAFT_PIV_v1 是最差的上采样
           ESRuRAFT_PIV_v2 是batchsize 2
            ESRuRAFT_PIV_v3 是batchsize 4
            ESRuRAFT_PIV_v4  之前都没有启用对抗损失 现在启用对抗损失
            ESRuRAFT_PIV_vtest  加EPE损失权重到1
           17） ESRuRAFT_PIV_vtest2  加EPE损失权重到1 使用RAFT128
           18） ESRuRAFT_PIV_v5 将生成器的图像对一致性损失改成 图像一致性不是直接比较两帧原坐标，而是先用光流对齐再比较 基于光流运动对齐的 warp 一致性思想
           """
        #运行环境是否是autoDL
        IS_AUTO_DL = True
        AUTODL_DATA_PATH = rf"/root/autodl-tmp" if IS_AUTO_DL else r""
        # =========================
        # 训练任务标识
        # =========================
        name = "ESRuRAFT_PIV"  # 当前实验名（用于输出目录/模型名/wandb run名）
        DESCRIPTION = "_trash_Test"  # 实验补充描述（可写损失配置、数据版本等）
        name +=DESCRIPTION

        #整体项目注释
        # 类别训练模式: "all" | "single" | "mixed"
        TRAIN_CLASS_MODE = "mixed"
        # 当 TRAIN_CLASS_MODE="single" 时可预设；为 None 则运行时让你输入选择
        SINGLE_CLASS_NAME = None
        # mixed 模式下的目录名/日志名
        MIXED_CLASS_TAG = "mixed_all_classes"
        #每个类别加载多少的数据 50%
        CLASS_SAMPLE_RATIO =0.5
        # =========================
        # 设备与模型加载
        # =========================
        device = torch.device("cuda")  # 训练设备
        IS_LOAD_EXISTS_MODEL = False  # 是否从已保存模型断点继续训练
        AMP =True #是否开启混合精度训练
        # =========================
        # 可视化与保存相关
        # =========================
        SAVE_AS_GRAY = True  # True: 保存为灰度图(1通道)；False: 按原通道保存 只影响图片对，不影响flo文件,同时处理相关损失函数也会按照这个
        # =========================
        # 训练主超参数
        # =========================
        EPOCH_NUMS = 50 # 训练轮数 50
        BATCH_SIZE = 4 # batch 大小
        PRE_TRIAN_G_EPOCH = 1 #预训练G完成的轮次 从1开始 就是从第几轮开始弃用对抗损失
        TRAIN_DATA_SAVING_STEP =250 #每隔多少steps保存一次生成的图片 50
        SHUFFLE = True  # 训练集是否打乱
        TARGET_SIZE = None  # 数据加载时是否统一 resize 到该尺寸
        RANDOM_SEED = 42  # 数据划分随机种子
        # SCALES = [2,math.sqrt(8),4] # 生成器上采样倍率（内部两次 PixelShuffle）
        SCALES = [2]  # 生成器上采样倍率（内部两次 PixelShuffle）
        # =========================
        # RAFT 配置
        # =========================
        GRU_ITERS = 12 #RAFT的GRU迭代次数 12
        """
        RAFT的上采样方法
            convex 上采样8倍  是原版 RAFT 最经典的做法。update_block 会额外预测一个 up_mask，然后用 3x3 邻域的加权组合做内容自适应上采样。
             bicubic 上采样2倍 bicubic8 上采样8倍  #nn.Upsample(scale_factor=2, mode='bicubic')
             lanczos4 ->2*        lanczos4_8 ->2*->4*->8*    用的LanczosUpsampling模块
        """
        RAFT_UPSAMPLE = 'convex'

        # =========================
        # 损失项系数
        # =========================
        LAMBDA_CONTENT = 1  # 感知损失中的内容项权重 vgg
        RAFT_EPE_WEIGHT = 1  # 生成器侧附加的 RAFT EPE 反作用权重，用来让更小的 EPE 反向约束 SR 结果
        # `LAMBDA_ADVERSARIAL` 作为“当前生效值”保留，
        # 每个 epoch 开始时会由 update_adversarial_weight(...) 动态刷新。
        LAMBDA_ADVERSARIAL = 0.0005
        # 动态对抗权重调度：
        # - 前期让 G 先学稳定重建，避免一上来就被 GAN 拉去造伪纹理
        # - 中后期再逐步给一点 adversarial，补局部真实感
        # 注意：这里的 END 不建议再设到 0.2，你已经验证过那会明显放大边界伪影。
        ADVERSARIAL_WEIGHT_START = 0.0005
        ADVERSARIAL_WEIGHT_END = 0.02
        ADVERSARIAL_WARMUP_EPOCHS = EPOCH_NUMS-10
        ADVERSARIAL_WEIGHT_SCHEDULE = "linear"  # 当前支持: linear | constant


        LAMBDA_PHYSICAL = 0#感知损失中的内容损失中的物理损失权重 如果需要单独跳里面的参数则设置1
        LAMBDA_STRUCTURE =0#感知损失中内容损失中的结构损失权重 如果需要单独跳里面的参数则设置1


        LAMBDA_regularization_loss = 2e-8  # 正则项权重（当前基本未启用）
        LAMBDA_loss_pixel = 1  # 像素损失总权重 （当前基本未启用）




        LAMBDA_PIXEL_L1 = 0.5 # 像素L1权重 0.5；主重建项，保证整体亮度与局部数值不要漂
        LAMBDA_PIXEL_FFT = 0.004  # 频域重建约束，稳住颗粒尺度与高频分布；过大容易让结果发硬
        LAMBDA_PIXEL_MSE = 1e-3  # 像素MSE权重（当前基本未启用）
        PIXEL_WHITE_ALPHA = 1.0  # 灰度场白点区域加权系数（当前基本未启用）
        LAMBDA_GRAY_CONS = 1e-2  # 灰度三通道一致性约束权重（当前基本未启用）
        # =========================
        # 图像对一致性损失超参数
        # =========================
        LAMBDA_FLOW_WARP_CONSISTENCY = 0.012  # GT flow 引导的 SR 图像对 warp 一致性权重

        # =========================
        # 结构相似性损失超参数
        # =========================
        LAMBDA_SSIM = 0.5    #！important SSIM结构相似损失权重，约束SR与HR在局部结构上的一致性 0.05

        SSIM_WINDOW_SIZE = 11  # SSIM高斯窗口大小，用于计算局部结构统计
        SSIM_SIGMA = 1.5  # SSIM高斯窗口标准差，控制局部统计平滑程度
        SSIM_DATA_RANGE = 1.0  # 图像动态范围，若输入已归一化到[0,1]则设为1.0
        SSIM_K1 = 0.01  # SSIM常数项k1，用于稳定亮度项
        SSIM_K2 = 0.03  # SSIM常数项k2，用于稳定对比度项

        # =========================
        # 优化器超参数
        # =========================
        # 正则项
        weight_decay = 0
        # 优化器 betas
        g_optimizer_betas = (0.5, 0.999)
        d_optimizer_betas = (0.5, 0.999)
        RAFT_optimizer_betas = (0.5, 0.999)
        # 学习率
        G_LR =0.0001 #0.0001
        D_LR = 0.0001 # 0.0001
        RAFT_LR =0.0001
        # =========================
        # 数据集划分比例
        # =========================
        # 训练数据集和验证集合比例 测试集  比例
        Train_nums_rate = 0.8
        Test_nums_rate = 0.1
        Validate_nums_rate = round(1 - Train_nums_rate - Test_nums_rate,2)

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
        DATA_TYPES = ['RAFT']
        # DATA_TYPES = ['image_pair', 'flo']
        # DATA_TYPES = ['image_pair']
        # DATA_TYPES =['flo']
        IMAGE_PAIR_TYPES = ['previous', 'next']  # 图像对中的两个时刻/帧
        """
        超参数 end
        """
        # 这里的顺序必须和 train.py 里 metric.add(...) 的顺序一一对应。
        # 任何一边新增/删除/换位，都要同步改另一边。
        loss_label = ['g_loss', 'g_perceptual_loss', "g_content_loss",
                      "g_adversarial_loss",  'g_loss_pixel',
                      "g_loss_pixel_l1", "g_loss_pixel_mse",'g_loss_ssim', 'g_loss_fft',
                       "g_flow_warp_consistency_loss", "g_flow_warp_consistency_weighted_loss",
                      'd_loss', 'd_real_loss', 'd_fake_loss',
                      'raft_loss', 'raft_epe','raft_1px', 'raft_3px','raft_5px',
                      ]
        validate_label = ['VAL_MSE_LOSS','VAL_SSIM_Loss', 'Avg_PSNR',"VAL_energy_spectrum_mse",
                          "VAL_AEE", "VAL_NORM_AEE_PER100PIXEL"]


        # 存储数据至csv的列名
        CSV_COLUMNS = ['EPOCH'] + loss_label + validate_label + ['time']
        # 实验级汇总指标 CSV：
        # 这一份不记录每个 epoch 的 loss，而是记录一次完整训练/评测后的整体开销指标。
        METRICS_SUMMARY_COLUMNS = [
            "run_name",
            "description",
            "class_name",
            "data_type",
            "scale",
            "device",
            "batch_size",
            "input_lr_shape",
            "input_hr_shape",
            "flow_shape",
            "training_time_hours",
            "gpu_memory_usage_gb",
            "flops_g",
            "inference_time_seconds",
            "trainable_params_m",
            "timestamp",
        ]
        # csv操作实例 CsvTable
        csvOperator = None
        metricsSummaryCsvOperator = None
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
        def get_adversarial_weight(cls, epoch: int) -> float:
            """
            根据当前 epoch 返回生效的对抗损失权重。

            设计原则：
            1. 训练早期优先学习内容重建，避免 GAN 过早主导训练。
            2. 只做小幅升权，把 adversarial 当作“纹理微调项”，不是主损失。
            3. 当 warmup 结束后，权重固定在 END，方便实验复现。
            因为 warmup_epochs = ADVERSARIAL_WARMUP_EPOCHS，所以从 epoch=0 到 epoch=ADVERSARIAL_WARMUP_EPOCHS-1
            一共 ADVERSARIAL_WARMUP_EPOCHS 个点，线性从 ADVERSARIAL_WEIGHT_START 涨到 ADVERSARIAL_WEIGHT_END
            """
            schedule = str(cls.ADVERSARIAL_WEIGHT_SCHEDULE).strip().lower()
            start = float(cls.ADVERSARIAL_WEIGHT_START)
            end = float(cls.ADVERSARIAL_WEIGHT_END)
            warmup_epochs = max(1, int(cls.ADVERSARIAL_WARMUP_EPOCHS))

            if schedule == "constant":
                return start

            if schedule != "linear":
                raise ValueError(f"Unsupported ADVERSARIAL_WEIGHT_SCHEDULE: {cls.ADVERSARIAL_WEIGHT_SCHEDULE}")

            if warmup_epochs == 1:
                return end

            clamped_epoch = min(max(int(epoch), 0), warmup_epochs - 1)
            progress = clamped_epoch / float(warmup_epochs - 1)
            return start + (end - start) * progress

        @classmethod
        def update_adversarial_weight(cls, epoch: int) -> float:
            """
            刷新当前 epoch 使用的对抗损失权重，并同步回运行时配置。
            """
            cls.LAMBDA_ADVERSARIAL = cls.get_adversarial_weight(epoch)
            return cls.LAMBDA_ADVERSARIAL

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
