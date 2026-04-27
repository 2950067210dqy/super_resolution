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
           18） ESRuRAFT_PIV_v5 ESRuRAFT_PIV_vtest2基础上将生成器的图像对一致性损失改成 图像一致性不是直接比较两帧原坐标，而是先用光流对齐再比较 基于光流运动对齐的 warp 一致性思想
           19） ESRuRAFT_PIV_v6 ESRuRAFT_PIV_v6基础上添加动态学习率根据指标的变化调整 生成器与RAFT都添加动态学习率 超分辨下效果不好，但是RAFT效果好！
                ESRuRAFT_PIV_v6_v1 ESRuRAFT_PIV_v6基础上添加动态对抗损失权重变化限制在前面10轮  超分辨效果好，但是RAFT效果不好 
                ESRuRAFT_PIV_v6_v2 ESRuRAFT_PIV_v6_v1基础上 RAFT_EPE_WEIGHT 1->10  LAMBDA_FLOW_WARP_CONSISTENCY 0.012 ->1.2 变成v6那种效果了
                ESRuRAFT_PIV_v7 ESRuRAFT_PIV_v6基础上去除生成器动态学习率变化  !运行到这里
                
                ESRuRAFT_PIV_v6_v3  ESRuRAFT_PIV_v6_v2基础上去除生成器动态学习率变化  RAFT_EPE_WEIGHT 1->3  LAMBDA_FLOW_WARP_CONSISTENCY 0.012 ->1.2
                ESRuRAFT_PIV_v7_v1 ESRuRAFT_PIV_v7基础上RAFT_EPE_WEIGHT 1->3  LAMBDA_FLOW_WARP_CONSISTENCY 0.012 ->1.2
                ESRuRAFT_PIV_v7_v2 前半个轮次数对对抗损失权重动态和图像一致性损失权重，等生成器稳定生成图像之后再后半个轮次数动态EPE损失
                            其中LAMBDA_ADVERSARIAL和LAMBDA_FLOW_WARP_CONSISTENCY按照0-int(EPOCH_NUMS/2) 轮
                            就从0.0005和0.012开始动态增长至0.02和1.2，
                            随后RAFT_EPE_WEIGHT按照int(EPOCH_NUMS/2)+1-EPOCH_NUMS轮从1动态增长至3
            20) ESRuRAFT_PIV_v8: ESRuRAFT_PIV_v7_v2基础上 判别器更改为方案C，D_input = torch.cat([prev, next, abs(next - prev)], dim=1)，
                将对抗损失最终权重改成0.0161；后续取消生成器侧自适应多任务权重，恢复为手动全局权重组合。
                ESRuRAFT_PIV_v8_v1 改FAMO初始权重
                ESRuRAFT_PIV_v8_v2：去除FAMO权重
                
                PIV_A_Esrgan v1:ESRuRAFT_PIV_v8基础上将判别器更改为A-ESRGAN的判别器
                PIV_A_Esrgan v2:PIV_A_Esrgan v1基础上RAFT128（使用论文训练好的RAFT256部分迁移训练） 基础上去除4个类别的数据，GRU迭代次数12变更成论文的16
                PIV_A_Esrgan v3:PIV_A_Esrgan v2基础上 划分数据集根据论文里来 fixed
                PIV_A_Esrgan v4:PIV_A_Esrgan v2基础上 数据量从50%提升到100%，每轮evaluate不在限制一轮batch
           """
        #运行环境是否是autoDL
        IS_AUTO_DL = True
        AUTODL_DATA_PATH = rf"/root/autodl-tmp" if IS_AUTO_DL else r""
        # =========================
        # 训练任务标识
        # =========================
        name = "PIV_A_Esrgan"  # 当前实验名（用于输出目录/模型名/wandb run名）
        DESCRIPTION = "_v4"  # 实验补充描述（可写损失配置、数据版本等）
        name +=DESCRIPTION

        #整体项目注释
        # 类别训练模式:
        # - "all":    每个类别单独训练一次；
        # - "single": 只训练 SINGLE_CLASS_NAME 指定的一个类别；
        # - "mixed":  所有类别混合后按比例随机划分；
        # - "fixed":  训练/验证样本完全由 FIXED_TRAIN_LIST_PATH 和 FIXED_VALIDATE_LIST_PATH 指定。
        TRAIN_CLASS_MODES = ("all", "single", "mixed", "fixed")
        TRAIN_CLASS_MODE = "mixed"
        # 当 TRAIN_CLASS_MODE="single" 时可预设；为 None 则运行时让你输入选择
        SINGLE_CLASS_NAME = None
        # mixed 模式下的目录名/日志名
        MIXED_CLASS_TAG = "mixed_all_classes"
        # fixed 模式下的目录名/日志名；该模式会保留 list 文件中的样本顺序，不使用随机划分。
        FIXED_CLASS_TAG = "fixed_train_validate"
        # fixed 模式训练列表。这里仅把 list 文件作为“样本文件名清单”读取，
        # list 行里的目录前缀不作为真实路径；真实 GR/LR 路径仍沿用原来的数据根目录逻辑。
        FIXED_TRAIN_LIST_PATH = rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_1/FlowData_train.list"
        # fixed 模式验证列表。Validate_nums_rate 会由 train/test 两个 list 的有效行数自动反推。
        FIXED_VALIDATE_LIST_PATH = rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_1/FlowData_test.list"
        # 排除类别超参数：
        # - None 或 []：不排除，读取所有类别；
        # - ["JHTDB_channel_hd", ...]：在 TRAIN_CLASS_MODE 为 all/mixed/fixed 时不加载这些类别。
        # single 模式保持原语义，不主动应用 EXCLUDE_CLASS，避免用户点名单类训练时被隐式过滤。
        EXCLUDE_CLASS = ["JHTDB_channel_hd", "JHTDB_isotropic1024_hd", "JHTDB_mhd1024_hd", "uniform"]
        #每个类别加载多少的数据 50% TRAIN_CLASS_MODE=fixed时无效
        CLASS_SAMPLE_RATIO =1
        # =========================
        # 设备与模型加载
        # =========================
        device = torch.device("cuda")  # 训练设备
        IS_LOAD_EXISTS_MODEL = False  # 是否从已保存模型断点继续训练
        AMP =False #是否开启混合精度训练
        # =========================
        # 训练模式开关
        # =========================
        # USE_RAFT=True:
        #   保持原来的“超分辨率 Generator + RAFT 光流估计”联合训练路径。
        #   Generator 除了 SR 重建/对抗损失，还会吃到 RAFT EPE 反作用项。
        # USE_RAFT=False:
        #   只训练 PIV 图像对超分辨率，不实例化 RAFT，不创建 RAFT optimizer/scheduler，
        #   训练日志、验证曲线、evaluate_all 也只输出 image_pair 的超分指标。
        #   该模式适合单独验证 Attention U-Net 判别器对颗粒图像超分质量的贡献。
        USE_RAFT = True
        # =========================
        # 可视化与保存相关
        # =========================
        SAVE_AS_GRAY = True  # True: 保存为灰度图(1通道)；False: 按原通道保存 只影响图片对，不影响flo文件,同时处理相关损失函数也会按照这个
        # =========================
        # 训练主超参数
        # =========================
        EPOCH_NUMS = 50 # 训练轮数 50
        START_EPOCH  = 1#从哪个epoch开始 从1开始
        BATCH_SIZE = 4 # batch 大小
        PRE_TRIAN_G_EPOCH = 1 #预训练G完成的轮次 从1开始 就是从第几轮开始弃用对抗损失
        TRAIN_DATA_SAVING_STEP =1000 #每隔多少steps保存一次生成的图片 50
        SHUFFLE = True  # 训练集是否打乱 TRAIN_CLASS_MODE=fixed时无效
        TARGET_SIZE = None  # 数据加载时是否统一 resize 到该尺寸
        RANDOM_SEED = 42  # 数据划分随机种子 TRAIN_CLASS_MODE=fixed时无效
        # SCALES = [2,math.sqrt(8),4] # 生成器上采样倍率（内部两次 PixelShuffle）
        SCALES = [2]  # 生成器上采样倍率（内部两次 PixelShuffle）
        # =========================
        # RAFT 配置
        # =========================
        GRU_ITERS = 16 #RAFT的GRU迭代次数；从 12 调整到 16，让训练/验证默认使用更充分的迭代 refinement。
        """
        RAFT的上采样方法
            convex 上采样8倍  是原版 RAFT 最经典的做法。update_block 会额外预测一个 up_mask，然后用 3x3 邻域的加权组合做内容自适应上采样。
             bicubic 上采样2倍 bicubic8 上采样8倍  #nn.Upsample(scale_factor=2, mode='bicubic')
             lanczos4 ->2*        lanczos4_8 ->2*->4*->8*    用的LanczosUpsampling模块
        """
        # RAFT_MODEL_TYPE 控制 PIV_ESRGAN_RAFT_Model 里 self.piv_RAFT 的具体网络结构。
        # 可选值：
        # - "RAFT":    最早的基础 RAFT 实现，保持原接口；
        # - "RAFT128": 内部在 1/4 分辨率估计光流，适合更细的相关体/GRU 网格；
        # - "RAFT256": 内部在 1/8 分辨率估计光流，显存更省，和 ckpt_256.tar 的结构一致。
        # 本分支之前硬编码为 RAFT128，所以默认仍为 "RAFT128"，保证旧实验不改配置时行为不变。
        RAFT_MODEL_TYPES = ("raft", "raft128", "raft256")
        RAFT_MODEL_TYPE = "RAFT128"
        RAFT_UPSAMPLE = 'convex'
        # 仅 PIV_A_Esrgan 支持这个迁移开关：
        # True 时，如果 RAFT_MODEL_TYPE="RAFT128"，会尝试把一个 RAFT256 checkpoint 中 shape 完全一致的权重
        # 迁移到 RAFT128，shape 不一致的层会跳过并保留 RAFT128 自身随机初始化。
        # 这不是无损迁移，尤其 update_block.mask.2 的 576 通道(8x8x9)无法直接装入 RAFT128 的
        # 144 通道(4x4x9)，因此建议只作为预训练初始化，之后继续 fine-tune。
        RAFT128_INIT_FROM_RAFT256 = True
        #相对 SRGAN 根目录的路径；
        RAFT128_INIT_FROM_RAFT256_CKPT = "RAFT_CHECKPOINT/ckpt_256.tar"
        # 下列两个开关只在 RAFT128_INIT_FROM_RAFT256=True 时生效。
        # optimizer 迁移会按参数名和 shape 做安全过滤：只迁移 shape 匹配参数的 AdamW 状态；
        # scheduler 迁移则直接读取 checkpoint 里的 ReduceLROnPlateau 状态。
        # 如果当前实验已经成功从 OUT_PUT_DIR 恢复了自己的 optimizer/scheduler，pipeline 会优先保留恢复结果。
        RAFT128_INIT_FROM_RAFT256_OPTIMIZER = True
        RAFT128_INIT_FROM_RAFT256_SCHEDULER = True

        # =========================
        # A-ESRGAN 风格判别器配置
        # =========================
        # 判别器仍然接收 3 通道输入，但这 3 个通道不是 RGB，而是：
        # [prev_gray, next_gray, abs(next_gray - prev_gray)]。
        # 这样 U-Net 判别器的空间 logit map 能同时评价单帧颗粒外观和帧间亮度变化。
        DISCRIMINATOR_BASE_CHANNELS = 32
        # 第一阶段建议保持 False，只使用单尺度 Attention U-Net，降低显存和训练震荡风险；
        # 如果单尺度稳定，再把它切成 True，启用 A-ESRGAN 类似的 1x + 2x 多尺度判别器。
        DISCRIMINATOR_USE_MULTISCALE = True
        # spectral normalization 用来稳定 GAN 判别器，尤其适合当前 batch size 较小的颗粒图训练。
        DISCRIMINATOR_SPECTRAL_NORM = True

        # =========================
        # 损失项系数
        # =========================
        LAMBDA_VGG = 1.0  # VGG 感知损失权重；感知损失现在只表示 VGG feature L1，不再混入对抗损失
        # =========================
        # 动态损失权重调度配置
        # =========================
        # 下面三个权重都采用同一套 warm-start 调度规则：
        # 1. epoch <= *_WARMSTART_EPOCHS 时，权重固定为 *_WEIGHT_START。
        # 2. *_WARMSTART_EPOCHS < epoch < *_WARMUP_EPOCHS 时，权重按 schedule 从 START 线性涨到 END。
        # 3. epoch >= *_WARMUP_EPOCHS 时，权重固定为 *_WEIGHT_END。
        # 注意：训练循环里的 epoch 是 0-based；显示日志里的第 N 轮对应 epoch=N-1。
        # 因此最后一轮的 epoch index 是 EPOCH_NUMS - 1。

        # `LAMBDA_ADVERSARIAL` 是当前 epoch 真正生效的生成器对抗损失权重。
        # 按你的设定：从第 0 轮开始由 0.0005 线性增长，到 int(EPOCH_NUMS/2) 达到 0.0161，之后保持 0.0161。
        LAMBDA_ADVERSARIAL = 0.0005
        ADVERSARIAL_WEIGHT_START = 0.0005
        ADVERSARIAL_WEIGHT_END = 0.0161
        ADVERSARIAL_WARMSTART_EPOCHS = 0
        ADVERSARIAL_WARMUP_EPOCHS = int(EPOCH_NUMS / 2)
        ADVERSARIAL_WEIGHT_SCHEDULE = "linear"  # 当前支持: linear | const | constant

        # `LAMBDA_FLOW_WARP_CONSISTENCY` 是 GT flow 引导的 SR 图像对 warp 一致性损失权重。
        # 按你的设定：从第 0 轮开始由 0.012 线性增长，到 int(EPOCH_NUMS/2) 达到 1.2，之后保持 1.2。
        LAMBDA_FLOW_WARP_CONSISTENCY = 0.012
        FLOW_WARP_CONSISTENCY_WEIGHT_START = 0.012
        FLOW_WARP_CONSISTENCY_WEIGHT_END = 1.057440 if CLASS_SAMPLE_RATIO!=1 else 0.2
        FLOW_WARP_CONSISTENCY_WARMSTART_EPOCHS = 0
        FLOW_WARP_CONSISTENCY_WARMUP_EPOCHS = int(EPOCH_NUMS / 2)
        FLOW_WARP_CONSISTENCY_WEIGHT_SCHEDULE = "linear"  # 当前支持: linear | const | constant

        # `RAFT_EPE_WEIGHT` 是 Generator 侧附加的 RAFT EPE 反作用权重。
        # 按你的设定：前半程保持 1，从 int(EPOCH_NUMS/2)+1 开始线性增长，最后一轮达到 3。
        RAFT_EPE_WEIGHT = 1
        RAFT_EPE_WEIGHT_START = 1
        RAFT_EPE_WEIGHT_END = 3 if CLASS_SAMPLE_RATIO != 1 else 1.5
        if not USE_RAFT:
            RAFT_EPE_WEIGHT = 0
            RAFT_EPE_WEIGHT_START = 0
            RAFT_EPE_WEIGHT_END = 0
        RAFT_EPE_WARMSTART_EPOCHS = int(EPOCH_NUMS / 2) + 1
        RAFT_EPE_WARMUP_EPOCHS = EPOCH_NUMS - 1
        RAFT_EPE_WEIGHT_SCHEDULE = "linear"  # 当前支持: linear | const | constant




        LAMBDA_PIXEL_L1 = 0.5 # 像素L1权重 0.5；主重建项，保证整体亮度与局部数值不要漂
        LAMBDA_PIXEL_FFT = 0.004  # 频域重建约束，稳住颗粒尺度与高频分布；过大容易让结果发硬
        LAMBDA_PIXEL_MSE = 1e-3  # 像素MSE权重（当前基本未启用）

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
        # FAMO 自适应多任务权重配置
        # =========================
        # USE_FAMO 是总开关：
        # - False: 完全沿用上面的手动全局损失权重，训练行为和不使用 FAMO 时一致。
        # - True: 仅对生成器的非对抗损失启用 FAMO；GAN 对抗损失仍由 LAMBDA_ADVERSARIAL 动态调度单独控制。
        USE_FAMO = False
        FAMO_GAMMA = 1e-5  # 论文实现里的 Adam weight_decay，控制 FAMO logits 的正则强度
        FAMO_W_LR = 2.5e-2  # 论文示例使用的 FAMO 权重学习率，决定任务权重调整速度
        FAMO_MAX_NORM = 1.0  # 保留论文接口字段；当前模型中不额外裁剪 Generator 梯度
        FAMO_UPDATE_AFTER_STEP = True  # Generator 更新后重新前向一次，用新 loss 按论文公式更新 FAMO 权重
        FAMO_GENERATOR_TASK_NAMES = (
            ['vgg', 'l1', 'mse', 'ssim', 'fft', 'flow_consistency', 'epe']
            if USE_RAFT
            else ['vgg', 'l1', 'mse', 'ssim', 'fft', 'flow_consistency']
        )
        FAMO_GENERATOR_INIT_WEIGHTS = (
            [0.08, 0.18, 0.22, 0.2, 0.07, 0.18, 0.07]
            if USE_RAFT
            else [0.1, 0.24, 0.24, 0.22, 0.08, 0.12]
        )  # 仅作为 softmax 初始比例，启用 FAMO 后权重会自动归一化为 1

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
        G_LR_reduce_factor =0.2 #新学习率 = 原学习率 × factor
        G_LR_patience_level =2 #容忍指标不改善的 epoch 数量，默认 10，超过该数量后触发学习率衰减
        G_LR_min=1e-8#学习率下限，衰减后不低于此值
        D_LR = 0.0001 # 0.0001
        D_LR_reduce_factor =0.2 #新学习率 = 原学习率 × factor
        D_LR_patience_level =2 #容忍指标不改善的 epoch 数量，默认 10，超过该数量后触发学习率衰减
        D_LR_min=1e-8#学习率下限，衰减后不低于此值
        RAFT_LR =0.0001 #0.0001
        RAFT_LR_reduce_factor =0.2 #新学习率 = 原学习率 × factor
        RAFT_LR_patience_level =2 #容忍指标不改善的 epoch 数量，默认 10，超过该数量后触发学习率衰减
        RAFT_LR_min=1e-8#学习率下限，衰减后不低于此值
        # =========================
        # 数据集划分比例
        # =========================
        # 训练数据集和验证集合比例 测试集  比例 TRAIN_CLASS_MODE=fixed时无效
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
        # =========================
        # RAFT256-PIV 风格 TFRecord 测试配置
        # =========================
        IS_VALIDATE_ALL = True  # 是否执行 evaluate_all 完整验证；默认 True，保持原有验证流程不变。
        IS_TEST = True  # 是否在 evaluate_all 之后启用 test_all；默认 False，避免改变原训练/验证流程。
        is_TEST_CLASS3 = True  # 是否额外测试 tbl/twcf 大图数据集；默认 False，节省显存和测试时间。
        TEST_DIR = "/test_all"  # test_all 统一输出目录，会在该目录下再按 dataset 名称分文件夹。
        TEST_BATCH_SIZE = 1  # RAFT256-PIV_test.py 测试默认 batch_size_test=1，这里单 GPU 保持一致。
        TEST_NUM_THREADS = 8  # DALI TFRecordReader 线程数，和参考测试脚本保持一致。
        TEST_SPLIT_SIZE = 1  # tbl/twcf 滑窗 patch 每次送入模型的数量；显存很大时可调大。
        TEST_OFFSET = 256  # tbl/twcf 全图滑窗窗口大小，沿用 RAFT256-PIV_test.py。
        TEST_SHIFT = 64  # tbl/twcf 全图滑窗步长，沿用 RAFT256-PIV_test.py。
        TEST_AMP = False  # 测试阶段是否启用 AMP；默认 False，和参考脚本一致。
        TEST_PLOT_RESULTS = True  # 是否保存每个 sample 的 png 可视化图。
        TEST_TFRECORD2IDX_SCRIPT = "tfrecord2idx"  # idx 文件缺失时用于生成 idx 的外部工具。
        PIV_RESULTS_TWCF_PATH = rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_3/TWCF/PIV_results_TWCF.npy"  # twcf PascalPIV 对比结果路径。
        MASK_TWCF_PATH = rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_3/TWCF/mask_TWCF.npy"  # twcf 可视化 mask 路径。
        TEST_DATASETS = {
            # 下面路径来自 RAFT256-PIV_test.py，统一放到全局变量，后续换数据只改这里。
            "backstep": {
                "test_tfrecord": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_backstep.tfrecord-00000-of-00001",
                "test_tfrecord_idx": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_backstep.idx",
                "image_height": 256,
                "image_width": 256,
                "label_shape": [12],
            },
            "cylinder": {
                "test_tfrecord": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_cylinder.tfrecord-00000-of-00001",
                "test_tfrecord_idx": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_cylinder.idx",
                "image_height": 256,
                "image_width": 256,
                "label_shape": [12],
            },
            "jhtdb": {
                "test_tfrecord": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_jhtdb.tfrecord-00000-of-00001",
                "test_tfrecord_idx": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_jhtdb.idx",
                "image_height": 256,
                "image_width": 256,
                "label_shape": [12],
            },
            "dns_turb": {
                "test_tfrecord": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_dns_turb.tfrecord-00000-of-00001",
                "test_tfrecord_idx": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_dns_turb.idx",
                "image_height": 256,
                "image_width": 256,
                "label_shape": [12],
            },
            "sqg": {
                "test_tfrecord": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_sqg.tfrecord-00000-of-00001",
                "test_tfrecord_idx": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/test/Test_Dataset_10Imgs_sqg.idx",
                "image_height": 256,
                "image_width": 256,
                "label_shape": [12],
            },
            "tbl": {
                "test_tfrecord": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_3/TBL/Dataset_TransTBL_Original8px_fullFrame_withGT.tfrecord-00000-of-00001",
                "test_tfrecord_idx": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_3/TBL/Dataset_TransTBL_Original8px_withGT_fullFrame.idx",
                "image_height": 256,
                "image_width": 3296,
                "label_shape": [12],
            },
            "twcf": {
                "test_tfrecord": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_3/TWCF/Test_Dataset_AR_rawImage.tfrecord-00000-of-00001",
                "test_tfrecord_idx": rf"{AUTODL_DATA_PATH}/study_datas/sr_dataset/class_3/TWCF/Test_Dataset_AR_rawImage.idx",
                "image_height": 2160,
                "image_width": 2560,
                "label_shape": [12],
            },
        }
        LOG_DIR = "/log" #日志目录
        use_gpu = torch.cuda.is_available()
        Path(OUT_PUT_DIR).mkdir(parents=True, exist_ok=True)

        # 需要训练的数据类型  # 参与训练的数据模态
        DATA_TYPES = ['RAFT'] if USE_RAFT else ['SR']
        # DATA_TYPES = ['image_pair', 'flo']
        # DATA_TYPES = ['image_pair']
        # DATA_TYPES =['flo']
        IMAGE_PAIR_TYPES = ['previous', 'next']  # 图像对中的两个时刻/帧
        """
        超参数 end
        """
        # 这里的顺序必须和 train.py 里 metric.add(...) 的顺序一一对应。
        # 任何一边新增/删除/换位，都要同步改另一边。
        BASE_LOSS_LABEL = [
            'g_loss', 'g_perceptual_loss', "g_content_loss",
            "g_adversarial_loss", 'g_loss_pixel',
            "g_loss_pixel_l1", "g_loss_pixel_mse", 'g_loss_ssim', 'g_loss_fft',
            "g_flow_warp_consistency_loss", "g_flow_warp_consistency_weighted_loss",
            'd_loss', 'd_real_loss', 'd_fake_loss',
        ]
        RAFT_LOSS_LABEL = ['raft_loss', 'raft_epe', 'raft_1px', 'raft_3px', 'raft_5px']
        loss_label = BASE_LOSS_LABEL + (RAFT_LOSS_LABEL if USE_RAFT else [])

        BASE_VALIDATE_LABEL = ['VAL_MSE_LOSS', 'VAL_SSIM_Loss', 'Avg_PSNR', "VAL_energy_spectrum_mse"]
        # 新增 VAL_C_AEE：
        # C-AEE = 0.5 * ESE_norm + 0.5 * AEE_norm
        # 这里只负责定义训练曲线 / CSV 的列名，真正的数值计算在 evaluate.py 中完成。
        RAFT_VALIDATE_LABEL = ["VAL_AEE", "VAL_NORM_AEE_PER100PIXEL", "VAL_C_AEE"]
        validate_label = BASE_VALIDATE_LABEL + (RAFT_VALIDATE_LABEL if USE_RAFT else [])


        # 存储数据至csv的列名
        CSV_COLUMNS = ['EPOCH'] + loss_label + validate_label + ['time']
        # 实验级汇总指标 CSV：
        # 这一份不记录每个 epoch 的 loss，而是记录一次完整训练/评测后的整体开销指标。
        # 这里同时保留两套 profiling 口径：
        # - inference_*：只统计 forward 推理过程，适合论文表格里的 Inference Time / FLOPs。
        # - training_step_*：统计完整 train_step，包含 forward、loss、backward、optimizer.step。
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
            "inference_gpu_memory_usage_gb",
            "inference_flops_g",
            "inference_time_seconds",
            "training_step_gpu_memory_usage_gb",
            "training_step_flops_g",
            "training_step_time_seconds",
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
        def normalized_train_class_mode(cls) -> str:
            """
            返回规范化后的类别训练模式。

            统一做 strip/lower 后，pipeline 只需要依赖这个入口；这样新增 fixed 后，
            不会出现某个脚本仍只接受 all/single/mixed 的旧判断。
            """
            return str(cls.TRAIN_CLASS_MODE).strip().lower()

        @classmethod
        def validate_train_class_mode(cls) -> str:
            """
            校验 TRAIN_CLASS_MODE 是否属于当前支持的模式。

            fixed 也是显式模式：它不会使用随机种子、shuffle 或比例划分，而是完全按两个 list 文件取样。
            """
            mode = cls.normalized_train_class_mode()
            if mode not in cls.TRAIN_CLASS_MODES:
                raise ValueError(
                    f"TRAIN_CLASS_MODE 仅支持 {cls.TRAIN_CLASS_MODES}，当前为: {cls.TRAIN_CLASS_MODE}"
                )
            return mode

        @classmethod
        def normalized_exclude_class_names(cls) -> list[str]:
            """
            标准化 EXCLUDE_CLASS。

            允许配置为 None、[]、单个字符串或字符串列表；返回去重后的类别名列表。
            这个方法只处理配置格式，不扫描磁盘，真正的类别存在性校验由 data_load 负责。
            """
            excluded = cls.EXCLUDE_CLASS
            if excluded is None:
                return []
            if isinstance(excluded, str):
                raw_names = [excluded]
            else:
                raw_names = list(excluded)

            normalized: list[str] = []
            seen_lower: set[str] = set()
            for class_name in raw_names:
                name = str(class_name).strip()
                if not name:
                    continue
                lower_name = name.lower()
                if lower_name in seen_lower:
                    continue
                seen_lower.add(lower_name)
                normalized.append(name)
            return normalized

        @classmethod
        def _fixed_list_row_is_excluded(cls, stripped_line: str) -> bool:
            """
            判断 fixed list 的一行是否命中 EXCLUDE_CLASS。

            list 中的路径前缀不作为真实读取路径，但里面通常带有类别目录名；
            这里只把这些目录片段用于“统计比例时跳过被排除类别”的判断。
            """
            excluded_lookup = {name.lower() for name in cls.normalized_exclude_class_names()}
            if not excluded_lookup:
                return False

            for raw_path in stripped_line.split()[:3]:
                normalized_path = str(raw_path).replace("\\", "/")
                for part in Path(normalized_path).parts:
                    if part.lower() in excluded_lookup:
                        return True
                # fixed list 里有时即使类别目录片段无法可靠识别，文件名本身也带类别前缀；
                # 例如 JHTDB_isotropic1024_hd_00469_img1.tif，应命中 JHTDB_isotropic1024_hd。
                file_stem = Path(normalized_path).stem.lower()
                for excluded_class in excluded_lookup:
                    if file_stem == excluded_class or file_stem.startswith(f"{excluded_class}_"):
                        return True
            return False

        @classmethod
        def _count_fixed_split_list_rows(cls, list_path: str | Path, split_name: str) -> int:
            """
            统计 fixed list 的有效样本行数。

            这里和 data_load.load_fixed_split_entries 保持同样规则：跳过空行和 `#` 注释行。
            同时会跳过 EXCLUDE_CLASS 命中的行；该计数只用于把 Train_nums_rate /
            Validate_nums_rate 反推成当前固定划分的真实比例。
            """
            path = Path(list_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"{split_name} fixed split list does not exist: {path}")

            count = 0
            with path.open("r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and not cls._fixed_list_row_is_excluded(stripped):
                        count += 1
            if count <= 0:
                raise ValueError(f"{split_name} fixed split list has no valid sample rows: {path}")
            return count

        @classmethod
        def update_fixed_split_rates(cls) -> dict:
            """
            根据 fixed train/validate list 的行数同步比例超参数。

            fixed 模式真实划分由 list 文件决定；这里更新比例只是为了日志、wandb 和 hyper_parameters.txt
            能显示和固定列表一致的 Train_nums_rate / Validate_nums_rate，Test_nums_rate 固定为 0。
            """
            train_count = cls._count_fixed_split_list_rows(cls.FIXED_TRAIN_LIST_PATH, "train")
            validate_count = cls._count_fixed_split_list_rows(cls.FIXED_VALIDATE_LIST_PATH, "validate")
            total_count = train_count + validate_count
            cls.Train_nums_rate = train_count / total_count
            cls.Validate_nums_rate = validate_count / total_count
            cls.Test_nums_rate = 0.0
            logger.info(
                "[FixedSplit] Update split rates from list rows: "
                f"train_count={train_count}, validate_count={validate_count}, "
                f"Train_nums_rate={cls.Train_nums_rate:.8f}, "
                f"Validate_nums_rate={cls.Validate_nums_rate:.8f}, Test_nums_rate=0.0"
            )
            return {
                "train_count": train_count,
                "validate_count": validate_count,
                "train_rate": cls.Train_nums_rate,
                "validate_rate": cls.Validate_nums_rate,
            }

        @classmethod
        def normalized_raft_model_type(cls) -> str:
            """
            返回规范化后的 RAFT 网络类型。

            统一在配置层做 strip/lower，允许用户写 "RAFT128"、"raft128" 或带空格的值，
            其它代码只依赖这个入口，避免不同文件里各自写字符串判断导致拼写不一致。
            """
            return str(cls.RAFT_MODEL_TYPE).strip().lower()

        @classmethod
        def validate_raft_model_type(cls) -> str:
            """
            校验 RAFT_MODEL_TYPE 是否属于允许的三种结构。

            这里启动即报错，可以避免训练跑到模型实例化或 checkpoint 加载时才发现
            "RAFT_128"、"raft-256" 这类拼写错误。
            """
            mode = cls.normalized_raft_model_type()
            if mode not in cls.RAFT_MODEL_TYPES:
                raise ValueError(
                    f"RAFT_MODEL_TYPE 仅支持 {cls.RAFT_MODEL_TYPES}，当前为: {cls.RAFT_MODEL_TYPE}"
                )
            return mode

        @classmethod
        def _get_scheduled_weight(
                cls,
                epoch: int,
                start: float,
                end: float,
                warmstart_epoch: int,
                warmup_epoch: int,
                schedule: str,
                schedule_name: str,
        ) -> float:
            """
            统一的 warm-start 权重调度函数。

            参数说明：
            - epoch: 当前训练循环中的 0-based epoch index；例如日志中的第 1 轮对应 epoch=0。
            - start: warm-start 阶段固定使用的初始权重。
            - end: warmup 结束后固定使用的目标权重。
            - warmstart_epoch: 在这个 epoch 及之前，权重保持 start，不做增长。
            - warmup_epoch: 在这个 epoch 及之后，权重保持 end。
            - schedule: 当前支持 linear / const / constant。

            分段逻辑：
            1. epoch <= warmstart_epoch: 返回 start。
            2. warmstart_epoch < epoch < warmup_epoch: 按线性比例从 start 增长到 end。
            3. epoch >= warmup_epoch: 返回 end。
            这样可以明确表达“先稳定一段，再逐步加权，最后固定”的训练策略。
            """
            schedule = str(schedule).strip().lower()
            start = float(start)
            end = float(end)
            warmstart_epoch = max(0, int(warmstart_epoch))
            warmup_epoch = max(warmstart_epoch, int(warmup_epoch))
            current_epoch = max(0, int(epoch))

            # const / constant 用于消融实验：全程固定在 start，不进入线性增长。
            if schedule in ("const", "constant"):
                return start

            if schedule != "linear":
                raise ValueError(f"Unsupported {schedule_name}: {schedule}")

            # warmup_epoch 与 warmstart_epoch 相同或更小时，说明没有线性区间，直接在边界后使用 end。
            if current_epoch <= warmstart_epoch:
                return start
            if current_epoch >= warmup_epoch:
                return end
            if warmup_epoch == warmstart_epoch:
                return end

            progress = (current_epoch - warmstart_epoch) / float(warmup_epoch - warmstart_epoch)
            return start + (end - start) * progress

        @classmethod
        def get_adversarial_weight(cls, epoch: int) -> float:
            """
            返回当前 epoch 的生成器对抗损失权重。
            当前配置为：0 -> int(EPOCH_NUMS/2) 从 0.0005 线性增长到 0.02，之后保持 0.02。
            """
            return cls._get_scheduled_weight(
                epoch=epoch,
                start=cls.ADVERSARIAL_WEIGHT_START,
                end=cls.ADVERSARIAL_WEIGHT_END,
                warmstart_epoch=cls.ADVERSARIAL_WARMSTART_EPOCHS,
                warmup_epoch=cls.ADVERSARIAL_WARMUP_EPOCHS,
                schedule=cls.ADVERSARIAL_WEIGHT_SCHEDULE,
                schedule_name="ADVERSARIAL_WEIGHT_SCHEDULE",
            )

        @classmethod
        def get_flow_warp_consistency_weight(cls, epoch: int) -> float:
            """
            返回当前 epoch 的 GT-flow warp 图像对一致性损失权重。
            当前配置为：0 -> int(EPOCH_NUMS/2) 从 0.012 线性增长到 1.2，之后保持 1.2。
            """
            return cls._get_scheduled_weight(
                epoch=epoch,
                start=cls.FLOW_WARP_CONSISTENCY_WEIGHT_START,
                end=cls.FLOW_WARP_CONSISTENCY_WEIGHT_END,
                warmstart_epoch=cls.FLOW_WARP_CONSISTENCY_WARMSTART_EPOCHS,
                warmup_epoch=cls.FLOW_WARP_CONSISTENCY_WARMUP_EPOCHS,
                schedule=cls.FLOW_WARP_CONSISTENCY_WEIGHT_SCHEDULE,
                schedule_name="FLOW_WARP_CONSISTENCY_WEIGHT_SCHEDULE",
            )

        @classmethod
        def get_raft_epe_weight(cls, epoch: int) -> float:
            """
            返回当前 epoch 的 Generator 侧 RAFT EPE 反作用权重。
            当前配置为：前半程保持 1，从 int(EPOCH_NUMS/2)+1 到最后一轮线性增长到 3。
            """
            return cls._get_scheduled_weight(
                epoch=epoch,
                start=cls.RAFT_EPE_WEIGHT_START,
                end=cls.RAFT_EPE_WEIGHT_END,
                warmstart_epoch=cls.RAFT_EPE_WARMSTART_EPOCHS,
                warmup_epoch=cls.RAFT_EPE_WARMUP_EPOCHS,
                schedule=cls.RAFT_EPE_WEIGHT_SCHEDULE,
                schedule_name="RAFT_EPE_WEIGHT_SCHEDULE",
            )

        @classmethod
        def update_dynamic_loss_weights(cls, epoch: int) -> dict:
            """
            每个 epoch 开始时调用一次，把三类动态权重同步回运行时配置。
            训练中的 loss 函数直接读取 cls.LAMBDA_ADVERSARIAL、cls.LAMBDA_FLOW_WARP_CONSISTENCY、cls.RAFT_EPE_WEIGHT，
            因此这里更新后，本 epoch 内所有 batch 都会使用同一组固定权重，日志和复现实验也更清晰。
            """
            cls.LAMBDA_ADVERSARIAL = cls.get_adversarial_weight(epoch)
            cls.LAMBDA_FLOW_WARP_CONSISTENCY = cls.get_flow_warp_consistency_weight(epoch)
            # 只做超分辨率时不再计算 RAFT forward，也就不能再把 RAFT EPE 作为 Generator 反作用项。
            # 这里显式把权重固定为 0，避免日志里看起来仍有 RAFT 监督在生效。
            cls.RAFT_EPE_WEIGHT = cls.get_raft_epe_weight(epoch) if cls.USE_RAFT else 0.0
            return {
                "lambda_adversarial": cls.LAMBDA_ADVERSARIAL,
                "lambda_flow_warp_consistency": cls.LAMBDA_FLOW_WARP_CONSISTENCY,
                "raft_epe_weight": cls.RAFT_EPE_WEIGHT if cls.USE_RAFT else "disabled",
            }

        @classmethod
        def update_adversarial_weight(cls, epoch: int) -> float:
            """
            兼容旧调用：只返回当前对抗权重。
            新训练流程请优先调用 update_dynamic_loss_weights(...)，同时刷新三项动态权重。
            """
            cls.update_dynamic_loss_weights(epoch)
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






