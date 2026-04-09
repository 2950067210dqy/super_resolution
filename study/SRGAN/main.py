from loguru import logger

from model.basic_srgan import srgan_pipeline
from study.SRGAN.model.esrgan import esrgan_pipeline as esrgan_pipeline
from study.SRGAN.model.esrgan_update import pipeline as esrgan_update_pipeline
from study.SRGAN.model.PIV_esrgan import pipeline as PIV_esrgan_pipeline
from study.SRGAN.model.PIV_esrgan_RAFT import pipeline as PIV_esrgan_RAFT_pipeline
from study.SRGAN.model.ESRuRAFT_PIV import pipeline as ESRuRAFT_PIV_pipeline




models = {
        "srgan":srgan_pipeline.main,
        "esrgan":esrgan_pipeline.main,
        "esrgan_update":esrgan_update_pipeline.main,
        "piv_esrgan":PIV_esrgan_pipeline.main,
        "piv_esrgan_RAFT":PIV_esrgan_RAFT_pipeline.main,
        "ESRuRAFT_PIV":ESRuRAFT_PIV_pipeline.main,
          }
def main():

    keys = list(models.keys())
    logger.info("==="*10)
    logger.info("可选模型：")
    for i, k in enumerate(keys, 1):
        logger.info(f"{i}. {k}")
    logger.info("===" * 10)
    while True:
        raw = input("请输入序号: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(keys):
                model_key = keys[idx - 1]
                logger.info(f"运行: {model_key}")
                models[model_key]()   # 这里才执行函数
                return
        logger.warning("输入无效，请重试。")
if __name__ == "__main__":
    main()
