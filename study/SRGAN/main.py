from loguru import logger
from study.SRGAN.model.basic_srgan.global_class import global_data
from model.basic_srgan import srgan_pipeline
from study.SRGAN.model.esrgan import esrgan_pipeline
from study.SRGAN.model.esrgan_update import pipeline as esrgan_pipeline

models = {
            "srgan":srgan_pipeline.main,
          "esrgan":esrgan_pipeline.main,
        "esrgan_update":esrgan_pipeline.main,
          }
def main():
    logger.add(
        f"{global_data.srgan.OUT_PUT_DIR}/running_log/running.log",
        rotation="100 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {process.name} | {thread.name} | {name}:{module}:{line} | {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True,

    )
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
