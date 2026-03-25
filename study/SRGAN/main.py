
from model.basic_srgan import srgan_pipeline

models = {"srgan":srgan_pipeline.main,}
def main():
    keys = list(models.keys())
    print("==="*10)
    print("可选模型：")
    for i, k in enumerate(keys, 1):
        print(f"{i}. {k}")
    print("===" * 10)
    while True:
        raw = input("请输入序号: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(keys):
                model_key = keys[idx - 1]
                print(f"运行: {model_key}")
                models[model_key]()   # 这里才执行函数
                return
        print("输入无效，请重试。")
if __name__ == "__main__":
    main()