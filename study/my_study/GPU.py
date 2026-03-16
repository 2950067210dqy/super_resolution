import torch


def try_gpu(i=0):
    """如果存在，则返回gpu（i），否则返回cpu"""
    if torch.cuda.device_count()>= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device(f'cpu')
def try_all_gpus():
    device = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return device if device else [torch.device('cpu')]
print(try_all_gpus())
