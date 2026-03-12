import  torch
x= torch.tensor(12)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(x)