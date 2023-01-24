import torch as t


def set_gpu_use(device_num):

    print(t.cuda.current_device())

    t.cuda.set_device(device_num)

    print("After change current device is", t.cuda.current_device())

# set_gpu_use()
# print(t.cuda.current_device())