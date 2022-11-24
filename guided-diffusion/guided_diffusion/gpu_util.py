import torch as t


def set_gpu_use():

    print(t.cuda.current_device())

    t.cuda.set_device(8)

    print("After change current device is", t.cuda.current_device())

# set_gpu_use()
# print(t.cuda.current_device())