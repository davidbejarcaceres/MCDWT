import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.python.client import device_lib


def getGPU_names() -> list:
    print("############ GPUs Information ###############")
    gpus: list = tf.config.experimental.list_physical_devices('GPU')
    numberOfGPUs: int = len(gpus)
    print("Numbr of GPUs:  " + str(numberOfGPUs))
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
    return gpus

def getCPU_names() -> list:
    print("############ CPUs Information ###############")
    cpus: list = tf.config.experimental.list_physical_devices('CPU')
    numberOfCPUs = len(cpus)
    print("Numbr of GPUs:  " + str(numberOfCPUs))
    for cpu in cpus:
        print("Name:", cpu.name, "  Type:", cpu.device_type)

    return cpus

def get_available_devices() :
    local_device_protos: list = device_lib.list_local_devices()
    print(local_device_protos)
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def getGPUs_count() -> int:
    gpus: list = tf.config.experimental.list_physical_devices('GPU')
    numberOfGPUs: int = len(gpus)
    return numberOfGPUs


def is_gpu_available() -> bool:
    return tf.test.is_gpu_available()


