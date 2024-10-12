import os
from network import neuron,layer,network
from network import process_acasxu_nnet_vnnlib, process_mnist_onnx_vnnlib

onnx_folder = "onnx"
nnet_folder = "nnet"
vnnlib_folder = "vnnlib"
config_nnet_path = "config/acasxu_config.txt"
config_onnx_path = "config/mnist_config.txt"
x_nnet_path = "config/acasxu_x.txt"
x_onnx_path = "config/mnist_x.txt"
def process_mnist_config(config_file_path):
    if os.path.getsize(config_file_path) == 0:
        return []
    onnx_vnnlib_pairs = []
    with open(config_file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if '  ' in line:
                onnx_model, num_vnnlib = line.split('  ')
                num_vnnlib = int(num_vnnlib)
                onnx_model_with_suffix = onnx_model + ".onnx"
                vnnlib_files = []
                for j in range(i + 1, i + 1 + num_vnnlib):
                    vnnlib_file = lines[j].strip() + ".vnnlib"
                    vnnlib_files.append(vnnlib_file)
                onnx_vnnlib_pairs.append((onnx_model_with_suffix, vnnlib_files))
                i += num_vnnlib
            i += 1

    return onnx_vnnlib_pairs

def process_acasxu_config(config_file_path):
    if os.path.getsize(config_file_path) == 0:
        return []
    nnet_vnnlib_pairs = []
    with open(config_file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if '  ' in line:
                nnet_model, num_vnnlib = line.split('  ')
                num_vnnlib = int(num_vnnlib)
                nnet_model_with_suffix = nnet_model + ".nnet"
                vnnlib_files = []
                for j in range(i + 1, i + 1 + num_vnnlib):
                    vnnlib_file = lines[j].strip() + ".vnnlib"
                    vnnlib_files.append(vnnlib_file)
                nnet_vnnlib_pairs.append((nnet_model_with_suffix, vnnlib_files))
                i += num_vnnlib
            i += 1

    return nnet_vnnlib_pairs

def read_x_from_file(file_path):
    with open(file_path, 'r') as file:
        return int(file.readline().strip())


onnx_vnnlib_pairs = process_mnist_config(config_onnx_path)
nnet_vnnlib_pairs = process_acasxu_config(config_nnet_path)
m_mnist = read_x_from_file(x_onnx_path)
m_acasxu = read_x_from_file(x_nnet_path)
'''
if len(onnx_vnnlib_pairs) != 0:
    process_mnist_onnx_vnnlib(onnx_folder, vnnlib_folder, onnx_vnnlib_pairs, m_mnist)
'''
if len(nnet_vnnlib_pairs) != 0:
    process_acasxu_nnet_vnnlib(nnet_folder, vnnlib_folder, nnet_vnnlib_pairs, m_acasxu)






