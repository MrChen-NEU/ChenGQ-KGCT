import torch
import json
import os
import random
import numpy as np
import subprocess
import re
import transformers
import logging

def save_json(content,
              path: str,
              name: str):
    """
    save experimental data
    :param content: content to save
    :param path: saved file path
    :param name: saved file n
    :return: None
    """
    if not os.path.exists(path):
        # create new directory
        os.makedirs(path)
    f = open(path + name + '.json', 'w', encoding='utf-8')
    json.dump(content, f)
    f.close()


def load_json(path: str, name: str):
    """
    load data
    :param path:file path
    :param name: file name
    :return: data loaded
    """
    f = open(path + name + '.json', 'r', encoding='utf-8')
    content = json.load(f)
    f.close()
    return content


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_default_fp(fp: str):
    if fp == 'fp16':
        torch.set_default_dtype(torch.float16)
    elif fp == 'bf16':
        torch.set_default_dtype(torch.bfloat16)
    elif fp == 'fp64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)



def select_gpu():
    try:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    print(f'auto select gpu-{sorted_used[0][0]}, sorted_used: {sorted_used}')
    return sorted_used[0][0]

def set_device(gpu) -> str:
    assert gpu < torch.cuda.device_count(), f'gpu {gpu} is not available'
    if not torch.cuda.is_available():
        print('NO GPU CAN USE! ')
        print('SUE CPU')
        return 'cpu'
    if gpu == -1:  gpu = select_gpu()
    return f'cuda:{gpu}'


def get_optimizer(args,
                  model: torch.nn.Module):
    if args is None:
        return torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    if args.opt == 'sgd':
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=args.amsgrad,
                               eps=args.eps)
    elif args.opt == 'transformers':
        opt = transformers.AdamW(model.parameters(), lr=5e-5, correct_bias=True)
    return opt

def set_logger(output_dir=None):
    """ set a root logger"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, 'process.log') 
    logging.basicConfig(format = '%(asctime)s %(levelname)-8s %(message)s',  # '%(asctime)s - %(levelname)-8s - %(name)s -   %(message)s'
                        datefmt = '%H:%M:%S %m/%d/%Y',
                        level = logging.INFO,
                        filename=log_filename,
                        filemode='w'
                        )
    
    # 设置日志也输出到console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
def log_hyperparams(args):
    for arg in vars(args):
        logging.info(f'{arg} = {getattr(args, arg)}')