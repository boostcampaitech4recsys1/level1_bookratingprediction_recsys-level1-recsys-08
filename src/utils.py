import os
import random
import numpy as np
from datetime import datetime
import subprocess as sp
from dotenv import load_dotenv
import torch
from sklearn.model_selection import train_test_split
load_dotenv()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_param(parser, args):
    default_set = set(vars(parser.parse_args([])).items())
    input_items = vars(args).items()
    result=dict()
    for i in input_items:
        if type(i[1])==list or i not in default_set:
            result[i[0]]=i[1]
    return result


def slack_post(parser, args, val_loss):
    name = os.environ.get("NAME", default="MARK8")
    api = os.environ.get("WEBHOOK_API", default="None")
    now = datetime.now().strftime("π (%Y-%m-%d %H:%M:%S)")
    param = get_param(parser, args)
    table_string = param
    param_table = f"\`\`\`Hyper Parameter\n{table_string}\n\`\`\`"
    text=f"[π₯{param['MODEL']}π₯] ({name})\nMin Validation Loss: ποΈ{val_loss:.5g}ποΈ\n{param_table}"
    message=f'curl -s -d "payload={{\\"username\\":\\"{name}\\", \\"text\\":\\"{text}\\"}}" "{api}"'
    e,o = sp.getstatusoutput(message)
    if e != 0:
        raise Exception(f"Slackμ λ©μμ§λ₯Ό λ³΄λ΄λ κ³Όμ μμ λ€μκ³Ό κ°μ μλ¬ λ°μ.\n{o}")
    return e,o,text