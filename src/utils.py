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
    input_set = set(vars(args).items())
    get_param_dict = dict(input_set - default_set)
    return get_param_dict


def slack_post(parser, args, val_loss):
    name = os.environ.get("NAME", default="MARK8")
    api = os.environ.get("WEBHOOK_API", default="None")
    now = datetime.now().strftime("🕐 (%Y-%m-%d %H:%M:%S)")
    param = get_param(parser, args)
    table_string = param
    param_table = f"\`\`\`Hyper Parameter\n{table_string}\n\`\`\`"
    text=f"[🔥{param['MODEL']}🔥] ({name})\nMin Validation Loss: 🎖️{val_loss:.5g}🎖️\n{param_table}"
    message=f'curl -s -d "payload={{\\"username\\":\\"{name}\\", \\"text\\":\\"{text}\\"}}" "{api}"'
    e,o = sp.getstatusoutput(message)
    if e != 0:
        raise Exception(f"Slack에 메시지를 보내는 과정에서 다음과 같은 에러 발생.\n{o}")
    return e,o,text