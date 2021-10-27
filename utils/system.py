import os
import sys
import time
import datetime
import psutil
import json

def get_top_info():

    lines=os.popen('top -bn 1 -p {}'.format(os.getpid())).read().split('\n')
    total_load=lines[0].split('load average:')[-1].replace(',','')[1:]
    try:
        total_load=[float(x) for x in total_load[0:1]]
    except:
        total_load=0
    cpu_rate=float([a for a in lines[-2].split('') if a!=''][8])
    mem_rate=float([a for a in lines[-2].split('') if a!=''][9])
    try:
        dur_time=([a for a in lines[-2].split('')if a!=''][10])
    except:
        dur_time=0
    return total_load,cpu_rate,mem_rate,dur_time

def list_and_save_all_member(cls,save_path=None):

    cfg={}
    for name,value in vars(cls).items():
        print('    - self.{}:{}'.format(name,value))
        cfg[name]=str(value)
    if save_path is None:
        return None
    with open(save_path,'w') as f:
        json.dump(cfg,f)

def force_print(*args,**kwargs):
    print(*args,**kwargs)
    sys.stdout.flush()

def get_cpu_nums():
    optimal_cpu_count=max(max(
        psutil.cpu_count(logical=False),
        int(os.getenv('BIOMIND_SERVER_CPUS',1)))-1,1)
    return optimal_cpu_count