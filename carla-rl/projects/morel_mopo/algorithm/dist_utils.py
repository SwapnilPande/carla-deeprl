import os
import random
import torch.multiprocessing as mp
from threading import Thread
import socket
import time
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def get_world_size():
    return int(os.environ['WORLD_SIZE'])

def get_rank():
    return int(os.environ['RANK'])

def get_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)

def get_addr():
    return os.environ['MASTER_ADDR']

def get_num_servers():
    return int(os.environ['NUM_SERVERS'])

def get_num_workers():
    return int(os.environ['NUM_WORKERS'])

def get_slurm_world_size():
    return int(os.environ['SLURM_NTASKS'])

def get_slurm_rank():
    return int(os.environ['SLURM_PROCID'])

def get_slurm_jobid():
    return int(os.environ['SLURM_JOB_ID'])

def get_slurm_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)

def get_slurm_nodelist():
    return os.environ['SLURM_NODELIST']

def get_slurm_srun_comm_host():
    return os.environ['SLURM_SRUN_COMM_HOST']

def get_slurm_addr():
    node_list = get_slurm_nodelist()
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0: pos1 = len(node_list)
        pos2 = node_list.find(',', beg)
        if pos2 < 0: pos2 = len(node_list)
        node_list = node_list[:min(pos1, pos2)].replace('[', '')

    last_ip = node_list.replace('-', '.').split(',')[0].rsplit('.', 1)[-1]
    comm_host = get_host_ip()
    addr = comm_host.rsplit('.', 1)[0] + '.' + last_ip

    return addr

# def init_param_server_comm(gpu_id_list=None):
def init_param_server_comm():
    rank, world_size = get_rank(), get_world_size()
    # if gpu_id_list is not None:
    #     gpu_id = gpu_id_list[rank % len(gpu_id_list)]
    #     torch.cuda.set_device(gpu_id)
    dist.init_process_group(backend=get_backend())

    num_servers = get_num_servers()
    server_list = list(range(num_servers))
    worker_list = list(range(num_servers, world_size))

    server_group = dist.new_group(ranks=server_list)
    worker_group = dist.new_group(ranks=worker_list)

    return rank, world_size, server_list, worker_list, server_group, worker_group

def run_slurm_param_server(num_servers, num_workers, port=23032, backend='gloo', method='fork'):
    os.environ['DISTRIBUTED_BACKEND'] = backend
    if mp.get_start_method(allow_none=True) != method:
        mp.set_start_method(method, force=True)

    rank, world_size = get_slurm_rank(), get_slurm_world_size()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
    else:
        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = get_slurm_addr()
        # os.environ['MASTER_ADDR'] = get_slurm_comm_host()
        # os.environ['MASTER_ADDR'] = get_host_ip()
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

    os.environ['NUM_SERVERS'] = str(num_servers)
    os.environ['NUM_WORKERS'] = str(num_workers)

    print('[dist util 105]', os.environ['MASTER_PORT'], get_slurm_nodelist(), get_slurm_srun_comm_host(),
            get_host_ip(), os.environ['MASTER_ADDR'], os.environ['WORLD_SIZE'], os.environ['RANK'])

    return rank, gpu_id, world_size


def run_param_server(server_func, worker_func, num_servers, num_workers,
    resources, ip, port, mp_method='fork'):
    world_size = num_servers + num_workers
    os.environ['MASTER_ADDR'] = str(ip)
    os.environ['MASTER_PORT'] = str(random.randint(10000, 50000))
    os.environ['DISTRIBUTED_BACKEND'] = 'gloo'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['NUM_SERVERS'] = str(num_servers)
    os.environ['NUM_WORKERS'] = str(num_workers)

    if mp.get_start_method(allow_none=True) != mp_method:
        mp.set_start_method(mp_method, force=True)

    proc_list = []
    for rank in range(num_servers):
        proc_list.append(mp.Process(target=server_func, args=(rank, resources)))
    for rank in range(num_servers, world_size):
        proc_list.append(mp.Process(target=worker_func, args=(rank, resources)))
    for p in proc_list:
        p.start()
    for p in proc_list:
        p.join()

def isend(overhead, payload=None, dst=0, tag=0, comm='cpu'):
    _overhead_msg, _payload_msg = overhead, payload
    if not isinstance(_overhead_msg, torch.Tensor):
        if hasattr(_overhead_msg, '__iter__'):
            _overhead_msg = torch.tensor(overhead, dtype=torch.float32)
        else:
            _overhead_msg = torch.tensor([overhead], dtype=torch.float32)
    msg = _overhead_msg.to(comm)
    if _payload_msg is not None:
        if not isinstance(_payload_msg, torch.Tensor):
            try:
                _payload_msg = parameters_to_vector(payload).detach()
            except:
                raise TypeError('unrecognized payload type, not a vector tensor nor an iterator')
        _payload_msg = _payload_msg.to(comm)
        msg = torch.cat((msg, _payload_msg))
    return dist.isend(msg, dst, tag=tag)

def recv(overhead_len, payload_len=0, src=None, tag=0, comm='cpu', device='cpu'):
    msg = torch.zeros(overhead_len, dtype=torch.float32, device=comm)
    if payload_len > 0:
        _payload_msg = torch.zeros(payload_len, dtype=torch.float32, device=comm)
        msg = torch.cat((msg, _payload_msg))
    dist.recv(msg, src, tag=tag)
    if payload_len > 0:
        _overhead_msg, _payload_msg = msg[:overhead_len], msg[overhead_len:].to(device)
        _overhead_msg = list(map(int, _overhead_msg))
        return _overhead_msg, _payload_msg
    return list(map(int, msg))


if __name__ == '__main__':
    run_slurm_param_server(0, 0)
    dist.init_process_group(backend=get_backend())
    time.sleep(3)
    print('DONE')




