from pathlib import Path
import torch
import yaml

def write_basic_config_yaml(save_location:str = "", gpu_ids:str="all", cuda_devices:str = "auto") -> bool:
    """
    Write a basic configuration file for accelerate.
    
    @param save_location: The path to use to store the config file. Will throw error if it is not specified, unlike the accelerate default config command.
    @param gpu_ids: The gpu_ids string, e.g. 'all' or '0,1,2,3'
    
    @return: True if the configuration file was written, False if it was not.
    """
    #compute_environment: LOCAL_MACHINE
    #debug: false
    #distributed_type: 'NO' # 'MULTI_GPU', 'NO'
    #downcast_bf16: 'no'
    #gpu_ids: all # or cuda device ids
    #machine_rank: 0
    #main_training_function: main
    #mixed_precision: 'no'
    #num_machines: 1 
    #num_processes: 1 # as gpu_ids
    #rdzv_backend: static
    #same_network: true
    #tpu_env: []
    #tpu_use_cluster: false
    #tpu_use_sudo: false
    #use_cpu: false
    
    assert save_location != "", "Please provide a save location for the configuration file."
    assert gpu_ids != "", "Please provide a gpu_ids string, e.g. 'all' or '0,1,2,3'"
    path = Path(save_location)
    path.parent.mkdir(parents=True, exist_ok=True)
    # write default yaml config object (which will be overwritten)
    config_dict = {
        'compute_environment': 'LOCAL_MACHINE',
        'debug': False,
        'distributed_type': 'NO',
        'downcast_bf16': 'no',
        'gpu_ids': gpu_ids,
        'machine_rank': 0,
        'main_training_function': 'main',
        'mixed_precision': 'no',
        'num_machines': 1,
        'num_processes': 1,
        'rdzv_backend': 'static',
        'same_network': True,
        'tpu_env': [],
        'tpu_use_cluster': False,
        'tpu_use_sudo': False,
        'use_cpu': False
    }
    gpu_ids_list = gpu_ids.split(',')
    if gpu_ids == 'all':
        config_dict['num_processes'] = torch.cuda.device_count() if cuda_devices.lower() == 'auto' else len(cuda_devices.split(','))
    else:
        config_dict['num_processes'] = len(gpu_ids_list)
    if config_dict['num_processes'] > 1:
        config_dict['distributed_type'] = 'MULTI_GPU'
    config_dict['gpu_ids'] = gpu_ids
    # write yaml config file
    with open(save_location, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f)
    return True
