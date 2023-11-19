import subprocess
import os
import time
import threading
import queue
from itertools import product
import argparse
import json
import random
import tempfile
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

execute_path = None
last_tmp_dir = None
project_name_base = None
entity_name = None

def update_config(tuning_config_path : str) -> None:
    """
    replace old keys with new keys
    """
    keys_to_replace = {
        "CUDA_VISIBLE_DEVICES" : "cuda_device",
        "PORT" : "port"
    }
    with open(tuning_config_path, 'r', encoding='utf-8') as f:
        tuning_config_new = json.load(f)
    for keys in keys_to_replace:
        if keys in tuning_config_new:
            tuning_config_new[keys_to_replace[keys]] = tuning_config_new[keys]
            del tuning_config_new[keys]
    with open(tuning_config_path, 'w', encoding='utf-8') as f:
        json.dump(tuning_config_new, f, indent=4)

def create_log_tracker_config(template_path_to_read:str, project_name, dict_args:dict, force_generate:bool=True, args_to_remove:list = []):
    """
    Creates log tracker config from template. Stringifies the setups, and adds random 6 length alphanumeric string to the end of the project name.
    """
    if not force_generate and template_path_to_read == 'none':
        return None
    # read template, if not exist, but force_generate is true, create new template
    if not os.path.exists(template_path_to_read):
        if force_generate:
            template = \
r'''[wandb]
    name = "{0}"
    entity = "{entity}"
'''
        else:
            raise OSError("Template path does not exist : "+template_path_to_read)
    else:
        with open(template_path_to_read, 'r', encoding='utf-8') as f:
            template = f.read()
    merged_string = f"{project_name}_"+"_".join([f"{key}={value}" for key, value in dict_args.items() if key not in args_to_remove]) + "_" + generate_random_string()
    new_template = template.format(
        merged_string,
        entity=entity_name
    )
    # if entity_name is None or "", remove entity line
    if entity_name == "" or entity_name is None:
        new_template_fixed = ""
        for line in new_template.split('\n'):
            if "entity" not in line:
                new_template_fixed += line + '\n'
        new_template = new_template_fixed
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_toml_file:
        temp_toml_file.write(new_template)
    return temp_toml_file.name
    
    
def generate_random_string(length:int=6) -> str:
    """
    Generates random string of length 6
    """
    # pick 10 + 26 = 36 characters
    characters_to_use = '0123456789abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(characters_to_use) for _ in range(length))


def generate_config(default_configs, **modified_kwargs) -> dict:
    """
    modified_kwargs: dict of key, value pairs to be modified from default_configs
    If value is empty string or None, it will not be modified.
    """
    copied_config = default_configs.copy()
    for key, value in modified_kwargs.items():
        if key not in default_configs:
            print(f"key {key} not in default_configs")
        if value == "" or value is None:
            continue
        copied_config[key] = value
    return copied_config

def load_default_config(config_path:str):
    """
    config_path: path to json file containing default configs
    Loads default configs from json file, and returns a dict of configs
    """
    default_configs = {
        'project_name_base' : "BASE", 
        'model_file' :'./model.safetensors',
        'optimizer' : 'AdamW8bit',
        'network_dim' : 16,
        'network_alpha' : 8,
        'conv_dim' : 8,
        'conv_alpha' : 1,
        'num_repeats' : 10,
        'epoch_num' : 10,
        'train_batch_size' : 4,
        'unet_lr' : 1e-4,
        'text_encoder_lr' : 2e-5,
        'target_path' : './train',
        'temp_dir' : './tmp',
        'images_folder' : '',
        'cuda_device' : '0',
        'repo_dir' : '.',
        'port' : 20060,
        'sample_opt' : 'epoch',
        'sample_num' : 1,
        'seed' : 42,
        'prompt_path' : './prompt/prompt.txt',
        'keep_tokens' : 0,
        'resolution' : 768,
        'lr_scheduler' : 'cosine_with_restarts',
        'lora_type' : 'LoRA',
        'custom_dataset' : None,
        'clip_skip' : 2,
        'max_grad_norm' : 0,
        'up_lr_weight' : '[1,1,1,1,1,1,1,1,1,1,1,1]',
        'down_lr_weight' : '[1,1,1,1,1,1,1,1,1,1,1,1]',
        'mid_lr_weight' : 1,
        'lbw_weights' : '', # [1,]*17 or [1]* 16, modify this if you want
        'adamw_weight_decay' : 0.01, #default 0.01
        'log_with' : None,
        'wandb_api_key' : '',
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            default_configs_loaded = json.load(f)
    except (FileNotFoundError):
        print(f"Couldn't load config file at {config_path}")
        if config_path != '':
            raise FileNotFoundError(f"Couldn't load config file at {config_path}")
        else:
            default_configs_loaded = {}
    except json.JSONDecodeError as e:
        print(f"Malformed json file at {config_path}")
        if config_path != '':
            raise json.JSONDecodeError(f"Malformed json file at {config_path}", e.doc, e.pos)
        else:
            default_configs_loaded = {}
    for keys in default_configs:
        if keys not in default_configs_loaded:
            default_configs_loaded[keys] = default_configs[keys]
    default_configs = default_configs_loaded
    return default_configs

def convert_relative_path_to_absolute_path(dict_config:dict):
    """
    dict_config: dict of configs
    Converts relative path to absolute path
    """
    for key, value in dict_config.items():
        if key in ['target_path', 'temp_dir', 'images_folder', 'model_file', 'prompt_path']:
            dict_config[key] = os.path.abspath(value)
    return dict_config

def generate_tuning_config(config_dict, **modified_kwargs) -> dict:
    """
    modified_kwargs: dict of key, value pairs to be modified from default_configs
    """
    new_config = config_dict.copy()
    for keys in config_dict.keys():
        # remove _list
        if keys.endswith('_list'):
            del new_config[keys]
    new_config.update(modified_kwargs)
    return new_config

def load_tuning_config(config_path:str):
    """
    config_path: path to json file containing default configs
    Loads default configs from json file, and returns a dict of configs
    """
    tuning_config = {
        # example, you can input as _list for iterating
        #'unet_lr_list' : [1e-5, 1e-4, 2e-4, 3e-4],
        #'text_encoder_lr_list' : [1e-5, 2e-5, 3e-5, 4e-5],
        #'network_alpha_list' : [2,4,8],
        #'network_dim_list' : [16],
        #'clip_skip_list' : [1,2],
        #'num_repeats_list' : [10],
        #'seed_list' : [42],
        'cuda_device' : '0',
        'port' : 20060,
        'sample_opt' : 'epoch',
        'sample_num' : 1,
        'prompt_path' : './prompt/prompt.txt',
        'keep_tokens' : 0,
        'resolution' : 768,
        'lr_scheduler' : 'cosine_with_restarts',
        'lora_type' : 'LoRA',
        'custom_dataset' : None,
    }
    update_config(config_path)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            tuning_config_loaded = json.load(f)
    except FileNotFoundError:
        print("Couldn't load config file")
        if config_path != '':
            raise FileNotFoundError(f"Couldn't load config file at {config_path}")
        else:
            tuning_config_loaded = {}
    except json.JSONDecodeError as decodeException:
        print("Malformed json file")
        if config_path != '':
            raise json.JSONDecodeError(f"Malformed json file at {config_path}", decodeException.doc, decodeException.pos)
        else:
            tuning_config_loaded = {}
    for keys in tuning_config:
        if keys not in tuning_config_loaded:
            # check if list exists instead, then skip
            tuning_config_loaded[keys] = tuning_config[keys]
    tuning_config = tuning_config_loaded
    return tuning_config

# generate_config('unet_lr' : 1e-5) -> returns new config modified with unet lr

def main_iterator(args):
    """
    Yields commands to be executed
    """
    global last_tmp_dir, execute_path, project_name_base
    project_name_base = args.project_name_base
    model_name = args.model_file
    images_folder = args.images_folder
    cuda_device = args.cuda_device
    venv_path = args.venv_path
    accelerate_path = 'accelerate' # default path
    index_to_skip = args.skip_to_index
    entity_name = args.entity_name
    previous_used_port = None
    # handling venv
    if venv_path != '':
        execute_path = os.path.join(venv_path, 'bin', 'python')
        accelerate_path = os.path.join(venv_path, 'bin', 'accelerate')
    else:
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            execute_path = sys.executable # get path of python executable
        else:
            print("venv not activated, activating venv. This uses relative path, so locate this script in the same folder as venv")
            venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv') # expected venv path
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            if not os.path.exists(venv_path):
                raise ValueError(f"venv not found at {venv_path}")
            # os-specific venv activation, windows -> Scripts, posix -> bin
            if os.name == 'nt': # windows
                execute_path = os.path.join(venv_path, 'Scripts', 'python.exe')
                #print("call " + os.path.abspath(".\\venv\\Scripts\\activate.bat"), shell=True)
                #subprocess.check_call(["call", os.path.abspath(".\\venv\\Scripts\\activate.bat")], shell=True)
            else: # posix
                execute_path = os.path.join(venv_path, 'bin', 'python')
                accelerate_path = os.path.join(venv_path, 'bin', 'accelerate')
                
    print(f"using python executable at {execute_path}")
    train_id = args.train_id_start
    default_configs = load_default_config(args.default_config_path)
    tuning_config = load_tuning_config(args.tuning_config_path)

    # warn if custom_dataset is not None
    if tuning_config['custom_dataset'] is not None:
        ignored_options_name = ['images_folder', 'num_repeats','shuffle_caption', 'keep_tokens', 'resolution']
        print(f"custom_dataset is not None, dataset options {ignored_options_name} will be ignored")
        
    # if log_tracker_config_template is not none, create log tracker config and remove the key
    template_path = None
    if tuning_config['log_tracker_config_template'] != 'none':
        template_path = tuning_config['log_tracker_config_template']
        del tuning_config['log_tracker_config_template']
                
    list_arguments_name = {}
    for arguments, values in tuning_config.items():
        if arguments.endswith('_list'):
            list_arguments_name[arguments.replace('_list', '')] = values
            
    singleton_args = []
    for args in list_arguments_name:
        if len(list_arguments_name[args]) == 1:
            print(f"argument {args} is singleton, will be removed from log_tracker_config")
            singleton_args.append(args)
    if "PORT" in tuning_config:
        tuning_config['port'] = tuning_config['PORT']
        del tuning_config['PORT']
    if tuning_config.get('project_name_base', 'BASE') != 'BASE':
        project_name_base = tuning_config['project_name_base']
    keys_to_remove = {'CUDA_VISIBLE_DEVICES', 'PORT'}
    sets_executed_args = set() # set of executed args
    # skip to index, compare current index until it is equal or bigger than index_to_skip
    current_index = -1
    for args_prod in product(*list_arguments_name.values()):
        current_index += 1
        list_arguments = dict(zip(list_arguments_name.keys(), args_prod))
        # check if this set of arguments is already executed with args_prod
        if template_path is not None:
            log_tracker_config_path = create_log_tracker_config(template_path, project_name_base, list_arguments, True, args_to_remove=singleton_args)
            list_arguments['log_tracker_config'] = log_tracker_config_path
        temp_tuning_config = generate_tuning_config(tuning_config, **list_arguments)
        # check validity
        if temp_tuning_config.get('network_alpha', 8) > temp_tuning_config.get('network_dim', 16):
            continue
        if temp_tuning_config.get('conv_alpha', 1) > temp_tuning_config.get('conv_dim', 8):
            continue
        if temp_tuning_config.get('unet_lr', 1e-4) < temp_tuning_config.get('text_encoder_lr', 2e-5):
            continue
        # if mask_loss is false, set mask_threshold to 0
        if not temp_tuning_config.get('mask_loss', True):
            temp_tuning_config['mask_threshold'] = 0
        # this arguments will be used for overriding default configs
        config_without_log_tracker_config = temp_tuning_config.copy()
        if 'log_tracker_config' in config_without_log_tracker_config:
            del config_without_log_tracker_config['log_tracker_config']
        if str(config_without_log_tracker_config) in sets_executed_args:
            print(f"skipping {config_without_log_tracker_config} because it is already executed")
            continue # skip
        if current_index < index_to_skip:
            print(f"skipping {config_without_log_tracker_config} because it is before index_to_skip")
            continue
        sets_executed_args.add(str(config_without_log_tracker_config))
        config = generate_config(default_configs=default_configs,**temp_tuning_config,
                                )
        # override args
        config['project_name_base'] = project_name_base if project_name_base != "BASE" else config['project_name_base']
        # check if project_name_base is valid, since it will be used for folder name
        project_name_to_check = config['project_name_base']
        if project_name_to_check == '':
            raise ValueError("project_name_base cannot be empty")
        # check invalid characters, {, }, [, ], /, \, :, *, ?, ", <, >, |, .
        invalid_characters = ['{', '}', '[', ']', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '.']
        for characters in invalid_characters:
            if characters in project_name_to_check:
                raise ValueError(f"project_name_base cannot contain {characters}")
        if model_name:
            config['model_file'] = model_name
        if images_folder:
            config['images_folder'] = images_folder
        config['cuda_device'] = temp_tuning_config['cuda_device'] if cuda_device == '' else cuda_device
        config_port = config.get('port', 20060)
        if config_port == '':
            config_port = 20060
        if config_port == previous_used_port:
            config['port'] = config_port + 1
            previous_used_port = config_port + 1
        for keys in keys_to_remove:
            if keys in config:
                del config[keys]
        print(f"running _{train_id}")
        command_inputs = [execute_path, "trainer.py"]
        for arguments, values in config.items():
            if values is None or values == '':
                continue
            command_inputs.append(f"--{arguments}")
            command_inputs.append(str(values))
        command_inputs.append("--custom_suffix")
        command_inputs.append(str(train_id))
        # add accelerate path
        command_inputs.append("--accelerate")
        command_inputs.append(accelerate_path)
        train_id += 1
        last_tmp_dir = config['temp_dir']
        yield command_inputs, config['cuda_device']
        
def execute_command(command_args_list, cuda_devices_to_use, stop_event, device_queue):
    """
    Execute a given command on specified CUDA devices.
    Sets stop_event if process returncode is not 0.
    Returns a dict of stdout, stderr, and returncode.
    """
    # find cuda_device in command_args_list and modify the next argument
    cuda_device_index = -1
    for index, args in enumerate(command_args_list):
        if args == '--cuda_device':
            cuda_device_index = index
            break
    if cuda_device_index == -1:
        raise ValueError("cuda_device not found in command_args_list")
    # replace cuda_device with cuda_devices_to_use
    command_args_list[cuda_device_index+1] = ','.join(cuda_devices_to_use) # we have ['cuda_device', '0,1,2,3']... as replacement
    try:
        process = subprocess.Popen(command_args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # start process with stdout and stderr
        stdout, stderr = process.communicate() # get stdout and stderr
        logging.info(f"Command '{command}' executed with devices {cuda_devices_to_use}. Return code: {process.returncode}")
    except Exception as e:
        logging.error(f"Error executing command '{command}': {e}")
        stop_event.set()
        return {
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

    if process.returncode != 0:
        logging.error(f"Command '{command}' failed with return code: {process.returncode}")
        # print error message
        logging.error(f"Error logs: {stdout.decode('utf-8')}")
        logging.error(f"Error message: {stderr.decode('utf-8')}")
        stop_event.set()
    
    # get used cuda devices back to queue
    for device in cuda_devices_to_use:
        device_queue.put(device)
        logging.info(f"Released device {device} back to the queue")
        
    return {
        'stdout': stdout.decode('utf-8'),
        'stderr': stderr.decode('utf-8'),
        'returncode': process.returncode
    }
    

if __name__ == '__main__':
    # check if venv is activated
    # if not, activate venv
    import sys
    abs_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(abs_path)) # execute from here
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name_base', type=str, default='BASE')
    parser.add_argument('--default_config_path', type=str, default='default_config.json')
    parser.add_argument('--tuning_config_path', type=str, default='tuning_config.json')
    # train_id_start
    parser.add_argument('--train_id_start', type=int, default=0) #optional
    # images_folder
    parser.add_argument('--images_folder', type=str, default='') #optional
    parser.add_argument('--model_file', type=str, default='') #optional
    parser.add_argument('--port', type=str, default='') #optional
    parser.add_argument('--cuda_device', type=str, default='') #optional
    parser.add_argument('--debug', action='store_true', default=False) #optional
    # venv path
    parser.add_argument('--venv_path', type=str, default='') #optional
    parser.add_argument('--skip_to_index', type=int, default=-1) #optional
    # cuda devices that can be used for parallel training
    # if empty, use from arguments
    parser.add_argument('--cuda_devices_distributed', type=str, default='') #optional

    # entity_name
    parser.add_argument('--entity_name', type=str, default='', help= "entity name for wandb, leave empty for default") #optional
    # python automate-train.py --project_name_base BASE --default_config_path default_config.json --tuning_config_path tuning_config.json 
    # --train_id_start 0 --images_folder '' --model_file '' --port '' --cuda_device ''

    args = parser.parse_args()
    entity_name = args.entity_name
    device_queue = queue.Queue()
    if args.cuda_devices_distributed != '':
        if args.cuda_devices_distributed == 'all':
            # get cuda devices from environment variable
            cuda_devices_distributed = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        else:
            cuda_devices_distributed = args.cuda_devices_distributed
        for device in cuda_devices_distributed.split(','):
            device_queue.put(device)
            
    threads:List[threading.Thread] = []
    stop_event = threading.Event()
    try:
        for command, required_devices in main_iterator(args):
            if stop_event.is_set():
                logging.info("Stopping further executions due to previous error")
                break
            if args.debug:
                print(command)
                continue
            if args.cuda_devices_distributed == '':
                # call subprocess.check_call
                subprocess.check_call(command)
            else:
                devices = []
                while len(devices) < len(required_devices.split(',')):
                    # get device from queue
                    device = device_queue.get()
                    devices.append(device)
                    logging.info(f"Allocated device {device} for command '{command}'")
                thread = threading.Thread(target=execute_command, args=(command, devices, stop_event, device_queue))
                thread.start()
                threads.append(thread)
                time.sleep(5) # wait for 5 seconds before starting next thread
        for _i, thread in enumerate(threads): # wait for all threads to finish
            thread.join()
            logging.info(f"Thread {_i} finished execution")
    except KeyboardInterrupt:
        # send kill signal to all threads
        stop_event.set()
        logging.warn("Keyboard interrupt received, stopping all threads, please wait...")
        for thread in threads:
            thread.join()
    logging.info("All threads have completed execution")
    
    if not args.debug:
        subprocess.check_call([execute_path, "merge_csv.py", "--path", last_tmp_dir, "--output", f"result_{project_name_base}.csv"])