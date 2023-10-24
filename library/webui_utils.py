"""
Util class to handle external WebUI calls
"""
import json
import os
import time
from typing import Tuple, Dict, Any
import re
import uuid
import numpy as np
from accelerate import Accelerator
from concurrent.futures import ThreadPoolExecutor
from library.webuiapi.webuiapi import WebUIApi, ControlNetUnit, QueuedTaskResult
from library.webuiapi.test_utils import open_controlnet_image, open_mask_image, raw_b64_img
from queue import Queue

work_queue = Queue() #FIFO queue
def work_queue_thread():
    while True:
        task = work_queue.get()
        task()
        work_queue.task_done()
executor_thread_pool = ThreadPoolExecutor(max_workers=1) # sequential execution
executor_thread_pool.submit(work_queue_thread)
    

def get_thread_pool_executor() -> ThreadPoolExecutor:
    # check if thread pool is shutdown, if so, restart
    global executor_thread_pool
    if executor_thread_pool._shutdown: # pylint: disable=protected-access
        executor_thread_pool = ThreadPoolExecutor(max_workers=1)
        executor_thread_pool.submit(work_queue_thread)
    # if thread pool is crashed, throw exception
    if executor_thread_pool._broken: # pylint: disable=protected-access
        raise RuntimeError("Thread pool is broken")
    return executor_thread_pool

def submit(func, *args, **kwargs):
    get_thread_pool_executor()
    work_queue.put(lambda: func(*args, **kwargs))
        
def wait_until_finished():
    # wait until all threads are finished
    executor_thread_pool.shutdown(wait=True)

def wrap_sample_images_external_webui(
        prompt_file_path:str,
        output_dir_path:str,
        output_name:str,
        accelerator:Accelerator,
        webui_url:str,
        webui_auth:str=None,
        abs_ckpt_path:str="",
        should_sync:bool=False
    ) -> Tuple[bool, str]:
    """
    Wrapped version of sample_images_external_webui.
    If should_sync is true, it will directly call sample_images_external_webui.
    If should_sync is false, it will submit the function to thread pool executor.
    """
    if should_sync:
        return sample_images_external_webui(
            prompt_file_path,
            output_dir_path,
            output_name,
            accelerator,
            webui_url,
            webui_auth,
            abs_ckpt_path
        )
    else:
        submit(sample_images_external_webui,
            prompt_file_path,
            output_dir_path,
            output_name,
            accelerator,
            webui_url,
            webui_auth,
            abs_ckpt_path
        )
        return True, "Sample request submitted to thread pool executor\n"
    
def sample_images_external_webui(
        prompt_file_path:str,
        output_dir_path:str,
        output_name:str,
        accelerator:Accelerator,
        webui_url:str,
        webui_auth:str=None,
        abs_ckpt_path:str=""
    ) -> Tuple[bool, str]:
    """
    Generate sample with external webui. Returns true if sample request was successful. Returns False if webui was not reachable.
    """
    # prompt file path can be .json file for webui
    if not prompt_file_path.endswith(".json") and not prompt_file_path.endswith(".txt"):
        return False, f"Invalid prompt file path. Must end with .json or .txt, got {prompt_file_path}"
    if not webui_url.endswith('/sdapi/v1'):
        # first split by /, then remove last element, then join back
        if webui_url.endswith('/'):
            webui_url = webui_url[:-1]
        webui_url = webui_url + '/sdapi/v1'
    webui_instance = WebUIApi(baseurl=webui_url)
    if webui_auth and ':' in webui_auth:
        if len(webui_auth.split(':')) != 2:
            return False, f"Invalid webui_auth format. Must be in the form of username:password, got {webui_auth}"
        webui_instance.set_auth(*webui_auth.split(':'))
    ping_response = ping_webui(webui_instance)
    sleep_task(5) # wait for 5 seconds to make sure lora is refreshed
    if ping_response is None:
        return False, f"WebUI at {webui_url} is not reachable"
    # now upload, request samples, and download results, then remove uploaded files
    
    ckpt_name_to_upload = str(uuid.uuid4()) # generate random uuid for checkpoint name
    # the following function calls are thread-blocking so it is called and queued in a thread
    upload_success, message = upload_ckpt(webui_instance, abs_ckpt_path, ckpt_name_to_upload)
    if not upload_success:
        return False, message
    sleep_task(5) # wait for 5 seconds to make sure lora is refreshed
    ckpt_name = os.path.basename(abs_ckpt_path) # get ckpt name from path
    assert_lora(webui_instance, ckpt_name, ckpt_name_to_upload) # assert lora exists
    # remove extension
    refresh_lora(webui_instance) # refresh lora to make sure it is up to date
    sleep_task(5) # wait for 5 seconds to make sure lora is refreshed
    if '.' in ckpt_name:
        ckpt_name = ckpt_name[:ckpt_name.rindex('.')]
    sample_success, msg = request_sample(
        prompt_file_path,
        output_dir_path,
        output_name,
        accelerator,
        webui_instance,
        ckpt_name=ckpt_name
    )
    msg = message + msg
    if not sample_success:
        return False, msg
    sleep_task(5) # wait for 5 seconds to make sure lora is refreshed
    remove_success, msg_remove = remove_ckpt(webui_instance, ckpt_name + '.safetensors', ckpt_name_to_upload)
    msg += msg_remove
    if not remove_success:
        return True, msg # still return true if remove failed
    return True, msg

def upload_ckpt(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str) -> Tuple[bool, str]:
    """
    Upload checkpoint to webui. Returns true if upload was successful. Returns False if webui was not reachable.
    """
    # check if ckpt_name is a valid path
    if not os.path.exists(ckpt_name):
        return False, f"Invalid checkpoint path. File does not exist: {ckpt_name}"
    def upload_thread(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str):
        response = webui_instance.upload_lora(ckpt_name, ckpt_name_to_upload)
        return response
    # submit thread
    response_text = ""
    reponse = upload_thread(webui_instance, ckpt_name, ckpt_name_to_upload)
    response_text = reponse.text if reponse is not None else ""
    return True, response_text

def sleep_task(seconds:int):
    time.sleep(seconds)

def refresh_lora(webui_instance:WebUIApi) -> Tuple[bool, str]:
    """
    Sends refresh request to webui.
    Always return true.
    """
    response_text = ""
    response = webui_instance.refresh_loras()
    response_text = response.text if response is not None else ""
    return True, response_text

def assert_lora(webui_instance:WebUIApi, ckpt_filename:str, subpath:str) -> Tuple[bool, str]:
    """
    Checks if checkpoint exists in webui.
    Returns true if checkpoint exists.
    Throws exception if webui is not reachable or file does not exist.
    """
    def get_query_hash_lora(webui_instance:WebUIApi, subpath:str, filename:str):
        filename_candidate_1 = subpath + '/' + filename
        filename_candidate_2 = subpath + '\\' + filename
        response_json = webui_instance.query_hash_loras()
        hashes_list = response_json['hashes']
        if filename_candidate_1 in hashes_list:
            return True, filename_candidate_1
        elif filename_candidate_2 in hashes_list:
            return True, filename_candidate_2
        else:
            # if executed in thread, crash the thread
            raise RuntimeError(f"File does not exist in webui: {filename}") # may crash ThreadPoolExecutor
    # submit thread
    exists, filename = get_query_hash_lora(webui_instance, subpath, ckpt_filename)
    if not exists:
        raise RuntimeError(f"File does not exist in webui: {filename}")
    return True, filename
        

def remove_ckpt(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str) -> Tuple[bool, str]:
    """
    Remove checkpoint from webui. Returns true if removal was successful. Returns False if webui was not reachable.
    """
    # submit thread
    response_text = ""
    response = webui_instance.remove_lora_model(f"{ckpt_name_to_upload}/{ckpt_name}")
    response_text = response.text if response is not None else ""  
    return True, response_text

def ping_webui(webui_instance:WebUIApi) -> bool:
    """
    Ping webui to check if it is reachable, and uploader is alive.
    WebUI requires webui-model-uploader, Agent scheduler, and required extensions to be running.
    """
    try:
        webui_instance.check_uploader_ping()
        return True
    except RuntimeError:
        return False
    
def log_wandb(
        accelerator:Accelerator,
        image:np.ndarray,
        prompt:str,
        negative_prompt:str,
        seed:int,
    ):
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        try:
            import wandb
        except ImportError: 
            raise ImportError("No wandb / wandb がインストールされていないようです")
        # log generation information to wandb
        logging_caption_key = f"prompt : {prompt} seed: {str(seed)}"
        # remove invalid characters from the caption for filenames
        logging_caption_key = re.sub(r"[^a-zA-Z0-9_\-. ]+", "", logging_caption_key)
        wandb_tracker.log(
            {
                logging_caption_key: wandb.Image(image, caption=f"prompt: {prompt} negative_prompt: {negative_prompt}"),
            }
        )
    except:  # wandb 無効時 # pylint: disable=bare-except
        pass

def parse_text(text_line:str) -> Dict[str, Any]:
    """
    Parse text line to json.
    see Available Options (--n, etc)
    
    example:
    1girl --n worst quality --d 42 --s 25 --w 512 --h 768 --l 7.5
    -> {
        'prompt' : '1girl',
        'negative_prompt' : 'worst quality',
        'seed' : 42,
        'steps' : 25,
        'width' : 512,
        'height' : 768,
        'cfg_scale' : 7.5
    }
    
    if arguments are not specified or found, it will be ignored.
    """
    # Split the text line into prompt and arguments
    prompt, *args = text_line.split('--')
    prompt = prompt.strip().split(',')
    
    # Initialize an empty dictionary to store parsed arguments
    parsed_args_dict = {}
    
    # Manually parse the arguments
    for arg in args:
        key, value = arg.strip().split(' ', 1)
        if key == 'n':
            parsed_args_dict['negative_prompt'] = value
        elif key == 'd':
            parsed_args_dict['seed'] = int(value)
        elif key == 's':
            parsed_args_dict['steps'] = int(value)
        elif key == 'w':
            parsed_args_dict['width'] = int(value)
        elif key == 'h':
            parsed_args_dict['height'] = int(value)
        elif key == 'l':
            parsed_args_dict['cfg_scale'] = float(value)
    
    # Add the prompt to the dictionary
    parsed_args_dict['prompt'] = ', '.join(prompt)
    return parsed_args_dict
    
def handle_txt_prompt(prompt_file_path:str):
    """
    Handle txt prompt file. Returns list of prompts json that can be used for webuiapi.
    Text file should not contain lora regex, it will be added automatically.
    Other lora does not matters.
    Available options :
        --n : negative_prompt
        --d : seed
        --s : step
        --w : width
        --h : height
        --l : cfg scale
    """
    if not prompt_file_path.endswith(".txt"):
        raise ValueError(f"Invalid prompt file path. Must end with .txt, got {prompt_file_path}")
    elif not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_file_path}")
    prompts = []
    # prompt dict : {'prompt':prompt_with_regex_added, 'negative_prompt':negative_prompt, 'seed':seed, 'steps':step, 'width':width, 'height':height, 'cfg_scale':cfg_scale}
    # prompt_with_regex_added : prompt + <lora:{lora_name_2}:1>
    # regex_to_replace : {lora_name_2}
    with open(prompt_file_path, 'r', encoding='utf-8') as prompt_file:
        for line in prompt_file.readlines():
            line = line.strip()
            if not line:
                continue
            prompt_dict = parse_text(line) # parse text to json
            assert isinstance(prompt_dict, dict), f"Invalid prompt format. Must be a dict, got {prompt_dict}"
            prompt_dict['regex_to_replace'] = "{lora_name_2}"
            prompt_dict["prompt"] = prompt_dict["prompt"] + " <lora:{lora_name_2}:1>"
            prompts.append(prompt_dict)
    return prompts

def request_sample(
        prompt_file_path:str,
        output_dir_path:str,
        output_name:str,
        accelerator:Accelerator,
        webui_instance:WebUIApi,
        ckpt_name:str=""
    ) -> Tuple[bool, str]:
    """
    Generate sample with external webui. This function is thread-locking. 
    Prompt file should be a json(or txt) file that can be parsed for webuiapi.
    real 'prompts' and 'negative prompt' should contain regex_to_replace, which will be replaced with ckpt_name.
    
    example prompt:
        "1girl, <lora:$1>"
    with example regex_to_replace $1
    will be converted to
        "1girl, <lora:ckpt_name>"
    """
    if prompt_file_path.endswith(".json"):
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    elif prompt_file_path.endswith(".txt"):
        prompts = handle_txt_prompt(prompt_file_path)
    if not isinstance(prompts, list):
        prompts = [prompts] # convert to list if not list
    os.makedirs(output_dir_path, exist_ok=True)
    message = ""
    any_success = False
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            message += f"Invalid prompt format. Must be a dict, got {prompt}\n"
            continue
        # controlnet params
        controlnet_params = ["controlnet_model", "controlnet_image", "controlnet_preprocessor"]
        # check controlnet params
        if prompt.get("controlnet_model", None) is not None:
            controlnet_model = prompt["controlnet_model"]
            controlnet_image = prompt.get("controlnet_image", None)
            if controlnet_image is None:
                message += f"Invalid prompt format. Must include controlnet_image when using controlnet_model, got {prompt}\n"
                continue
            controlnet_preprocessor = prompt.get("controlnet_preprocessor", "none")
            if not os.path.exists(controlnet_image):
                message += f"Invalid controlnet_image path. File does not exist: {controlnet_image}\n"
                continue
            controlnet_unit = ControlNetUnit(input_image=open_controlnet_image(controlnet_image), module=controlnet_preprocessor, model=controlnet_model)
            controlnet_units = [controlnet_unit]
        else:
            controlnet_units = []
            
        if "regex_to_replace" not in prompt:
            message += f"Invalid prompt format. Must include regex_to_replace, got {prompt}\n"
            continue
        # replace regex_to_replace with ckpt_name, search in prompt and negative prompt
        if "prompt" not in prompt:
            message += f"Invalid prompt format. Must include prompt, got {prompt}\n"
            continue
        orig_prompt:str = prompt["prompt"]
        prompt["prompt"] = orig_prompt.replace(prompt["regex_to_replace"], ckpt_name)
        if "negative_prompt" in prompt: # well this is optional but suggested
            prompt["negative_prompt"] = prompt["negative_prompt"].replace(prompt["regex_to_replace"], ckpt_name)
        # pop regex_to_replace
        prompt.pop("regex_to_replace")
        # pop controlnet params
        for param in controlnet_params:
            if param in prompt:
                prompt.pop(param)
        # if seed is not specified, generate random seed
        if "seed" not in prompt or prompt["seed"] == -1:
            prompt["seed"] = np.random.randint(0, 100000)
        # fix alwayson_script args if there is any 'file' references as strings
        if 'alwayson_scripts' in prompt:
            alwayson_scripts_dict = prompt['alwayson_scripts']
            for key, dictvalue in alwayson_scripts_dict.items():
                for _key, value in dictvalue.items():
                    # value is list of strings
                    for idx, elem in enumerate(value):
                        if isinstance(elem, str) and os.path.isfile(elem):
                            # convert to base64
                            base64_str = raw_b64_img(open_mask_image(elem))
                            alwayson_scripts_dict[key][_key][idx] = base64_str
                            message += f"Converted {elem} to base64 string\n"
        positive_prompt, negative_prompt, seed = prompt.get("prompt", "positive:None"), prompt.get("negative_prompt", "negative:None"), prompt.get("seed", 0)
        queued_task_result = webui_instance.txt2img_task(
            controlnet_units=controlnet_units,
            **prompt
        )
        # start thread to wait for result
        def wait_and_save(queued_task_result:QueuedTaskResult, output_dir_path, output_name, accelerator, orig_prompt, negative_prompt, seed):
            while not queued_task_result.is_finished(): # can throw exception if webui is not reachable or broken
                time.sleep(5) # wait 5 seconds before checking again
            # 6 digits of time
            strftime = f"{time.strftime('%Y%m%d_%H%M%S')}"
            image = queued_task_result.get_image()
            if image is None:
                raise RuntimeError(f"Image is None while waiting for result, task id: {queued_task_result.task_id}")
            image.save(os.path.join(output_dir_path, f"{output_name}_{strftime}_{i}.png"))
            log_wandb(accelerator, image, orig_prompt, negative_prompt, seed)
        wait_and_save(queued_task_result, output_dir_path, output_name, accelerator, orig_prompt, negative_prompt, seed)
        any_success = True
    if not any_success:
        return False, "No valid prompts found\n" + message
    return True, message
        