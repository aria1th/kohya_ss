"""
Util class to handle external WebUI calls
"""
import json
import os
import time
from typing import Tuple
import re
import uuid
import numpy as np
from accelerate import Accelerator
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from library.webuiapi.webuiapi import WebUIApi, ControlNetUnit, QueuedTaskResult
from library.webuiapi.test_utils import open_controlnet_image, open_mask_image, raw_b64_img

executor_thread_pool = ThreadPoolExecutor(max_workers=1) # sequential execution
executor_thread_pool.submit(lambda: None) # start thread
    
def get_thread_pool_executor() -> ThreadPoolExecutor:
    # check if thread pool is shutdown, if so, restart
    global executor_thread_pool
    if executor_thread_pool._shutdown: # pylint: disable=protected-access
        executor_thread_pool = ThreadPoolExecutor(max_workers=1)
        executor_thread_pool.submit(lambda: None)
    return executor_thread_pool
        
def wait_until_finished():
    # wait until all threads are finished
    executor_thread_pool.shutdown(wait=True)
    
def sample_images_external_webui(
        prompt_file_path:str,
        output_dir_path:str,
        output_name:str,
        accelerator:Accelerator,
        webui_url:str,
        webui_auth:str=None,
        abs_ckpt_path:str="",
        should_sync:bool = False
    ) -> Tuple[bool, str]:
    """
    Generate sample with external webui. Returns true if sample request was successful. Returns False if webui was not reachable.
    """
    # prompt file path can be .json file for webui
    if not prompt_file_path.endswith(".json"):
        return False, f"Invalid prompt file path. Must end with .json, got {prompt_file_path}"
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
    if ping_response is None:
        return False, f"WebUI at {webui_url} is not reachable"
    # now upload, request samples, and download results, then remove uploaded files
    
    ckpt_name_to_upload = str(uuid.uuid4()) # generate random uuid for checkpoint name
    # the following function calls are thread-blocking so it is called and queued in a thread
    upload_success, message = upload_ckpt(webui_instance, abs_ckpt_path, ckpt_name_to_upload, should_sync=should_sync)
    if not upload_success:
        return False, message
    ckpt_name = os.path.basename(abs_ckpt_path) # get ckpt name from path
    # remove extension
    if '.' in ckpt_name:
        ckpt_name = ckpt_name[:ckpt_name.rindex('.')]
    sample_success, msg = request_sample(
        prompt_file_path,
        output_dir_path,
        output_name,
        accelerator,
        webui_instance,
        ckpt_name=ckpt_name,
        should_sync=should_sync
    )
    msg = message + msg
    if not sample_success:
        return False, msg
    remove_success, msg_remove = remove_ckpt(webui_instance, ckpt_name + '.safetensors', ckpt_name_to_upload, should_sync=should_sync)
    msg += msg_remove
    if not remove_success:
        return True, msg # still return true if remove failed
    return True, msg

def upload_ckpt(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str, should_sync:bool = False) -> Tuple[bool, str]:
    """
    Upload checkpoint to webui. Returns true if upload was successful. Returns False if webui was not reachable.
    """
    # check if ckpt_name is a valid path
    if not os.path.exists(ckpt_name):
        return False, f"Invalid checkpoint path. File does not exist: {ckpt_name}"
    def upload_thread(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str):
        response = webui_instance.upload_lora(ckpt_name, ckpt_name_to_upload)
    # submit thread
    get_thread_pool_executor().submit(upload_thread, webui_instance, ckpt_name, ckpt_name_to_upload)
    if should_sync:
        # wait until job is done and executor is idle
        get_thread_pool_executor().shutdown(wait=True)
    return True, ""

def remove_ckpt(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str, should_sync:bool = False) -> Tuple[bool, str]:
    """
    Remove checkpoint from webui. Returns true if removal was successful. Returns False if webui was not reachable.
    """
    def remove_thread(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str):
        response = webui_instance.remove_lora_model(f"{ckpt_name_to_upload}/{ckpt_name}")
    # submit thread
    get_thread_pool_executor().submit(remove_thread, webui_instance, ckpt_name, ckpt_name_to_upload)
    if should_sync:
        # wait until job is done and executor is idle
        get_thread_pool_executor().shutdown(wait=True)
    return True, ""


    
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
    except:  # wandb 無効時
        pass

def request_sample(
        prompt_file_path:str,
        output_dir_path:str,
        output_name:str,
        accelerator:Accelerator,
        webui_instance:WebUIApi,
        ckpt_name:str="",
        should_sync:bool = False
    ) -> Tuple[bool, str]:
    """
    Generate sample with external webui. This function is thread-locking. 
    Prompt file should be a json file that can be parsed for webuiapi.
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
        prompt["prompt"] = prompt["prompt"].replace(prompt["regex_to_replace"], ckpt_name)
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
        def wait_and_save(queued_task_result:QueuedTaskResult, output_dir_path, output_name, accelerator):
            while not queued_task_result.is_finished(): # can throw exception if webui is not reachable or broken
                time.sleep(1)
            # 6 digits of time
            strftime = f"{time.strftime('%Y%m%d_%H%M%S')}"
            image = queued_task_result.get_image()
            if image is None:
                print("Failed to generate sample")
                return
            image.save(os.path.join(output_dir_path, f"{output_name}_{strftime}_{i}.png"))
            log_wandb(accelerator, image, positive_prompt, negative_prompt, seed)
        # start thread
        
        
        if should_sync:
            # wait until job is done and executor is idle
            get_thread_pool_executor().shutdown(wait=True) # wait until previous job is done
            # here directly execute the function
            wait_and_save(queued_task_result, output_dir_path, output_name, accelerator)
        else:
            get_thread_pool_executor().submit(wait_and_save, queued_task_result, output_dir_path, output_name, accelerator)
        any_success = True
    if not any_success:
        return False, "No valid prompts found\n" + message
    return True, message
        