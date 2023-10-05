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
from webuiapi.webuiapi import WebUIApi, ControlNetUnit, QueuedTaskResult
from webuiapi.test_utils import InstanceHolder, open_controlnet_image, open_mask_image, raw_b64_img

executor_thread_pool = ThreadPoolExecutor(max_workers=1) # sequential execution
    
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
        ckpt_name:str=""
    ) -> bool:
    """
    Generate sample with external webui. Returns true if sample request was successful. Returns False if webui was not reachable.
    """
    # prompt file path can be .json file for webui
    if not prompt_file_path.endswith(".json"):
        print(f"Invalid prompt file path. Must end with .json, got {prompt_file_path}")
        return False
    webui_instance = WebUIApi(webui_url)
    if webui_auth and ':' in webui_auth:
        if len(webui_auth.split(':')) != 2:
            print(f"Invalid webui_auth format. Must be in the form of username:password, got {webui_auth}")
            return False
        webui_instance.set_auth(*webui_auth.split(':'))
    ping_response = ping_webui(webui_url)
    if ping_response is None:
        print(f"WebUI at {webui_url} is not reachable")
        return False
    # now upload, request samples, and download results, then remove uploaded files
    
    ckpt_name_to_upload = str(uuid.uuid4()) # generate random uuid for checkpoint name
    # the following function calls are thread-blocking so it is called and queued in a thread
    upload_ckpt(webui_instance, ckpt_name, ckpt_name_to_upload)
    request_sample(
        prompt_file_path,
        output_dir_path,
        output_name,
        accelerator,
        webui_instance,
        ckpt_name=ckpt_name
    )
    remove_ckpt(webui_instance, ckpt_name)
    pass # TODO

def upload_ckpt(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str):
    """
    Upload checkpoint to webui. Returns true if upload was successful. Returns False if webui was not reachable.
    """
    response = webui_instance.upload_lora(ckpt_name, ckpt_name_to_upload) # uploaded to ckpt_name_to_upload/ckpt_name
    if response is None or response.status_code != 200:
        print(f"Failed to upload checkpoint {ckpt_name} to webui")
        return False
    return True

def remove_ckpt(webui_instance:WebUIApi, ckpt_name:str, ckpt_name_to_upload:str):
    """
    Remove checkpoint from webui. Returns true if removal was successful. Returns False if webui was not reachable.
    """
    response = webui_instance.remove_lora_model(f"{ckpt_name_to_upload}/{ckpt_name}")
    if response is None or response.status_code != 200:
        print(f"Failed to remove checkpoint {ckpt_name} from webui")
        return False
    return True


    
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
                logging_caption_key: wandb.Image(image, caption=f"negative_prompt: {negative_prompt}"),
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
        ckpt_name:str=""
    ):
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
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            print(f"Invalid prompt format. Must be a dict, got {prompt}")
            continue
        # controlnet params
        controlnet_params = ["controlnet_model", "controlnet_image", "controlnet_preprocessor"]
        # check controlnet params
        if prompt.get("controlnet_model", None) is not None:
            controlnet_model = prompt["controlnet_model"]
            controlnet_image = prompt.get("controlnet_image", None)
            if controlnet_image is None:
                print(f"Invalid prompt format. Must include controlnet_image when using controlnet_model, got {prompt}")
                continue
            controlnet_preprocessor = prompt.get("controlnet_preprocessor", "none")
            if not os.path.exists(controlnet_image):
                print(f"Invalid controlnet_image path. File does not exist: {controlnet_image}")
                continue
            controlnet_unit = ControlNetUnit(input_image=open_controlnet_image(controlnet_image), module=controlnet_preprocessor, model=controlnet_model)
            controlnet_units = [controlnet_unit]
        else:
            controlnet_units = []
            
        if "regex_to_replace" not in prompt:
            print(f"Invalid prompt format. Must include regex_to_replace, got {prompt}")
            continue
        # replace regex_to_replace with ckpt_name, search in prompt and negative prompt
        prompt["prompt"] = prompt["prompt"].replace(prompt["regex_to_replace"], ckpt_name)
        prompt["negative_prompt"] = prompt["negative_prompt"].replace(prompt["regex_to_replace"], ckpt_name)
        # pop regex_to_replace
        prompt.pop("regex_to_replace")
        # pop controlnet params
        for param in controlnet_params:
            if param in prompt:
                prompt.pop(param)
        # if seed is not specified, generate random seed
        if "seed" not in prompt or prompt["seed"] is -1:
            prompt["seed"] = np.random.randint(0, 100000)
        # fix alwayson_script args if there is any 'file' references as strings
        if 'alwayson_scripts' in prompt:
            alwayson_scripts_dict = prompt['alwayson_scripts']
            for key, value in alwayson_scripts_dict.items():
                if isinstance(value, str) and os.path.isfile(value):
                    # convert to base64
                    base64_str = raw_b64_img(open_mask_image(value))
                    alwayson_scripts_dict[key] = base64_str
                    print(f"Converted {value} to base64 string")
        positive_prompt, negative_prompt, seed = prompt.get("positive_prompt", "positive:None"), prompt.get("negative_prompt", "negative:None"), prompt.get("seed", 0)
        queued_task_result = webui_instance.txt2img_task(
            controlnet_units=controlnet_units,
            **prompt
        )
        # start thread to wait for result
        def wait_and_save(queued_task_result:QueuedTaskResult, output_dir_path, output_name, accelerator):
            while not queued_task_result.check_finished(): # can throw exception if webui is not reachable or broken
                time.sleep(1)
            # 6 digits of time
            strftime = f"{time.strftime('%Y%m%d_%H%M%S')}"
            result = queued_task_result.get_image()
            if result is None:
                print("Failed to generate sample")
                return
            image = result.image
            image.save(os.path.join(output_dir_path, f"{output_name}_{strftime}_{i}.png"))
            log_wandb(accelerator, image, positive_prompt, negative_prompt, seed)
        # start thread
        executor_thread_pool.submit(wait_and_save, queued_task_result, output_dir_path, output_name, accelerator)