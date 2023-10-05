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
from webuiapi.test_utils import InstanceHolder, open_controlnet_image

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
    """
    instance_holder = InstanceHolder(webui_instance)
    if prompt_file_path.endswith(".json"):
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    if not isinstance(prompts, list):
        prompts = [prompts] # convert to list if not list
    os.makedirs(output_dir_path, exist_ok=True)
    for i, prompt in enumerate(prompts):
        # check if prompt is a dict
        if isinstance(prompt, dict):
            negative_prompt = prompt.get("negative_prompt", None)
            sample_steps = prompt.get("sample_steps", 30)
            width = prompt.get("width", 512)
            height = prompt.get("height", 512)
            scale = prompt.get("scale", 7.5)
            seed = prompt.get("seed", -1)
            controlnet_image = prompt.get("controlnet_image", None)
            rp_mode = prompt.get("rp_mode", None)
            positive_prompt = prompt.get("prompt", None)
            checkpoint_name = prompt.get("checkpoint_name", None)
            enable_hr = prompt.get("enable_hr", False)
            hr_scale = prompt.get("hr_scale", 1.0)
            hr_model = prompt.get("hr_model", 'R-ESRGAN 4x+ Anime6B')
            hr_noise = prompt.get("hr_noise", 0.5)
            rp_mode_params = None
            if rp_mode is not None:
                # get rp_mode params
                prompt_1_arg: str = prompt.get("prompt_1_arg", "") # necessary
                prompt_2_arg: str = prompt.get("prompt_2_arg", "") # necessary
                attach_lora_to_which: int = prompt.get("attach_lora_to_which", 1)
                lora_stop_steps: Tuple[int, int] = prompt.get("lora_stop_steps", (0, 0))
                extra_args: dict = prompt.get("extra_args", {})
                controlnet_path: str = prompt.get("controlnet_path", "")
                controlnet_preprocessor: str = prompt.get("controlnet_preprocessor", "none") 
                controlnet_model: str = prompt.get("controlnet_model", "none")
                base_prompt: str = prompt.get("base_prompt", "") 
                common_prompt: str = prompt.get("common_prompt", "") 
                negative_prompt: str = prompt.get("negative_prompt", "")
                regional_prompter_args: dict | None = prompt.get("regional_prompter_args", None)
                # check if necessary params are present
                if prompt_1_arg == "" or prompt_2_arg == "":
                    print(f"Invalid prompt format. Must include prompt_1_arg, prompt_2_arg, controlnet_path, controlnet_model when using rp_mode, got {prompt}")
                    continue
                # wrap as dict
                rp_mode_params = {
                    "prompt_1_arg": prompt_1_arg + f"<lora:{ckpt_name}:1>" if attach_lora_to_which == 0 else prompt_1_arg,
                    "prompt_2_arg": prompt_2_arg + f"<lora:{ckpt_name}:1>" if attach_lora_to_which == 1 else prompt_2_arg,
                    "lora_stop_steps": lora_stop_steps,
                    "extra_args": extra_args,
                    "controlnet_path": controlnet_path,
                    "controlnet_preprocessor": controlnet_preprocessor,
                    "controlnet_model": controlnet_model,
                    "base_prompt": base_prompt,
                    "common_prompt": common_prompt,
                    "negative_prompt": negative_prompt,
                    "regional_prompter_args": regional_prompter_args,
                }
                # reconstruct positive_prompt 
                positive_prompt = f"{common_prompt} {prompt_1_arg} BREAK {common_prompt} {prompt_2_arg}"
        else:
            print(f"Invalid prompt format. Must be a dict, got {prompt}")
            continue
        # check controlnet params
        if prompt.get("controlnet_model", None) is not None:
            controlnet_model = prompt["controlnet_model"]
            controlnet_image = prompt.get("controlnet_image", None)
            if controlnet_image is None:
                print(f"Invalid prompt format. Must include controlnet_image when using controlnet_model, got {prompt}")
                continue
            controlnet_preprocessor = prompt.get("controlnet_preprocessor", None)
            if not os.path.exists(controlnet_image):
                print(f"Invalid controlnet_image path. File does not exist: {controlnet_image}")
                continue
            controlnet_unit = ControlNetUnit(input_image=open_controlnet_image(controlnet_image), module=controlnet_preprocessor, model=controlnet_model)
            controlnet_units = [controlnet_unit]
        else:
            controlnet_units = []
        
        queued_task_result = None
        if rp_mode is None: # basic test
            queued_task_result = instance_holder.instance.txt2img_task(
                controlnet_units=controlnet_units,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                cfg_scale=scale,
                seed=seed,
                steps=sample_steps,
                checkpoint_name=checkpoint_name, # can be None
                enable_hr=enable_hr,
                hr_scale=hr_scale,
                hr_noise=hr_noise,
                hr_upscaler=hr_model,
            )
        elif rp_mode == "division": # divide mode
            queued_task_result = instance_holder.test_division(
                controlnet_model=controlnet_model,
                controlnet_path=controlnet_image,
                controlnet_preprocessor=controlnet_preprocessor,
                **rp_mode_params
            )
        elif rp_mode == 'mask':
            queued_task_result = instance_holder.test_mask(
                controlnet_model=controlnet_model,
                controlnet_path=controlnet_image,
                controlnet_preprocessor=controlnet_preprocessor,
                **rp_mode_params
            )
        else:
            print(f"Invalid rp_mode. Must be division or mask, got {rp_mode}")
            continue
        # start thread to wait for result
        def wait_and_save(queued_task_result:QueuedTaskResult, output_dir_path, output_name, accelerator):
            while not queued_task_result.check_finished():
                time.sleep(1)
            result = queued_task_result.get_image()
            if result is None:
                print("Failed to generate sample")
                return
            image = result.image
            image.save(os.path.join(output_dir_path, f"{output_name}_{i}.png"))
            log_wandb(accelerator, image, positive_prompt, negative_prompt, seed)
        # start thread
        executor_thread_pool.submit(wait_and_save, queued_task_result, output_dir_path, output_name, accelerator)