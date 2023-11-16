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
from concurrent.futures import ThreadPoolExecutor, Future
from library.webuiapi.webuiapi import WebUIApi, ControlNetUnit, QueuedTaskResult
from library.webuiapi.test_utils import open_controlnet_image, open_mask_image, raw_b64_img

executor_thread_pool = ThreadPoolExecutor(max_workers=1) # sequential execution
executor_thread_pool.submit(lambda: None) # submit dummy task to start thread

any_error_occurred = False # this is used to check if any error occurred in thread pool executor
error_obj = None # this is used to store error object if any error occurred in thread pool executor

run_id = None # this is used to store run id for wandb

jobs = {} # this is used to store jobs in thread pool executor
futures : Dict[int, Future] = {} # this is used to store futures in thread pool executor
jobs_explanation = {} # this is used to store jobs explanation in thread pool executor
job_idx = 0 # this is used to store job index in thread pool executor

recent_messages = ["None"] # threads will append to this list

def check_webui_state(): #throws exception if any error occurred in thread pool executor
    global any_error_occurred
    global error_obj
    if any_error_occurred:
        raise error_obj

def get_thread_pool_executor() -> ThreadPoolExecutor:
    # check if thread pool is shutdown, if so, restart
    global executor_thread_pool
    if executor_thread_pool._shutdown: # pylint: disable=protected-access
        print("Thread pool is shutdown, restarting...")
        executor_thread_pool = ThreadPoolExecutor(max_workers=1)
        executor_thread_pool.submit(lambda: None)
    # if thread pool is crashed, throw exception
    if executor_thread_pool._broken: # pylint: disable=protected-access
        raise RuntimeError("Thread pool is broken")
    return executor_thread_pool

def submit(func, job_name:str = "", *args, **kwargs):
    assert func is not None, "func cannot be None"
    global jobs, job_idx
    jobs[job_idx] = False # mark job as not finished
    jobs_explanation[job_idx] = job_name
    def wrap_func_with_job(func):
        def wrapped_func(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                jobs[job_idx] = True
                return result
            except Exception as e:
                global any_error_occurred
                global error_obj
                any_error_occurred = True
                error_obj = e
            finally:
                jobs[job_idx] = True
        return wrapped_func
    job_idx += 1
    future = get_thread_pool_executor().submit(wrap_func_with_job(func), *args, **kwargs)
    futures[job_idx] = future

def log_recent_message(message:str):
    global recent_messages
    recent_messages.append(message)
    if len(recent_messages) > 10:
        recent_messages = recent_messages[-10:]

def print_jobs():
    for idx, job in jobs.items():
        if not job:
            print(f"Job {idx} : {jobs_explanation[idx]}")
    print(f"latest messages: {recent_messages[-1]}")
def wait_until_finished():
    if executor_thread_pool._shutdown: # pylint: disable=protected-access
        return
    if executor_thread_pool._broken: # pylint: disable=protected-access
        return # do nothing if thread pool is broken
    global jobs
    if not all(jobs.values()):
        print(f"Waiting for jobs to finish... {len(jobs)} jobs left")
        print_jobs()
        for idx, future in futures.items():
            # wait for future to finish
            future.result()
    # wait until all threads are finished
    # future.result() will throw exception if any error occurred in thread pool executor
    executor_thread_pool.shutdown(wait=True)

def wrap_sample_images_external_webui(
        prompt_file_path:str,
        output_dir_path:str,
        output_name:str,
        accelerator:Accelerator,
        webui_url:str,
        webui_auth:str=None,
        abs_ckpt_path:str="",
        should_sync:bool=False,
        steps:int=0,
    ) -> Tuple[bool, str]:
    """
    Wrapped version of sample_images_external_webui.
    If should_sync is true, it will directly call sample_images_external_webui.
    If should_sync is false, it will submit the function to thread pool executor.
    """
    if any_error_occurred:
        raise error_obj
    if should_sync:
        return sample_images_external_webui(
            prompt_file_path,
            output_dir_path,
            output_name,
            accelerator,
            webui_url,
            webui_auth,
            abs_ckpt_path,
            steps=steps
        )
    else:
        submit(sample_images_external_webui,
            job_name=f"Sampling images at {steps} steps",
            prompt_file_path = prompt_file_path,
            output_dir_path = output_dir_path,
            output_name = output_name,
            accelerator = accelerator,
            webui_url = webui_url,
            webui_auth = webui_auth,
            abs_ckpt_path =abs_ckpt_path,
            steps=steps,
        )
        return True, "Sample request submitted to thread pool executor\n"

def check_ping_webui(webui_url:str, webui_auth:str=None):
    if not webui_url.endswith('/sdapi/v1'):
        # first split by /, then remove last element, then join back
        if webui_url.endswith('/'):
            webui_url = webui_url[:-1]
        webui_url = webui_url + '/sdapi/v1'
    webui_instance = WebUIApi(baseurl=webui_url)
    if webui_auth and ':' in webui_auth:
        if len(webui_auth.split(':')) != 2:
            raise ValueError(f"Invalid webui_auth format. Must be in the form of username:password, got {webui_auth}")
        webui_instance.set_auth(*webui_auth.split(':'))
    ping_response = ping_webui(webui_instance)
    if ping_response is None or ping_response is False:
        raise RuntimeError(f"WebUI at {webui_url} is not reachable")
    

def sample_images_external_webui(
        prompt_file_path:str,
        output_dir_path:str,
        output_name:str,
        accelerator:Accelerator,
        webui_url:str,
        webui_auth:str=None,
        abs_ckpt_path:str="",
        steps:int=0,
    ) -> Tuple[bool, str]:
    """
    Generate sample with external webui. Returns true if sample request was successful. Returns False if webui was not reachable.
    """
    # timestamp, used for checkpoint name
    timestamp = time.strftime('%Y%m%d%H%M%S')
    file_name = os.path.basename(abs_ckpt_path)
    file_without_ext, ext = os.path.splitext(file_name)
    filename_with_timestamp = f"{file_without_ext}_{timestamp}{ext}"
    # prompt file path can be .json file for webui
    if not prompt_file_path.endswith(".json") and not prompt_file_path.endswith(".txt"):
        accelerator.print(f"Invalid prompt file path. Must end with .json or .txt, got {prompt_file_path}")
        return False, f"Invalid prompt file path. Must end with .json or .txt, got {prompt_file_path}"
    if not webui_url.endswith('/sdapi/v1'):
        # first split by /, then remove last element, then join back
        if webui_url.endswith('/'):
            webui_url = webui_url[:-1]
        webui_url = webui_url + '/sdapi/v1'
    webui_instance = WebUIApi(baseurl=webui_url)
    if webui_auth and ':' in webui_auth:
        if len(webui_auth.split(':')) != 2:
            accelerator.print(f"Invalid webui_auth format. Must be in the form of username:password, got {webui_auth}")
            return False, f"Invalid webui_auth format. Must be in the form of username:password, got {webui_auth}"
        webui_instance.set_auth(*webui_auth.split(':'))
    ping_response = ping_webui(webui_instance)
    log_recent_message(f"WebUI at {webui_url} is reachable: {ping_response}")
    if ping_response is None:
        accelerator.print(f"WebUI at {webui_url} is not reachable")
        return False, f"WebUI at {webui_url} is not reachable"
    # now upload, request samples, and download results, then remove uploaded files
    
    subdir = str(uuid.uuid4()) # generate random uuid for checkpoint name
    # the following function calls are thread-blocking so it is called and queued in a thread
    log_recent_message(f"Uploading checkpoint to webui...")
    upload_success, message = upload_ckpt(webui_instance, abs_ckpt_path, subdir, custom_name=filename_with_timestamp)
    if not upload_success:
        accelerator.print(message)
        return False, message

    ckpt_name = filename_with_timestamp # checkpoint name
    log_recent_message(f"Asserting checkpoint exists in webui...")
    _true, message_assertion = assert_lora(webui_instance, filename_with_timestamp, subdir) # assert lora exists
    # remove extension
    log_recent_message(f"Refreshing lora...")
    refresh_lora(webui_instance) # refresh lora to make sure it is up to date
    sleep_task(3) # wait for 5 seconds to make sure lora is refreshed
    if '.' in ckpt_name:
        ckpt_name = ckpt_name[:ckpt_name.rindex('.')]
    log_recent_message(f"Requesting sample...")
    sample_success, msg = request_sample(
        prompt_file_path,
        output_dir_path,
        output_name,
        accelerator,
        webui_instance,
        ckpt_name=ckpt_name,
        steps=steps
    )
    msg = message + msg
    if not sample_success:
        accelerator.print(msg)
        return False, msg
    log_recent_message(f"Removing checkpoint from webui...")
    remove_success, msg_remove = remove_ckpt(webui_instance, ckpt_name + '.safetensors', subdir)
    msg += msg_remove
    if not remove_success:
        accelerator.print(msg)
        return True, msg # still return true if remove failed
    return True, msg

def upload_ckpt(webui_instance:WebUIApi, ckpt_name:str, subdir:str, custom_name:str="") -> Tuple[bool, str]:
    """
    Upload checkpoint to webui. Returns true if upload was successful. Returns False if webui was not reachable.
    """
    # check if ckpt_name is a valid path
    if not os.path.exists(ckpt_name):
        return False, f"Invalid checkpoint path. File does not exist: {ckpt_name}"
    response = webui_instance.upload_lora(ckpt_name, subdir, custom_name=custom_name)
    assert response is not None, f"WebUI at {webui_instance.baseurl} is not reachable"
    assert response.status_code == 200, f"WebUI at {webui_instance.baseurl} returned status code {response.status_code}, response: {response.text}"
    assert response.json() is not None, f"WebUI at {webui_instance.baseurl} returned invalid json, response: {response.text}"
    assert response.json().get('success', False), f"WebUI at {webui_instance.baseurl} returned success false, response: {response.text}"
    response_text = response.text if response is not None else ""
    return response.json().get('success', False), response_text

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
            any_error_occurred = True
            error_obj = RuntimeError(f"File does not exist in webui: {filename}")
            raise RuntimeError(f"File does not exist in webui: {filename}") # may crash ThreadPoolExecutor
    # submit thread
    exists, filename = get_query_hash_lora(webui_instance, subpath, ckpt_filename)
    if not exists:
        any_error_occurred = True
        error_obj = RuntimeError(f"File does not exist in webui: {filename}")
        raise RuntimeError(f"File does not exist in webui: {filename}")
    return True, filename
        

def remove_ckpt(webui_instance:WebUIApi, ckpt_name:str, subdir:str) -> Tuple[bool, str]:
    """
    Remove checkpoint from webui. Returns true if removal was successful. Returns False if webui was not reachable.
    """
    # submit thread
    response_text = ""
    response = webui_instance.remove_lora_model(f"{subdir}/{ckpt_name}")
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
        steps:int=0,
        index:int=0,
    ):
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        try:
            import wandb
        except ImportError: 
            raise ImportError("No wandb / wandb がインストールされていないようです")
        # log generation information to wandb
        global run_id
        if run_id is None:
            run_id = wandb_tracker.run.id # get run id
        # resume if stopped
        # wandb.init(id=run_id, resume="must")
        logging_caption_key = f"image_{index}"
        # remove invalid characters from the caption for filenames
        logging_caption_key = re.sub(r"[^a-zA-Z0-9_\-. ]+", "", logging_caption_key)
        wandb_tracker.log(
            {
                'custom_step' : steps,
                logging_caption_key: wandb.Image(image, caption=f"prompt: {prompt} negative_prompt: {negative_prompt} seed: {seed}"),
            },
            commit=False
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
        ckpt_name:str="",
        steps:int=0,
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
                any_error_occurred = True
                error_obj = RuntimeError(f"Image is None while waiting for result, task id: {queued_task_result.task_id}")
                raise RuntimeError(f"Image is None while waiting for result, task id: {queued_task_result.task_id}")
            image.save(os.path.join(output_dir_path, f"{output_name}_{strftime}_{i}.png"))
            log_wandb(accelerator, image, orig_prompt, negative_prompt, seed, steps=steps, index=i)
        wait_and_save(queued_task_result, output_dir_path, output_name, accelerator, orig_prompt, negative_prompt, seed)
        any_success = True
    if not any_success:
        return False, "No valid prompts found\n" + message
    return True, message
        