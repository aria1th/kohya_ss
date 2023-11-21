# Automated Training

이 문서에서는 자동화된 학습에 대해 다룹니다.

세 파트로 구성되어 있습니다.

**A, 데이터셋 구성**

A1. 단순한 방법 (이미지 및 캡션만)

A2. 여러 폴더를 구성해서 사용하는 방법 및 고급 설정들

**B. 학습 방법**

UI를 사용하는 경우는 다루지 않습니다. (UI로 다루더라도 결국 실행되는 것은 커맨드이며, UI는 어디까지나 보조 도구로 구현되어 있습니다)

B1. 학습 시 설정들과 이를 덮어씌워, 여러 설정으로 학습을 테스트하는 방법

B2. 여러 GPU가 있을 경우, 이를 병렬적으로 학습에 사용하는 방법

**C. 학습 로그 기록 및 검색**

C0. 추론용 WebUI 설정 및 요구 사항

C1. Wandb를 사용한 로그 기록 및 확인

C2. AimStack을 사용한 로그 기록 및 확인

- Section A. Dataset Preparation
    
    이미지를 저장한 후 Tagging / Captioning 작업을 수행합니다. 여러가지 방법이 있으나, 별도 문서에서 다루도록 합니다.
    
    원칙적으로 같은 폴더 내에 이미지와 동일한 이름으로 .txt 확장자로 태그(캡션)을 저장합니다. 
    
    A1. 단순한 방법의 경우, 더이상 고려할 필요가 없습니다.
    
    - A2. Dataset toml 구성법
        
        ```toml
        [[datasets]]
        
            [[datasets.subsets]]
            num_repeats = 200
            image_dir = "/dataset/images"
            class_tokens = "arisu"
            mask_dir ="/dataset/mask"
        
        [general]
        resolution = 768
        shuffle_caption = true
        keep_tokens = 0
        flip_aug = false
        caption_extension = ".txt"
        enable_bucket = true
        bucket_reso_steps = 64
        bucket_no_upscale = false
        min_bucket_reso = 320
        max_bucket_reso = 1280
        ```
        
        위와 같은 Config를 작성합니다.
        
        mask_dir의 경우, masked loss를 적용하고 싶을 경우 사용 가능하며 위와 같이 사용해야 합니다.
        
        여러 폴더를 각각의 dataset subset로 작성할 수 있습니다. 각각에 repeat를 지정하는 등의 세팅이 가능합니다.
        
- Section B. Training
    
    학습 Repository는 [https://github.com/aria1th/kohya_ss](https://github.com/aria1th/kohya_ss) 를 참고해 주세요.
    
    기본적인 학습 command는 train-network.py를 사용합니다.
    
    본 섹션에서는 자동화를 위한 automate-train.py를 다룹니다.
    
    자동화 학습에서는 두가지 config를 다루게 됩니다.
    
    1. 학습 command로 들어갈 기본 설정들
    2. 학습 command에서 덮어씌울 설정들 (바꿀 설정들)
    
    다음과 같은 설정을 생각해 주세요.
    
    `설정 A로 학습할 때, 학습률을 0.0001, 0.0004, 0.0008로 바꾸는 경우`
    
    이 경우 1. 은 학습률을 0.0001로 세팅한 설정에 해당하며, 2.에서는 세가지 설정을 리스트로 주는 것을 생각할 수 있습니다.
    
    따라서, 다음과 같이 config를 작성한다고 가정합시다.
    
    (주석의 경우 제거해야 json으로 사용 가능합니다)
    
    ```toml
    # Default Config
    {
            "project_name_base" : "character_name",
            "model_file" :"/data/base_model.ckpt",
            "optimizer" : "AdamW8bit",
            "network_dim" : 64,
            "network_alpha" : 8,
            "zero_terminal_snr" : false,
            "gor_regularization" : false,
            "gor_regularization_type" : "inter",
            "face_crop_aug_range" : "2.0,3.0",
            "gor_ortho_decay" : 0.99,
            "conv_dim" : 8,
            "conv_alpha" : 1,
            "num_repeats" : 15,
            "clip_skip" : 1,
            "epoch_num" : 15,
            "train_batch_size" : 2,
            "unet_lr" : 3e-05,
            "text_encoder_lr" : 6e-05,
            "resolution" : "768,768",
        	"target_path" : "/data/trains/results/train", # Move artifact?
            "temp_dir" : "/data/trains/tmp/dir_to_log/", #Logging Dir at ./_logs
            "images_folder" : "",
            "cuda_device" : "1,2",
        	"repo_dir" : "/data/kohya_ss",
            "port" : 20067,
            "v_parameterization" : false,
            "log_with" : "tensorboard",
            "wandb_api_key" : "",
            "log_tracker_config" : "",
            "webui_url" : "",
            "use_external_webui" : false,
            "should_wait_webui_process" : false,
            "random_crop" : false,
            "mask_loss" : false,
            "mask_threshold" : 1,
            "process_title": "character_name_lora",
            "webui_auth" : "<Id>:<Password>"
    }
    ```
    
    위와 같이 default에 argument를 저장할 수도 있으며, 반대로 tuning에서 지정해서 override할 수 있습니다.
    
    ```toml
    # Tuning Config
    {
        "project_name_base": "character_name-test-debiased-loss",
        "epoch_num": 10,
        "sample_opt": "epoch",
        "sample_num": 1,
        "seed_list": [
            42
        ],
        "num_repeats_list": [
            15
        ],
        "keep_tokens_list": [
            1
        ],
        "unet_lr_list": [
            0.0001,
            0.00025,
            0.0004
        ],
        "clip_skip_list": [
            1,
            2
        ],
        "text_encoder_lr_list": [
            5e-05
        ],
        "resolution_list": [
            768
        ],
        "lr_scheduler": "cosine_with_restarts",
        "lora_type": "LoRA",
        "prompt_path": "/data/character_name-lineart.json",
        "port": 20100,
        "cuda_device": "1",
        "lbw_weights_list": [
            "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
        ],
        "face_crop_aug_range_list": [
            "1.0,1.0"
        ],
        "random_crop_list": [
            false
        ],
        "images_folder": "/data5/character_dataset/ioms/character_name_dataset/15_boy",
        "log_with": "aim",
        "wandb_api_key": "",
        "log_tracker_config_template": "/data/log_config.toml",
        "webui_url": "http://127.0.0.1:9050/",
        "use_external_webui": true,
        "should_wait_webui_process": false,
        "webui_auth": "username:password",
        "debiased_estimation_loss_list": [
            false,
            true
        ],
        "process_title": "character_name_lora_debiased_loss_sh"
    }
    ```
    
    위와 같이 설정할 수 있습니다. default argument에서 `_list` 로 처리된 keyword들이 순차적으로 조합되어 들어가게 됩니다.
    
    이때 여러 list를 사용한 경우에 대해서도 살펴보겠습니다.
    
    ```toml
        "seed_list": [
            42
        ],
        "unet_lr_list": [
            0.0001,
            0.00025,
            0.0004
        ],
        "clip_skip_list": [
            1,
            2
        ],
    ```
    
     위와 같이 list임에도 element가 하나인 경우, list는 없는 것으로 취급됩니다 (seed로)
    
    반면, `unet_lr_list` 의 경우 3가지, `clip_skip_list` 은 2가지 조합이므로 `product`를 사용해 6가지 조합을 테스트하게 됩니다.
    
    위와 같은 ‘변화한 경우들’은 wandb 또는 aim 등의 run name으로 등록되므로 결과에서 쉽게 비교 가능합니다.
    
    위 학습을 진행하는 command는 `python3.10 [automate-train.py](http://automate-train.py/) --default_config_path /data/default-config.json --tuning_config_path /data/tuning-config.json` 과 같습니다.
    
    kohya_ss 내에 venv를 세팅한 후 실행해 주세요. conda 등에서의 실행은 가정하지 않았습니다.
    
    학습을 중단할 경우, 이후 command들은 실행되지 않습니다. index 표기나 index skip 등의 설정도 있으니 `python 3.10 [automate-train.py](http://automate-train.py) --help`로 확인해 주세요.
    
    CUDA device가 여유로울 경우, 이를 병렬적으로 실행시키는 것으로 학습 속도를 증가시킬 수 있습니다. 이 경우, log는 확인하기 어려워지므로 유의해 주세요.
    
    CUDA device를 0, 1, 2, 3을 사용할 경우 다음과 같은 command를 사용 가능합니다:
    
    `python3.10 [automate-train.py](http://automate-train.py/) --default_config_path /data/default-config.json --tuning_config_path /data/tuning-config.json --cuda_devices_distributed "0,1, 2, 3"`
    
    예를 들어 학습 config에서 CUDA devices가 0으로 지정되어 있었을 경우, deivce number는 무시되고, count를 사용해서 4개의 학습을 병렬적으로 수행하게 됩니다.
    
    단 하나라도 실패가 발생한다면 더이상의 추가 command를 실행하지 않고, 진행중이던 학습이 종료되기를 기다리게 됩니다.
    
    여러 명령을 수행할 때, GPU 메모리가 충분할 경우 자동으로 사용하고 싶다면 
    
    `-cuda-memory-limit 9` 등을 사용할 수 있습니다. 이 경우 9GB 이상의 VRAM이 있는 경우에만 대상 디바이스로 지정됩니다.
    
- Section X. WebUI Inference
    
    학습시, kohya_ss만으로 inference를 하는 것은 속도 측면, 또는 기능 측면에서 부족한 점이 있을 수 있습니다. 따라서 webui에 모델을 업로드, 동기화, 추론하는 기능을 제공합니다.
    
    WebUI는 외부 URL일수도, 내부 URL일 수도 있습니다.
    
    단, 두가지 extension을 필요로 합니다:
    
    **[https://github.com/aria1th/webui-model-uploader](https://github.com/aria1th/webui-model-uploader)**
    
    **[https://github.com/aria1th/sd-webui-agent-scheduler](https://github.com/aria1th/sd-webui-agent-scheduler) (NoneType 버그 발생시)**
    
    이 외에도 API를 사용하고 싶은 extension등을 실제 환경과 동일하게 맞춰주세요.
    
    inference에 쓸 prompt 또는 setting은 .txt, .json 형식을 사용할 수 있습니다.
    
    (.toml 등은 json과 동일하게 parsing되도록 합니다)
    
    ```toml
    [
    	{
            "prompt": "iom, 1boy, upper body shot, school uniform, jacket",
            "seed": 43,
            "steps": 23,
            "negative_prompt": "easynegative, low quality, worst quality, bad anatomy, negative_hand-neg, poor, low effort",
            "cfg_scale": 7.5,
            "width": 512,
            "height": 768,
            "hr_scale": 2, # optional
            "enable_hr": true,  # optional
            "hr_upscaler": "R-ESRGAN 4x+ Anime6B",  # optional
            "denoising_strength": 0.5,  # optional
            "sampler_name": "DPM++ 2M Karras"  # optional
        },
    ...
    ]
    ```
    
    위와 같은 json을 통해, 기본적인 upscale을 한 webui inference가 가능합니다.
    
    또한 nested dict 등을 통해 API argument를 모두 사용 가능합니다.
    
    예외적으로, lora name을 직접 지정하고 싶은 경우가 있을 수 있습니다.
    
    ```toml
    [
    	{
            "prompt": "iom, 1boy, upper body shot, school uniform, jacket <lora:$1:1>",
            "seed": 43,
            "steps": 23,
    				"regex_to_replace" : "$1" # Here is the regex
            "negative_prompt": "easynegative, low quality, worst quality, bad anatomy, negative_hand-neg, poor, low effort",
            "cfg_scale": 7.5,
            "width": 512,
            "height": 768,
            "hr_scale": 2, # optional
            "enable_hr": true,  # optional
            "hr_upscaler": "R-ESRGAN 4x+ Anime6B",  # optional
            "denoising_strength": 0.5,  # optional
            "sampler_name": "DPM++ 2M Karras"  # optional
        },
    ...
    ]
    ```
    
    위 형식을 사용할 경우 prompt 및 negative prompt 내의 $1이 해당 lora의 이름으로 바뀌어 들어가게 됩니다.
    
    가장 단순한 txt 형태의 프롬프트는 다음과 같습니다. 예를 들어, 12가지의 prompt를 테스트하고 싶다면:
    
    ```toml
    iom 1girl, upper body shot, school uniform,  jacket --d 43 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, full body shot, riding motorcycle, school uniform,   jacket --d 43 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, close up shot, school uniform,  jacket --d 43 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, upper body shot, eating cake, school uniform,   jacket --d 42 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, full body shot, school uniform,   jacket --d 42 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, close up shot, school uniform,   jacket --d 42 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, upper body shot,monochrome, simple white background, lineart, eating cake, school uniform,   jacket --d 42 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, full body shot,monochrome, simple white background, lineart, school uniform,   jacket --d 42 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, close up shot,monochrome, simple white background, lineart, school uniform,   jacket --d 42 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, upper body shot,monochrome, simple white background,lineart, school uniform,  jacket --d 43 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, full body shot,monochrome, simple white background,lineart, riding motorcycle, school uniform,   jacket --d 43 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    iom 1girl, close up shot,monochrome, simple white background,lineart, school uniform,  jacket --d 43 --s 30 --n easynegative, low quality, worst quality, bad anatomy,bad composition, poor, low effort --l 8.5 --w 512 --h 768
    ```
    
    이와 같이 여러 프롬프트를 inference하도록 할 수 있습니다.
    
- Section C. Reviewing Logs
    
    `temp_dir`를 지정했던 경로에 `_logs` 폴더가 생성되며, 해당 폴더에 accelerate / aim / wandb log가 저장됩니다.
    
    Wandb는 API key 및 log method를 설정하면 자동으로 진행되므로, AIM의 경우에 대해 설명하겠습니다.
    
    `/data/trains/tmp/dir_to_log/_logs`
    
    위 경로에 로그가 생성되었을 경우 aim은 다음 명령어로 실행이 가능합니다.
    
    `aim up -p 20089 --repo "/data/trains/tmp/dir_to_log/_logs"`
    
    단, aim은 venv 내에서 실행이 가능하며, 리눅스 환경에만 설치가 가능하다는 점을 유의해주세요. (2023-11-21 기준)
    
    이후 [localhost:20089](http://localhost:20089) 또는 ngrok 등으로 포워딩된 address로 접속하는 것으로, AIM log를 볼 수 있습니다.
