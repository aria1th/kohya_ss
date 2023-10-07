import tempfile
import os
import gradio as gr
from easygui import msgbox

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


###
### Gradio common sampler GUI section
###


def run_cmd_sample(
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    output_dir,
    external_webui_address:str = "",
    external_webui_auth:str = "", # "user:password"
    webui_sample_json_file_path:str = "", # "sample.json"
):
    output_dir = os.path.join(output_dir, 'sample')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_cmd = ''

    if sample_every_n_epochs == 0 and sample_every_n_steps == 0:
        return run_cmd

    # Create the prompt file and get its path
    sample_prompts_path = os.path.join(output_dir, 'prompt.txt')

    with open(sample_prompts_path, 'w') as f:
        f.write(sample_prompts)

    run_cmd += f' --sample_sampler={sample_sampler}'

    if not sample_every_n_epochs == 0:
        run_cmd += f' --sample_every_n_epochs="{sample_every_n_epochs}"'

    if not sample_every_n_steps == 0:
        run_cmd += f' --sample_every_n_steps="{sample_every_n_steps}"'
        
    if external_webui_address != "":
        # check webui address is valid (http:// or https://), and json file path is valid
        if not (external_webui_address.startswith("http://") or external_webui_address.startswith("https://")):
            msgbox(f"Invalid webui address: {external_webui_address}")
            return run_cmd
        if external_webui_address.endswith("/"):
            external_webui_address = external_webui_address[:-1]
        if not webui_sample_json_file_path.endswith(".json"):
            msgbox(f"Invalid webui sample json file path: {webui_sample_json_file_path}")
            return run_cmd
        if external_webui_auth != "" and (":" not in external_webui_auth or len(external_webui_auth.split(":")) != 2):
            msgbox(f"Invalid webui auth: {external_webui_auth}")
            return run_cmd
        # now we can add the webui address and json file path to the run_cmd
        # add use_external_webui
        run_cmd += f' --use_external_webui=True'
        run_cmd += f' --webui_url="{external_webui_address}"'
        if external_webui_auth != "":
            run_cmd += f' --webui_auth="{external_webui_auth}"'
        # wait with should_wait_webui_process
        run_cmd += f' --should_wait_webui_process=True'
        # add webui_sample_json_file_path
        run_cmd += f' --sample_prompts="{webui_sample_json_file_path}"'
    else:
        run_cmd += f' --sample_prompts="{sample_prompts_path}"'
    return run_cmd


class SampleImages:
    def __init__(
        self,
    ):
        # with gr.Accordion('Sample images config', open=False):
        with gr.Row():
            self.sample_every_n_steps = gr.Number(
                label='Sample every n steps',
                value=0,
                precision=0,
                interactive=True,
            )
            self.sample_every_n_epochs = gr.Number(
                label='Sample every n epochs',
                value=0,
                precision=0,
                interactive=True,
            )
            self.sample_sampler = gr.Dropdown(
                label='Sample sampler',
                choices=[
                    'ddim',
                    'pndm',
                    'lms',
                    'euler',
                    'euler_a',
                    'heun',
                    'dpm_2',
                    'dpm_2_a',
                    'dpmsolver',
                    'dpmsolver++',
                    'dpmsingle',
                    'k_lms',
                    'k_euler',
                    'k_euler_a',
                    'k_dpm_2',
                    'k_dpm_2_a',
                ],
                value='euler_a',
                interactive=True,
            )
        with gr.Row():
            self.sample_prompts = gr.Textbox(
                lines=5,
                label='Sample prompts',
                interactive=True,
                placeholder='masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28',
                info='Enter one sample prompt per line to generate multiple samples per cycle. Optional specifiers include: --w (width), --h (height), --d (seed), --l (cfg scale), --s (sampler steps) and --n (negative prompt). To modify sample prompts during training, edit the prompt.txt file in the samples directory.',
            )

class SampleImagesExternal:
    """
    This class is for the external webui sample images config.
    """
    def __init__(self) -> None:
        with gr.Row():
            self.external_webui_address = gr.Textbox(
                lines=1,
                label='External webui address',
                interactive=True,
                placeholder='http://localhost:7860'
            )
            self.external_webui_auth = gr.Textbox(
                lines=1,
                label='External webui auth, can be empty',
                interactive=True,
                placeholder='user:password'
            )
            self.webui_sample_json_file_path = gr.Textbox(
                lines=1,
                label='External webui sample json file path(No backslash allowed, uses / instead for path separator)',
                interactive=True,
                placeholder='sample.json'
            )
            