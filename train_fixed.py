"""
Some hardcoded training version
Only accepts path to data as argument
Returns a trained model path
"""
import glob
import os

def get_file_count(path):
    """
    Returns the number of files in a directory
    """
    return len(glob.glob1(path,"*.[!txt]*"))

def get_closest_repeat(path):
    """
    Returns the closest repeat number
    """
    return (76*15) // get_file_count(path)

def format_dataset_config(path, character_name:str):
    """
    Formats the dataset_config.txt file, saves and returns the path
    """
    content = ""
    with open("examples/example_dataset.toml", "r", encoding="utf-8") as f:
        content = f.read()
    content = content.format(
        path = path,
        repeats = get_closest_repeat(path),
        cls_token = character_name,
    )
    if not os.path.exists("temp"):
        os.makedirs("temp")
    with open(f"temp/dataset_config_{character_name}.toml", "w", encoding="utf-8") as f:
        f.write(content)
    return f"temp/dataset_config_{character_name}.toml"

def format_train_config(character_name:str, output_dir:str, prompt_dir:str, pretrained_model:str,):
    """
    Formats the train_config.txt file, saves and returns the path
    """
    content = ""
    with open("examples/example_config.toml", "r", encoding="utf-8") as f:
        content = f.read()
    content = content.format(
        character = character_name,
        output_dir = output_dir,
        prompts = prompt_dir,
        pretrained_model = pretrained_model,
    )
    with open(f"temp/train_config_{character_name}.toml", "w", encoding="utf-8") as f:
        f.write(content)
    return f"temp/train_config_{character_name}.toml"

is_running = False
def train_auto(character_name:str, path:str, output_dir:str, prompt_dir:str, pretrained_model:str, devices:str="0"):
    """
    Trains a model with the given parameters
    """
    global is_running
    if is_running:
        return
    dataset_config = format_dataset_config(path, character_name)
    train_config = format_train_config(character_name, output_dir, prompt_dir, pretrained_model)
    is_running = True
    os.system(f"export CUDA_VISIBLE_DEVICES={devices} && python train_network.py --dataset_config {dataset_config} --config_file {train_config}")
    is_running = False

import gradio as gr

with gr.Blocks(analytics_enabled=False) as training_interface:
    with gr.Tab("path-base"):
        character_name = gr.Textbox(lines=1, label="Character Name", value="iom")
        path = gr.Textbox(lines=1, label="Path to Dataset", value="dataset/iom")
        output_dir = gr.Textbox(lines=1, label="Output Directory", value="outputs/iom")
        prompt_dir = gr.Textbox(lines=1, label="Prompt Directory", value="examples/iom.txt")
        pretrained_model = gr.Textbox(lines=1, label="Pretrained Model_path", value="models/animefull-all.ckpt")
        cuda_device_num = gr.Textbox(lines=1, label="Cuda Device Number", value="0")

        train_button = gr.Button(label="Train")
        train_button.click(
            train_auto,
            inputs=[character_name, path, output_dir, prompt_dir, pretrained_model, cuda_device_num],
        )
if __name__ == "__main__":
    training_interface.launch(share=True)
