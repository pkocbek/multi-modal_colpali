import asyncio
import base64
import gc
import inspect
import os
import pickle
import smtplib
import subprocess
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Tuple

import requests
import torch
from IPython.display import Markdown, display
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, ColPaliForRetrieval, ColPaliProcessor


def get_less_used_gpu(gpus=None, debug=False):
    """Inspects cached/reserved and allocated memory on specified gpus and return the id of the less used device.

    Args:
        gpus (list, optional): A list of gpu ids to check. Defaults to None.
        debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        int: The id of the less used gpu.
    """
    if gpus is None:
        warn = "Falling back to default: all gpus"
        gpus = range(torch.cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(",")]

    sys_gpus = list(range(torch.cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f"WARNING: Specified {len(gpus)} gpus, but only {torch.cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}"
    elif set(gpus).difference(sys_gpus):
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[: len(unavailable_gpus)]
        warn = f"GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}"

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = torch.cuda.memory_allocated(i)
        cur_cached_mem[i] = torch.cuda.memory_reserved(i)
        max_allocated_mem[i] = torch.cuda.max_memory_allocated(i)
        max_cached_mem[i] = torch.cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print(
            "Current allocated memory:",
            {f"cuda:{k}": v for k, v in cur_allocated_mem.items()},
        )
        print(
            "Current reserved memory:",
            {f"cuda:{k}": v for k, v in cur_cached_mem.items()},
        )
        print(
            "Maximum allocated memory:",
            {f"cuda:{k}": v for k, v in max_allocated_mem.items()},
        )
        print(
            "Maximum reserved memory:",
            {f"cuda:{k}": v for k, v in max_cached_mem.items()},
        )
        print("Suggested GPU:", min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    """Deletes GPU memory for a list of variables.

    Args:
        to_delete (list): A list of variable names to delete.
        debug (bool, optional): If True, prints debug information. Defaults to False.
    """
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print("Before:")
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()
    if debug:
        print("After:")
        get_less_used_gpu(debug=True)


def encode_image(image_path):
    """Encodes an image to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_to_pickle(filepath, **kwargs):
    """Saves a dictionary of objects to a pickle file.

    Args:
        filepath (str): The path to the pickle file.
        **kwargs: The objects to save.
    """
    objects_dict = kwargs
    with open(filepath, "wb") as file:
        pickle.dump(objects_dict, file)


def format_msgs(prompt: str, img_links: list, text: str = ""):
    """Formats a message for the OpenAI API.

    Args:
        prompt (str): The prompt for the message.
        img_links (list): A list of image links.
        text (str, optional): Additional text for the message. Defaults to "".

    Returns:
        list: A list of messages formatted for the OpenAI API.
    """
    part = [{"type": "text", "text": prompt + text}] if text else [{"type": "text", "text": prompt}]
    if img_links:
        for img_link in img_links:
            part.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_link)}"},
                }
            )

    return [{"role": "user", "content": part}]


def check_vllm_status(url: str = "http://localhost:8000/health") -> bool:
    """Checks if the VLLM server is running and healthy.

    Args:
        url (str, optional): The URL of the health check endpoint. Defaults to "http://localhost:8000/health".

    Returns:
        bool: True if the server is healthy, False otherwise.
    """
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def monitor_vllm_process(
    vllm_process: subprocess.Popen,
    check_interval: int = 5,
) -> Tuple[bool, str, str]:
    """Monitors a VLLM process and returns its status, stdout, and stderr.

    Args:
        vllm_process (subprocess.Popen): The VLLM process to monitor.
        check_interval (int, optional): The interval at which to check the server status. Defaults to 5.

    Returns:
        Tuple[bool, str, str]: A tuple containing the success status, stdout, and stderr.
    """
    print("Starting VLLM server monitoring...")

    while vllm_process.poll() is None:
        if check_vllm_status():
            print("✓ VLLM server is up and running!")
            return True, "", ""

        print("Waiting for VLLM server to start...")
        time.sleep(check_interval)

        if vllm_process.stdout.readable():
            stdout = vllm_process.stdout.read1().decode("utf-8")
            if stdout:
                print("STDOUT:", stdout)

        if vllm_process.stderr.readable():
            stderr = vllm_process.stderr.read1().decode("utf-8")
            if stderr:
                print("STDERR:", stderr)

    stdout, stderr = vllm_process.communicate()
    return False, stdout.decode("utf-8"), stderr.decode("utf-8")