import pandas as pd
import os
import requests
from openai import OpenAI


from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    #RapidOcrOptions,
    TableFormerMode,
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
import httpx
from docling_core.types.doc.document import DoclingDocument


from docling.chunking import HybridChunker

from pathlib import Path
from io import BytesIO
import time

from langchain_core.documents import Document
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import RefItem
from docling.datamodel.pipeline_options import (
    granite_picture_description,
    smolvlm_picture_description,
)


from PIL import Image
import torch
#import flash_attn
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader

from PIL import Image
from IPython.display import display, Markdown
from pdf2image import convert_from_path
from transformers import ColPaliForRetrieval, ColPaliProcessor
import json
import re


def doc_conv (ocr = True, check_only = False):
    """
    pdf parser and chunker
    """
    #defining pipeline options for pdf parsing
    #scale=1 correspond of a standard 72 DPI image
    IMAGE_RESOLUTION_SCALE = 2.0
    pipeline_options = PdfPipelineOptions()
    if ocr == True:
        # rapidocr_models_root = "./src/utils"
        # det_model_path = os.path.join(rapidocr_models_root, "en_PP-OCRv3_det_infer.onnx")
        # rec_model_path = os.path.join(rapidocr_models_root, "ch_PP-OCRv4_rec_server_infer.onnx")
        # cls_model_path = os.path.join(rapidocr_models_root, "ch_ppocr_mobile_v2.0_cls_train.onnx")

        # ocr_options = RapidOcrOptions(
        #     force_full_page_ocr=True,
        #     det_model_path=det_model_path,
        #     rec_model_path=rec_model_path,
        #     cls_model_path=cls_model_path,
        # )
        pipeline_options.do_ocr = True
        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options        
    else:
        pipeline_options.do_ocr = False


    if check_only == False: 
        pipeline_options.do_table_structure=True
        pipeline_options.table_structure_options.do_cell_matching = True  # uses text cells predicted from table structure model
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = (
            granite_picture_description # <-- the model choice
        )
        pipeline_options.picture_description_options.prompt = (
            "Describe the image in four sentences. Be consise, scientific and accurate. Provide numbers if it improves the description."
        )

    
    accelerator_options = AcceleratorOptions(
        num_threads=8,  device=AcceleratorDevice.CUDA,
        cuda_use_flash_attention2 = True
    )
    
    pipeline_options.accelerator_options = accelerator_options

    doc_conv = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.CSV,
                InputFormat.MD,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline  # , backend=MsWordDocumentBackend
                ),
            },
        )
    )
    ###########################################################################
    return doc_conv

def check_ocr (file_name):
    """
    Simple ocr checker - checks first page if any text is processed, if not returns True (for ocr)
    Note that special cases are not considered - if text is processed but is not legible (mixed font maps, etc)
    """
    #check first page, if no text switch to ocr
    parse_first_pg = doc_conv(ocr = False, check_only = True).convert(source=file_name, page_range=[1,1])
    text = []

    for element, _level in parse_first_pg.document.iterate_items():
        if isinstance(element, TextItem):
            #print(element.text)
            text.append (element.text)

    #if 1 or more elements are texts we presume no ocr is needed, otherwise we do ocr
    return False if len(text)>=0 else True  


def get_less_used_gpu(gpus=None, debug=False):
    """
    Housekeeping for processes
    Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device
    """
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(torch.cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(torch.cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {torch.cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

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
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    """
    Housekeeping, deletes GPU memory for variable
    """
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=True)    


from pathlib import Path

from PIL import Image

def resize_image(image: Image.Image, min_size: int = 224, max_size: int = 1300 ) -> Image.Image:
    """
    Resize an image to a minimum size
    """
    if min(image.size) < min_size:
        ratio = min_size / min(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    return image


import uuid
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
from docling.chunking import HybridChunker
from docling_core.types.doc.document import PictureDescriptionData

from langchain_core.documents import Document
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import RefItem

#did not change vd_dir in metadata, since handled elsewhere, also note that one of the parameter could be username or ss.username (for refactoring whole code)
def data_preparation (conversion:list, vd_dir, vd_tokenizer, mm_dir="",  only_text=False, page_images= True):
    """
    data preparation for vectorization
    type : text, multi, both
    """
    all_docs=[]
    for el in conversion:
        filename = el["filename"]
        filename_link = el["link"]
        documents_id= str(uuid.uuid4())
        
        save_dir = vd_dir if mm_dir == "" else mm_dir
        save_root = Path(save_dir)
        save_root.mkdir(parents=True, exist_ok=True)

        doc_stem = Path(filename).stem
        doc_name = el["document"].name
        pg_image_dir = save_root / "pg_images"
        # Save page images
        if page_images:
            pg_image_dir.mkdir(parents=True, exist_ok=True)
            for page_no, page in  el["document"].pages.items():
                page_no = page.page_no
                page_image_filename = pg_image_dir / f"{doc_stem}_{page_no:03d}.png"
                with page_image_filename.open("wb") as fp:
                    img = resize_image(page.image.pil_image)
                    img.save(fp, format="PNG")

        if not only_text:
            tables_dir = save_root / "tables"
            images_dir = save_root / "images"
            # Save images of figures and tables
            table_counter = 0
            picture_counter = 0
            for element, _level in el["document"].iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    if table_counter == 1:
                        tables_dir.mkdir(parents=True, exist_ok=True)
                    element_image_filename = tables_dir / f"{doc_stem}_table_{table_counter:03d}.png"
                    with element_image_filename.open("wb") as fp:
                        img = resize_image(element.get_image(el["document"]))
                        img.save(fp, "PNG")

                if isinstance(element, PictureItem):
                    picture_counter += 1
                    if picture_counter == 1:
                        images_dir.mkdir(parents=True, exist_ok=True)
                    element_image_filename = images_dir / f"{doc_stem}_img_{picture_counter:03d}.png"
                    with element_image_filename.open("wb") as fp:
                        img = resize_image(element.get_image(el["document"]))
                        img.save(fp, "PNG")

        texts: list[Document] = []
            #doi=""
        for chunk in HybridChunker(tokenizer=vd_tokenizer).chunk(el["document"]):

            items = chunk.meta.doc_items
            if len(items) == 1 and isinstance(items[0], TableItem) and not only_text:
                continue # we will process tables later
            ref = " ".join(map(lambda item: item.get_ref().cref, items))
            #print(ref)
            text = chunk.text
            # if doi=="":
            #     doi = re.search(r'\b10\.\d{4,9}/[-.;()/:\w]+', txt).group(0)
            document = Document(
                page_content=text,
                metadata={
                    "document_name": el["document"].origin.filename,
                    "document_id":documents_id,
                    "document_link":filename_link,
                    "type": "text",
                    "page_no": chunk.meta.doc_items[0].prov[0].page_no, #get first page_no
                    "ref": ref,
                    "caption":"",
                    "img_link":"",
            },
            )
            texts.append(document)
        #proc_documents.append(texts)
        if only_text:
            all_docs.extend(texts)
            print(f"For {filename} there were {len(texts)} texts processed only_text={only_text}.")
            continue

    
        tables: list[Document] = []
        for idx, table in enumerate(el["document"].tables, start=1):
            #print(docling_document)
            if table.label in [DocItemLabel.TABLE]:
                ref = table.get_ref().cref
                table_df: pd.DataFrame = table.export_to_dataframe()
                text = table_df.to_markdown()
                try:
                    caption = RefItem(cref=table.captions[0].cref).resolve(el["document"]).text
                except:
                    caption = ""
                
                document = Document(
                    page_content= caption +" "+ text if caption != "" else text,
                    metadata={
                        "document_name": el["document"].origin.filename,
                        "document_id":documents_id,
                        "document_link":filename_link,
                        "type": "table",
                        "page_no": table.prov[0].page_no,
                        "ref": ref,
                        "caption":caption,
                        "img_link":str(tables_dir / f"{doc_name}_table_{idx:03d}.png"),
                    },
                )
                tables.append(document)
        #proc_documents.append(tables)



        pictures: list[Document] = []
            
        for idx, picture in enumerate(el["document"].pictures, start=1):
            ref = picture.get_ref().cref
            try:
                caption = RefItem(cref=picture.captions[0].cref).resolve(el["document"]).text
            except:
                caption = ""
            link = str(images_dir / f"{doc_name}_img_{idx:03d}.png")
            #print(idx)
            text_ =[]
            for annotation in picture.annotations:
                if not isinstance(annotation, PictureDescriptionData):
                    continue
                text_.append(annotation.text)
            text = " ".join(text_)
            #print(text)
            #vision_model.invoke(vision_prompt, image=encode_image(image))
            document = Document(
                page_content= caption+text,
                metadata={
                    "document_name": el["document"].origin.filename,
                    "document_id":documents_id,
                    "document_link":filename_link,
                    "type": "image",
                    "page_no": picture.prov[0].page_no,
                    "ref": ref,
                    "caption":caption,
                    "img_link":link,  

                },
            )
            pictures.append(document)
        #proc_documents.append(pictures)
        #free_memory([pipe])

        all_docs.extend(texts+tables+pictures)
        
        print(f"For {filename} there were {len(texts)} texts, {len(tables)} tables and {len(pictures)} images processed (needs VLM descriptions), equals {len(texts+tables+pictures)} documents.")

    print(f"Total number of elements processed: {len(all_docs)}.")

        
        
    return all_docs

def models_local (ports, api_key="EMPTY"):
    dict_list=[]
    for port in ports: 
        api_url = os.getenv('API_URL', "http://localhost:"+str(port)+"/v1")
        try:
            response = requests.get(api_url) 
            print(f"{response.status_code}, port: {port}")
            client = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key=api_key,
                base_url=api_url,
            )

            dict_list.append({"model_id": client.models.list().data[0].id, "api_url": api_url, "api_key" : os.getenv('API_URL', api_key)})
                #print(dict_list)
        except requests.exceptions.RequestException as err:
            print(f"api connection error: {err}")
            dict_list.append({"model_id": "Not_working", "api_url": api_url, "api_key" : os.getenv('API_URL', api_key)})

    return dict_list

def models_used (local_ports, gpt_models, VD_text, VD_MM):

    models_used = models_local (local_ports)

    gpt_models = [{"model_id": gpt, "api_url": os.getenv('API_URL', "https://api.openai.com/v1/"), "api_key" : os.getenv("OPENAI_API_KEY")} for gpt in gpt_models]
    models_used = [*models_used, *gpt_models ]

    models_used = [{**model, **{"vd_text": VD_text}}  for model in models_used]

    if len(VD_MM) == len(models_used ):
        models_used = [{**model, **{"vd_MM": vd}}  for model,vd in zip(models_used,VD_MM)]
    else:
        print(f"Length of multimodal vectoDB {len(VD_MM)} not equals to models  {len(models_used )}.")

    models_used = [model for model in models_used if model["model_id"] != "Not_working"]

    return(models_used)


import base64

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


import pickle

def save_to_pickle(filepath, **kwargs):
    objects_dict = kwargs
    with open(filepath, 'wb') as file:
        pickle.dump(objects_dict, file)


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


def format_msgs (prompt:str, img_links:list, text:str =""):
    """
    formating message for openai api, the structure is as follows
    prompt: prompt for generation
    img_links: a list of local image links, can be from from vector DB, note that models have limitations and mostly we do summaries image by image
    text: some additional text after prompt, for example if context a mix of text and images this can merged 
    """
    if text == "":
        part = [{"type": "text", "text": prompt}]
    else:
        part = [{"type": "text", "text": prompt + text}]
    if img_links != []:    
        for img_link in img_links:
                part.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_link)}"}})

    msgs = [{"role": "user", "content": part}]

    return msgs


# def api_models_imgs (img_links, prompt, model_id, base_url = "http://localhost:8000/v1" , api_key = "EMPTY", max_tokens=300, text ="", text_cutoff=1500):
#     try:
#         openai.api_key = api_key
#     except:
#         pass
#     #print(f"Processing for {model_id}.")
#     client = OpenAI(
#         # defaults to os.environ.get("OPENAI_API_KEY")
#         api_key=  api_key,
#         base_url= base_url,
#     )
   
#     # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
#     # def completion_with_backoff(**kwargs):
#     #     return client.chat.completions.create(**kwargs)
#     if text == "":
#         part = [{"type": "text", "text": prompt}]
#     else:
#         part = [{"type": "text", "text": prompt + text}]
#     if img_links != []:    
#         for img_link in img_links:
#                 part.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_link)}"}})

#     msgs = [{"role": "user", "content": part}]

#     chat_response = client.chat.completions.create(
#         model=model_id,
#         messages=msgs,
#         max_completion_tokens=max_tokens,
#     )

#     return chat_response.choices[0].message.content[:text_cutoff]



def api_models_one_img (img_links, texts, models_data, img_prompt, text_prompt, max_tokens=300, text_cutoff=1500, save_tmp= "tmp_save2.pkl"):
    
    gen_text=[]

    for model_data in models_data:
        print(f"Processing for {model_data["model_id"]}.")
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key= model_data["api_key"],
            base_url=model_data["api_url"],
        )
        @retry(wait=wait_random_exponential(min=10, max=180), stop=stop_after_attempt(10))
        def completion_with_backoff(**kwargs):
            return client.chat.completions.create(**kwargs)
        tmp_data=[]
        for img_link, text in tqdm(zip(img_links,texts)):
            if img_link =="":
                part = [{"type": "text", "text": text_prompt + text}]
            else:
                part = [
                    {"type": "text", "text": img_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_link)}"}},
                         ]

            msgs = [{"role": "user", "content": part}]

            chat_response = completion_with_backoff(
                model=model_data["model_id"],
                messages=msgs,
                max_completion_tokens=max_tokens,
            )
            one_resp = {"model": model_data["model_id"], "link": img_link, "output": chat_response.choices[0].message.content[:text_cutoff]}

            tmp_data.append (one_resp)
            gen_text.append(one_resp)

            tmp_save = model_data["model_id"]
            tmp_save =tmp_save.split("/")[-1] 
        save_to_pickle(tmp_save[:15] + ".pkl", processed_tmp=tmp_data)
           
    return gen_text



import subprocess

import requests
import time
from typing import Tuple
import sys

def check_vllm_status(url: str = "http://localhost:8000/health") -> bool:
    """Check if VLLM server is running and healthy."""
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def monitor_vllm_process(vllm_process: subprocess.Popen, check_interval: int = 5) -> Tuple[bool, str, str]:
    """
    Monitor VLLM process and return status, stdout, and stderr.
    Returns: (success, stdout, stderr)
    """
    print("Starting VLLM server monitoring...")

    while vllm_process.poll() is None:  # While process is still running
        if check_vllm_status():
            print("✓ VLLM server is up and running!")
            return True, "", ""

        print("Waiting for VLLM server to start...")
        time.sleep(check_interval)

        # Check if there's any output to display
        if vllm_process.stdout.readable():
            stdout = vllm_process.stdout.read1().decode('utf-8')
            if stdout:
                print("STDOUT:", stdout)

        if vllm_process.stderr.readable():
            stderr = vllm_process.stderr.read1().decode('utf-8')
            if stderr:
                print("STDERR:", stderr)

    # If we get here, the process has ended
    stdout, stderr = vllm_process.communicate()
    return False, stdout.decode('utf-8'), stderr.decode('utf-8')


def modify_orig (orig_documents, gen_texts):

    new_doc = []

    for gen_text, el in zip(gen_texts, orig_documents):
    
        tmp_el = el
        if tmp_el.metadata["type"] in [ "image"]:
            tmp_el.page_content = gen_text
            #tmp_el.metadata["user_id"] = "test_user"
            #tmp_el.metadata["user_access"] = "user"
            #tmp_el.metadata["orig_text"] = el.page_content
        new_doc.append(tmp_el) 

    return new_doc


def show_results(qdrant_retrieval):
    if hasattr(qdrant_retrieval, 'points'):
        for el in qdrant_retrieval.points:
            print(f"Score: {el.score}, file: {el.payload["document_name"]}, page: {el.payload["page_no"]}, type: {el.payload["type"]}, link: {el.payload["document_link"]}. ")
            image = Image.open(el.payload["img_link"]).convert('RGB')
            display(image)
    else:
        for el in qdrant_retrieval:
            print(f"Score: {el[1]}, file: {el[0].metadata["document_name"]}, page: {el[0].metadata["page_no"]}, type: {el[0].metadata["type"]}, link: {el[0].metadata["document_link"]}. ")

            if el[0].metadata["type"] in ["image", "pdf_page"]:
                image = Image.open(el[0].metadata["img_link"]).convert('RGB')
                display(image)
            if el[0].metadata["type"] in ["text"]:
                print(f"{el[0].page_content} \n")
            if el[0].metadata["type"] in ["table"]:
                display(Markdown(el[0].page_content))
                #print(f"{el[0].page_content} \n")

# def summarise_context (context, model_info, img_prompt, text_prompt):
#     try:
#         openai.api_key = model_info["api_key"]
#     except:
#         pass
#     summary = []
#     if hasattr(context, 'points'):
#         for el in context.points:
#             sum_text= api_models_imgs ([el.payload["img_link"]], img_prompt, model_info["model_id"], base_url = model_info["api_url"], api_key = model_info["api_key"], max_tokens=300, text =" ", text_cutoff=1500)
#             summary.append(sum_text)
#     else:
#         for el in context:
#             if el[0].metadata["type"] in ["image", "pdf_page"]:
#                 sum_text= api_models_imgs ([el[0].metadata["img_link"]], img_prompt, model_info["model_id"], base_url = model_info["api_url"], api_key = model_info["api_key"], max_tokens=300, text =" ", text_cutoff=1500)
#                 summary.append(sum_text)
#                 #print(sum_text)
#             elif el[0].metadata["type"] in ["text", "table"]:
#                 sum_text= api_models_imgs ([], text_prompt, model_info["model_id"], base_url = model_info["api_url"], api_key = model_info["api_key"], max_tokens=300, text =el[0].page_content, text_cutoff=1500)
#                 summary.append(sum_text)
#                 #print(sum_text)
#             else:
#                 print(f"Format of document is not correct!!")

#     return "\n".join(summary)


def convert_pdfs_to_images(pdf_files, save_loc):
    """Convert PDFs into a dictionary of PIL images."""
    #pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    all_images = []
    Path(save_loc).mkdir(parents=True, exist_ok=True)

    for doc_id, pdf_file in enumerate(pdf_files):
        #pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_file)
        
        for page_num, image in enumerate(images, start = 1):
            img_loc=f'{save_loc}/{pdf_file.split("/")[-1].split(".")[0]}_{(page_num):03d}.png'
            img= resize_image(image.convert("RGB"))
            all_images.append({"filename": pdf_file.split("/")[-1], "page_no": page_num, "image": img,"img_link": img_loc})
            img.save(img_loc, "png")

    return all_images


def convert_pdf_dir_to_images(pdf_dir: str) -> dict[str, list[Image.Image]]:
    """
    Convert every PDF in a directory to in-memory PIL images.

    Returns a mapping of filename -> list of page images.
    """
    pdf_dir_path = Path(pdf_dir)
    pdf_filenames = sorted(
        [entry.name for entry in pdf_dir_path.iterdir() if entry.suffix.lower() == ".pdf"]
    )

    images_per_pdf: dict[str, list[Image.Image]] = {}
    for filename in pdf_filenames:
        pdf_path = pdf_dir_path / filename
        images_per_pdf[filename] = convert_from_path(str(pdf_path))

    return images_per_pdf

def encode_image_to_data_url(image_path: str, fixed_width: int = 1024) -> str | None:
    """Convert an image into a base64 data URL for multimodal prompts."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    width, height = img.size
    if width <= 0 or height <= 0:
        return None
    new_height = int(fixed_width * height / width)
    resized = img.resize((fixed_width, max(new_height, 1)), resample=Image.LANCZOS)
    buffer = BytesIO()
    resized.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def build_choice_string(answers: list[str]) -> str:
    """Return a formatted string with MCQ answer choices."""
    return "\n".join(f"{letter}. {option}" for letter, option in zip(["A", "B", "C", "D"], answers))

def build_instruction_block(question: str, answers: list[str]) -> str:
    """Create the base instruction block used for multimodal MCQ prompting."""
    return (
        "You are an expert biomedical researcher. Carefully read the question and the answer choices.\n"
        f"Question: {question}\nChoices:\n{build_choice_string(answers)}\n"
        "If contextual snippets are provided, use them judiciously. "
        "Respond with a single capital letter (A, B, C, or D)."
    )

def build_reference_from_metadata(metadata: dict) -> str:
    """Compact reference label composed of document name and page index."""
    doc = metadata.get("document_name") or metadata.get("file_name") or "doc"
    page = metadata.get("page_no") or metadata.get("page_id")
    return f"{doc}_pg_{page}" if page is not None else doc

def document_to_context_entry(doc, score: float) -> dict:
    """
    Convert a LangChain document returned from Qdrant into a neutral context entry
    that can be consumed by multimodal prompt builders.
    """
    metadata = doc.metadata or {}
    doc_type = metadata.get("type", "text")
    return {
        "type": "image" if doc_type in {"image", "pdf_page"} else "text",
        "text": doc.page_content if doc_type in {"text", "table"} else "",
        "image_path": metadata.get("img_link"),
        "reference": build_reference_from_metadata(metadata),
        "score": score,
    }

def create_document_embeddings(
    pdf_dir: str,
    model,
    processor,
    batch_size: int = 2,
) -> list[dict[str, object]]:
    """
    Build multi-modal embeddings for every page of every PDF in a directory.

    Each entry in the returned list contains:
        - embedding: torch.Tensor on CPU
        - doc_id: int index of the PDF
        - page_id: int index of the page within the PDF
        - file_name: original PDF filename
    """
    images_per_pdf = convert_pdf_dir_to_images(pdf_dir)
    all_embeddings: list[dict[str, object]] = []

    for doc_idx, (filename, images) in enumerate(images_per_pdf.items()):
        dataloader = DataLoader(
            dataset=images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: processor.process_images(batch),
        )

        page_counter = 0
        for batch in tqdm(dataloader, desc=f"Processing {filename}"):
            with torch.no_grad():
                inputs = {k: v.to(model.device) for k, v in batch.items()}
                batch_embeddings = model(**inputs)
                cpu_embeddings = list(torch.unbind(batch_embeddings.to("cpu")))

            for embedding in cpu_embeddings:
                all_embeddings.append(
                    {
                        "embedding": embedding,
                        "doc_id": doc_idx,
                        "page_id": page_counter,
                        "file_name": filename,
                    }
                )
                page_counter += 1

    return all_embeddings


import stamina

@stamina.retry(on=Exception, attempts=3) # retry mechanism if an exception occurs during the operation
def upsert_to_qdrant(qdrant_client, collection_name, points):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
    except Exception as e:
        print(f"Error during upsert: {e}")
        return False
    return True

def colpali_qdrant(dataset, papers, doi, model, processor, qdrant_client, qdrant_collection, batch_size=4):

    with tqdm(total=len(dataset), desc="Indexing Progress") as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            # Extract images
            images = [item["image"] for item in batch]

            # Process and encode images
            with torch.no_grad():
                batch_images = processor.process_images(images).to(model.device)
                image_embeddings = model(**batch_images)

            # Prepare points for Qdrant
            points = []
            for j, embedding in enumerate(image_embeddings.embeddings):
                uuid_idx=[str(uuid.uuid4()) for _ in range(len(image_embeddings.embeddings))]
                points.append(
                    models.PointStruct(
                        id=uuid_idx[j],  # Use the batch index as the ID
                        vector=embedding.tolist(),  # Convert to list
                        payload={
                            "document_name": batch[j]["filename"],
                            "document_id": str(uuid.uuid4()),                            
                            "document_link": [doi for paper, doi in zip(papers, doi) if paper.split("/")[-1]  == batch[j]["filename"]][0],
                            "type": "pdf_page",
                            "page_no": batch[j]["page_no"],
                            "ref": "",
                            "caption":"",
                            "img_link": batch[j]["img_link"],
                        },  
                    )
                )

            # Upload points to Qdrant
            try:
                #upsert_to_qdrant(collection_name, points)
                qdrant_client.upsert(collection_name=qdrant_collection, points=points)
            except Exception as e:
                print(f"Error during upsert: {e}")
                continue

            # Update the progress bar
            pbar.update(batch_size)

    print("Indexing complete!")

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
#from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def retrieve_colpali (query, processor, model, qdrant_client, username, colection_name, top_k):

    with torch.no_grad():
        text_embedding = processor.process_queries([query]).to(model.device)  
        text_embedding = model(**text_embedding)

    token_query = text_embedding.embeddings[0].cpu().float().numpy().tolist()

    start_time = time.time()
    if username=="":
            query_result = qdrant_client.query_points(collection_name=colection_name,
                                        query=token_query,
                                        limit=top_k,
                                        search_params=models.SearchParams(
                                        quantization=models.QuantizationSearchParams(
                                        ignore=True,
                                        rescore=True,
                                        oversampling=2.0
                                        )
                                    )
                                )
    else:
        query_result = qdrant_client.query_points(collection_name=colection_name,
                                        query=token_query,
                                        limit=top_k,
                                        query_filter=models.Filter(
                                            must=[
                                                models.FieldCondition(
                                                    key="username",
                                                    match=models.MatchValue(
                                                        value=username,
                                                    ),
                                                )
                                            ]
                                        ),
                                        search_params=models.SearchParams(
                                        quantization=models.QuantizationSearchParams(
                                        ignore=True,
                                        rescore=True,
                                        oversampling=2.0
                                        )
                                    )
                                )

    print(f"Time taken = {(time.time()-start_time):.3f} s")
    return query_result


#from langchain.schema.retriever import BaseRetriever
from typing import TYPE_CHECKING, Any, Dict, List, Optional 
#from langchain.callbacks.manager import CallbackManagerForRetrieverRun

from langchain_core.retrievers import BaseRetriever
#from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import Runnable


# class CustomRetriever(BaseRetriever):
#     processor: Any = None
#     model: Any  =None
#     qdrant_client: Any = None
#     qdrant_embeddings: Any = None
#     colection_name:str = ""
#     top_k:int = 0
#     model_info:dict = {}
#     image_prompt1:str = ""
    
#     def __init__(self, processor, model, qdrant_client, qdrant_embeddings, colection_name,top_k, model_info,image_prompt1,  **kwargs):
#         super().__init__(**kwargs)
#         self.processor=processor
#         self.model=model
#         self.qdrant_client=qdrant_client
#         self.qdrant_embeddings= qdrant_embeddings
#         self.colection_name=colection_name
#         self.top_k=top_k
#         self.model_info=model_info
#         self.image_prompt1=image_prompt1


#     def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> str:
#         """
#         _get_relevant_documents is function of BaseRetriever implemented here

#         :param query: String value of the query

#         """
#         if self.processor != None:
#             retrieved_docs = retrieve_colpali (query, processor=self.processor, model=self.model, qdrant_client=self.qdrant_client, colection_name=self.colection_name, top_k=self.top_k)
#         else:
#             qdrant = QdrantVectorStore(
#                 client=self.qdrant_client,
#                 collection_name=self.colection_name,
#                 embedding=self.qdrant_embeddings
#             )

#             retrieved_docs = qdrant.similarity_search_with_score(query, self.top_k)

#         context_sum = summarise_context (retrieved_docs, self.model_info, self.image_prompt1, " ")

#         return context_sum

import argparse
import aiohttp
import asyncio
import os
#from time import time
import pickle
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, field_validator
from enum import Enum

def prompt_prep (docs, prompt_image, prompt_text):
    """
    Function that created summariey for images and text (or tables), using a prompt for images and a prompt for text
    the documents need to be in langchain docs format with specific metadata of type 
    """
            
    q_prompt = []
    for el in docs:
        #print(el)
        if el.metadata["type"] in ["image"]:         
            part_prompt = format_msgs(prompt_image,[el.metadata["img_link"]],"")
        elif el.metadata["type"] in ["text", "table"]:
            part_prompt = format_msgs(prompt_text,[],el.page_content)
        else:
            part_prompt = format_msgs(prompt_text,[],"")
        q_prompt.append(part_prompt)
        
    return q_prompt

import ssl

async def post_request_with_retries (session, url, headers, data, retries=5, backoff=1):
    #certificate_path = './src/GTS Root R4.pem'
    #ssl_context = ssl.create_default_context(cafile=certificate_path)

    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    choices = response_data.get('choices', [{}])
                    #text = choices[0]["logprobs"]
                    text = choices[0].get('message').get('content')
                    return text
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                await asyncio.sleep(backoff * (2 ** attempt))  # Exponential backoff
            else:
                return f"[error] Retries FAILED [error]."

async def get_responses (model, vllm_port, processed_prompts):

    if model.startswith("gpt"):
        url = f"https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ["OPENAI_API_KEY"]}",
                "Content-Type": "application/json"}
    else:
        url = f"http://localhost:{vllm_port}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ["VLLM_API_KEY"]}",
         "Content-Type": "application/json"}  
    

    # Create a TCPConnector with a limited number of connections

    conn = aiohttp.TCPConnector(limit=512)

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            post_request_with_retries(session, url=url, headers=headers, 
                data={
                "model": model,
                "messages": msg,
            })
            for msg in processed_prompts
        ]
        responses = await asyncio.gather(*tasks) 
        print(responses[0:4])

    return responses     

def delete_papers (username, vd_list, vd_colpali, file_loc, key_value:list, key_name="metadata.document_name",  key_link = "metadata.img_link"):
    """
    Fucntion deletes all elements from qdrant databases, both for standar normal and colpali vector databases, it removes all files stored in metadata.img_link and the pdf files itself, the default directory is ./papers 
    """    
    qdrant_client = QdrantClient(url="http://localhost:6333", api_key=os.environ["QDRANT_API_KEY"])



    img_list =[]
    for vd in vd_list:
        result = qdrant_client.scroll(
            collection_name=vd,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key_name,
                        match=models.MatchAny(any=key_value)
                    ),
                    models.FieldCondition(key="metadata.username", match=models.MatchValue(value=username),)
                ]
            ), limit=10000
        )
        
        lst = [res for res in result][0]
        links= [el.payload[key_link.split(".")[0]][key_link.split(".")[-1] ] for el in lst if el.payload[key_link.split(".")[0]][key_link.split(".")[-1] ]!='']
        img_list.extend(links)


    for vd in vd_colpali:
        result = qdrant_client.scroll(
        collection_name=vd,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=key_name.split(".")[-1],
                    match=models.MatchAny(any=key_value)
                    ),
                models.FieldCondition(key="username", match=models.MatchValue(value=username),)
                ]
            ), limit=10000
        )
        lst = [res for res in result][0]

        links1 = [el.payload[key_link.split(".")[-1]] for el in lst if el.payload[key_link.split(".")[-1]] != '']
        img_list.extend(links1)

    list_files= sorted(list(set(img_list)))

    for file in list_files:
        # If file exists, delete it.
        if os.path.isfile(file):
            os.remove(file)
        else:
            # If it fails, inform the user.
            print(f"Error: {file} file not found")

    for paper in key_value:
        file_loc_= file_loc +"papers/" + paper
        if os.path.isfile(file_loc_):
            os.remove(file_loc_)
        else:
            # If it fails, inform the user.
            print(f"Error: {file_loc} file not found")

    #log=[]
    for vd in vd_list:
        tmp_log = qdrant_client.delete(
                collection_name=vd,
                points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key_name,
                            match=models.MatchAny(any=key_value),
                        ),
                        models.FieldCondition(key="metadata.username", match=models.MatchValue(value=username),)
                    ],
                )
            ), 
        )   
        print(f"For VD {vd}, delete log shows_ {tmp_log}")
        #log.append(tmp_log)    

    #log1 = []
    for vd in vd_colpali:
        tmp_log1 = qdrant_client.delete(
            collection_name=vd,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key_name.split(".")[-1],
                            match=models.MatchAny(any=key_value),
                        ),
                        models.FieldCondition(key="username", match=models.MatchValue(value=username),)
                    ],
                )
            ), 
        )
        print(f"For VD {vd}, delete log shows_ {tmp_log1}")
        #log1.append(tmp_log1)     

def get_vd_elements (qdrant_client, username, vd_name, paper_dir):

    #PAPERS_DIR ="./papers/"
    papers = [os.path.join(paper_dir, f) for f in sorted(os.listdir(paper_dir)) if f.lower().endswith('.pdf')]

    retrived_elmns = qdrant_client.scroll(
        collection_name=vd_name,
        scroll_filter=models.Filter(
            must_not=[
                models.FieldCondition(key="metadata.document_name", match=models.MatchValue(value="")),
                
            ],
            #must=[
            #    models.FieldCondition(key="metadata.username", match=models.MatchValue(value=username),)
            #],
        ),
        limit=100000,
        with_payload=True,
    )

    lst = []
    for el in retrived_elmns[0]:
        dct= {k:v for k,v in el.payload["metadata"].items() if k in ["document_name","document_link"]}
        lst.append(dct)
    lst = [dict(t) for t in {tuple(d.items()) for d in lst}]
    lst = sorted(lst, key=lambda d: d['document_name'])

    dt = [el["document_name"] for el in lst]
    doi_links = [el["document_link"] for el in lst]

    links = [paper for el in dt for paper in papers if el in paper]

    return dt, links, doi_links

def get_vd_elements_colpali (qdrant_client, username, vd_name, paper_dir):

    #PAPERS_DIR ="./papers/"
    papers = [os.path.join(paper_dir, f) for f in sorted(os.listdir(paper_dir)) if f.lower().endswith('.pdf')]

    retrived_elmns = qdrant_client.scroll(
        collection_name=vd_name,
        scroll_filter=models.Filter(
            must_not=[
                models.FieldCondition(key="metadata.document_name", match=models.MatchValue(value="")),
                
            ],
            must=[
                models.FieldCondition(key="metadata.username", match=models.MatchValue(value=username),)
            ],
        ),
        limit=100000,
        with_payload=True,
    )

    lst = []
    for el in retrived_elmns[0]:
        dct= {k:v for k,v in el.payload.items() if k in ["document_name","document_link"]}
        lst.append(dct)
    lst = [dict(t) for t in {tuple(d.items()) for d in lst}]
    lst = sorted(lst, key=lambda d: d['document_name'])

    dt = [el["document_name"] for el in lst]
    doi_links = [el["document_link"] for el in lst]

    links = [paper for el in dt for paper in papers if el in paper]

    return dt, links, doi_links

import copy

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    print(messages)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

async def get_img_summary (docs_multi, prompts, model, vllm_port, save_output):

    tmp_docs = copy.deepcopy(docs_multi)
    list_img = [el for el in tmp_docs if el.metadata["type"] in ["image"]]

    if list_img ==[]:
        return tmp_docs
    
    idx_img, img_docs = zip(*[[idx, el] for idx, el in enumerate(tmp_docs) if el.metadata["type"] in ["image"]]) 

    processed_prompts = prompt_prep (img_docs, prompts['img_summary'], prompts['text_summary'])

    # if model.startswith("Qwen"):
    #     os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    #     processor = AutoProcessor.from_pretrained(model)
    #     processed_prompts  = [prepare_inputs_for_vllm(message, processor) for message in  processed_prompts ]

    processed_out = await get_responses (model, vllm_port, processed_prompts)

    #out_docs = [x for x in docs_multi]
    #for idx_, el in zip(idx_img, processed_out):
    #    out_docs[idx_].page_content = el

    out_docs = []
    for idx_el, element in enumerate(tmp_docs):
        tmp=element
        if idx_el in idx_img:
            tmp.page_content = [el for id, el in zip(idx_img,processed_out) if idx_el == id ][0]
        out_docs.append(tmp)


    if save_output != "":
        with open(save_output, 'wb') as file:
            pickle.dump( out_docs, file)
    
    return out_docs

async def process_models (processed_multi, prompts, MODELS):

    #orig_data = copy.deepcopy(processed_multi)
    dict_out= {"orig_model":processed_multi}

    for model in MODELS:
        task1 = asyncio.create_task(get_img_summary (dict_out["orig_model"], prompts, model["model_name"], model["port"], ""))
        dict_out[model["model_short"]]= await task1

    return dict_out

def qdrant_process(docs, qdrant_client, vec_db, emb_dim, embeddings, url="http://localhost:6333"):

    print(f"Processing data for colection {vec_db}.")

    if not qdrant_client.collection_exists(collection_name=vec_db):
        qdrant_client.create_collection(
                collection_name=vec_db,
                on_disk_payload=True,
                vectors_config= models.VectorParams(
                    size=emb_dim,
                    on_disk=True, 
                    distance=models.Distance.COSINE
                    ),
                )

    QdrantVectorStore.from_documents(
    docs,
    embedding=embeddings,
    url=url,
    #client=qdrant_client,
    collection_name=vec_db,
    retrieval_mode=RetrievalMode.DENSE,
    )
    print(f"Processing of {len(docs)} for colection {vec_db} complete.")


def pdf_loader (papers, doi_links, filenames, vd_dir, vd_tokenizer, username=""):

    conversion = []
    for paper, doi_paper, filename in zip(papers, doi_links, filenames):

        ocr=check_ocr(paper)
        conversion.append({"filename": filename, "document": doc_conv(ocr = ocr).convert(source=paper).document, "link": doi_paper})
    
    if username != "":
        mm_dir=f"./src/vectordb/user_data/{username}/"
        pg_image=False
    else:
        mm_dir=""
        pg_image=True

    processed_multi = data_preparation (conversion, vd_dir, vd_tokenizer=vd_tokenizer, mm_dir = mm_dir,only_text=False, page_images=pg_image)

    processed_text = data_preparation (conversion, vd_dir, vd_tokenizer=vd_tokenizer, mm_dir= mm_dir,only_text=True, page_images=pg_image)

    return processed_multi, processed_text

def conv_docs1(file, file_path, port_docling, use_gemma=False):
    """
    Asynchronously sends a file to a local server for conversion.

    Returns:
        dict or None: The JSON response from the server if successful, None otherwise.
    """
            #use_gemma currently not working
    url = f"http://localhost:{port_docling}/v1/convert/file"
    parameters = {
    "from_formats": ["docx", "pptx", "html", "image", "pdf", "asciidoc", "md", "xlsx"],
    "to_formats": ["json"],
    "do_ocr": True,
    "force_ocr": False,
    "ocr_engine": "easyocr",
    "ocr_lang": ["en"],
    "pdf_backend": "dlparse_v4",
    "table_mode": "accurate",
    "do_table_structure": True,
    "abort_on_error": False,
    "include_images": True,
    "images_scale": 2.0,
    #"do_formula_enrichment": True,
    #"do_picture_description": True,
    #"picture_description_api": prep_descr_api
    }

    if use_gemma:
        prep_descr_api = {
            "url": "http://localhost:8006/v1/chat/completions",
            "headers": {
                "Authorization": f"Bearer {os.environ["VLLM_API_KEY"]}",
                "Content-Type": "application/json"
                },
            "params": { "model": "google/gemma-3-27b-it" },
            "timeout": 300,
            "prompt": "Describe this image in a few sentences."
        }
        parameters["do_picture_description"] = True 
        parameters["picture_description_api"] = prep_descr_api 

    #current_dir = os.path.dirname(__file__)
    #file_path1 = os.path.join(current_dir, file)
    files = {
        'files': (file, open(file_path+file, 'rb'), 'application/pdf'),
    }
    with httpx.Client(timeout=2399.0) as client:
        i=0
        while(i<10):
            try:
                response = client.post(url, files=files, data=parameters)
                response.raise_for_status()  # Raise an exception for bad status codes
                data = response.json()
                client.close()
                return DoclingDocument.model_validate(data["document"]["json_content"])

            except httpx.HTTPError as e:
                print(f"HTTP error occurred: {e}")
                i += 1
                #return None
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
                print(f"Raw response: {response.text}")
                i += 1
                #return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                i += 1
                #return None
            except i==10:
                print(f"Failed to process {file} after multiple attempts.")
                return None


def pdf_loader1 (filenames, doi_links, papers_dir, vd_dir, port_docling, vd_tokenizer, username=""):

    conversion = []
    for filename, doi_paper in zip(filenames, doi_links):
        print(f"Processing: {filename}")
        conversion.append({"filename": filename, "document": conv_docs1(filename, papers_dir, port_docling, use_gemma=False), "link": doi_paper})
    
    if username != "":
        mm_dir=f"./src/vectordb/user_data/{username}/"
        pg_image=False
    else:
        mm_dir=""
        pg_image=False
    #print(conversion[0])
    processed_multi = data_preparation (conversion, vd_dir, vd_tokenizer=vd_tokenizer, mm_dir = mm_dir,only_text=False, page_images=pg_image)

    processed_text = data_preparation (conversion, vd_dir, vd_tokenizer=vd_tokenizer, mm_dir= mm_dir,only_text=True, page_images=pg_image)

    return processed_multi, processed_text


async def process_and_add (papers,  doi_links, filenames, vd_dir, emb_tokenizer, models, prompts, embeddings, username="", batch_size=4):
    processed_multi, processed_text = pdf_loader1 (papers, doi_links, filenames, vd_dir, emb_tokenizer, username =username)

    processed_multi, processed_text = pdf_loader1 (papers, doi_papers, PAPERS_DIR, VD_DIR, 5001, emb_tokenizer, username="primozk1")
    #dataset = convert_pdfs_to_images(papers, vd_dir + "/pg_images")
    dct = await process_models (processed_multi, prompts, models)
    dct["text_only"] = processed_text
    if username !="":
        dct_modified = update_vd_new_user(username, models, orig_dict="./src/vectordb/context_proces_25_files_2025-04-11.pkl",  RAG_text="text_only", orig_pref="./src/vectordb/", local_dict=dct)
        setup_initial_vector_db (username, models, embeddings, local_dict= dct_modified)

    
def get_colpali (model_id = "vidore/colpali-v1.3-hf"):
    

    #model_id = "vidore/colpali-v1.3-hf"

    model = ColPaliForRetrieval.from_pretrained(
        model_id,
        torch_dtype=torch.float,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()

    processor = ColPaliProcessor.from_pretrained(model_id)

    return model, processor

def prompt_prep_query (query, prompts, qdrant_client, username,  vector_db,  embeddings, top_k, type, cp_model, cp_processor, join_context=False):
    """
    preparing prompts from retrival - for each question get TOP_K as context send it to vision LLM
    list of lists
    """
    if type == "mm_RAG":
        qdrant = QdrantVectorStore(
            client=qdrant_client,
            collection_name=vector_db,
            embedding=embeddings,
        )
    elif type == "colpali":
        print(f"Colpali processor and model should be loaded!")
        # model_id = "vidore/colpali-v1.3-hf"

        # cp_model = ColPaliForRetrieval.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.float,
        #     device_map="cuda:0",  # or "mps" if on Apple Silicon
        # ).eval()

        # cp_processor = ColPaliProcessor.from_pretrained(model_id)
    elif type == "":
        print(f"Inference w/o RAG will be used!")
    else:
        print(f"Error, either enter mm_RAG or colpali or '' for RAG variable")


    prompt_image = prompts.format(query=query)
    prompt_text = prompts.format(query=query)
    #sudo = prompts["text_summary_query"].format(query=query)
    #context = []
    #q_prompt = []
    if type in ["", "mm_RAG"]:
        
        if type == "":
            context = []
        else:    
            context = qdrant.similarity_search_with_score(query, top_k,
                     filter=models.Filter(must=[models.FieldCondition(key="metadata.username", match=models.MatchValue(value=username),)]))
            
        if join_context== False:
            q_prompt = []
            for el in context:
                if el[0].metadata["type"] in ["image"]:         
                    part_prompt = format_msgs(prompt_image,[el[0].metadata["img_link"]],"")
                elif el[0].metadata["type"] in ["text", "table"]:
                    part_prompt = format_msgs(prompt_text,[],el[0].page_content)
                else:
                    part_prompt = format_msgs(prompt_text,[],"")
                q_prompt.append(part_prompt)
        else:
            img_links = [el[0].metadata["img_link"] for el in context if el[0].metadata["type"] in ["image"]]
            text_joined = []
            text_joined = [el[0].page_content for el in context if el[0].metadata["type"] in ["text", "table"]]
            text_joined = "\n".join(text_joined)
            if len(text_joined) ==0:
                text_joined=""
            
            q_prompt = format_msgs(prompt_image, img_links, text_joined )
        
    elif type == "colpali" and cp_processor != "" and cp_model != "":
        #colpali
        context = retrieve_colpali (query, cp_processor, cp_model, qdrant_client, username, vector_db, top_k)
        if join_context== False:
            q_prompt = []
            for el in context.points:
                part_prompt = format_msgs(prompt_image,[el.payload["img_link"]],"")
                q_prompt.append(part_prompt)
        else:
            img_links = [el.payload["img_link"] for el in context.points]
            q_prompt = format_msgs(prompt_image, img_links, "")

    else:
        print(f"Error, either enter mm_RAG or colpali or '' for RAG variable")

    
        
    return {"query": query, "context": context, "q_prompts": q_prompt}

def prompt_prep_query_emb (query, prompts, qdrant_client, username,  vector_db,  embed_prompt, top_k, type, join_context=False):
    """
    preparing prompts from retrival - for each question get TOP_K as context send it to vision LLM
    list of lists
    """

    prompt_image = prompts['rag_summary_query'].format(query=query)
    prompt_text = prompts['text_summary_query'].format(query=query)
    #sudo = prompts["text_summary_query"].format(query=query)
    #context = []
    #q_prompt = []
    if type in ["", "mm_RAG"]:
        
        if type == "":
            context = []
            return {"query": query, "context": "", "q_prompts": format_msgs(prompt_text,[],"")}
        else:    
            context = qdrant_client.query_points(
                collection_name=vector_db,
                query=embed_prompt,
                limit = top_k,)

                #query_filter=models.Filter(must=[models.FieldCondition(key="metadata.username", match=models.MatchValue(value=username),)]))
        #print(context.points[0].payload['metadata'])
        if join_context== False:
            q_prompt = []
            for el in context.points:
                if el.payload['metadata']["type"] in ["image"]:         
                    part_prompt = format_msgs(prompt_image,[el.payload['metadata']["img_link"]],"")
                elif el.payload['metadata']["type"] in ["text", "table"]:
                    part_prompt = format_msgs(prompt_text,[],el.payload["page_content"])
                else:
                    part_prompt = format_msgs(prompt_text,[],"")
                q_prompt.append(part_prompt)
        else:
           
            img_links = [el.payload['metadata']["img_link"] for el in context.points if el.payload['metadata']["type"] in ["image"]]
            text_joined = []
            text_joined = [el.payload["page_content"] for el in context.points if el.payload['metadata']["type"] in ["text", "table"]]
            text_joined = "\n".join(text_joined)
            if len(text_joined) ==0:
                text_joined=""
            
            q_prompt = format_msgs(prompt_image, img_links, text_joined )
        
    else:
        print(f"Error, either enter mm_RAG or '' for RAG variable")

    
        
    return {"query": query, "context": context, "q_prompts": q_prompt}

def prompt_prep_query1 (query, prompts,  username,  vector_db,  embeddings, top_k, type, join_context=False):
    """
    preparing prompts from retrival - for each question get TOP_K as context send it to vision LLM
    list of lists
    """


    prompt_query = prompts.format(query=query)

    if type == "":
        context = []
    elif type in ["mm_vd", "text_vd"]:    
        try:
            qdrant = QdrantVectorStore.from_existing_collection(
                url="http://localhost:6333",
                api_key=os.environ["QDRANT_API_KEY"],
                collection_name=vector_db,
                embedding=embeddings,
            )

            context = qdrant.similarity_search_with_score(query, top_k,
                        filter=models.Filter(must=[models.FieldCondition(key="metadata.username", match=models.MatchValue(value=username),)]))
        except:
            context = []
            print(f"Error accessing qdrant vectorstore")
    else:
        context = []
        print(f"Error, either enter mm_RAG or colpali or '' for RAG variable")
            
    if join_context== False:
        q_prompt = []
        for el in context:
            if el[0].metadata["type"] in ["image"]:         
                part_prompt = format_msgs(prompt_query ,[el[0].metadata["img_link"]],"")
            elif el[0].metadata["type"] in ["text", "table"]:
                part_prompt = format_msgs(prompt_query,[],el[0].page_content)
            else:
                part_prompt = format_msgs(prompt_query,[],"")
            q_prompt.append(part_prompt)
    else:
        if context==[]:
            img_links=[]
        else:
            img_links = [el[0].metadata["img_link"] for el in context if el[0].metadata["type"] in ["image"]]
        text_joined = []
        text_joined = [el[0].page_content for el in context if el[0].metadata["type"] in ["text", "table"]]
        text_joined = "\n".join(text_joined)
        if len(text_joined) ==0:
            text_joined=""
        
        q_prompt = format_msgs(prompt_query, img_links, text_joined )


        
    return {"query": query, "context": context, "q_prompts": q_prompt}

async def post_request_with_retries (session, url,headers, data, retries=10, backoff=1):
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    choices = response_data.get('choices', [{}])
                    #text = choices[0]["logprobs"]
                    text = choices[0].get('message').get('content')
                    return text
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                await asyncio.sleep(backoff * (2 ** attempt))  # Exponential backoff
            else:
                return f"[error] {response.status}\n{data}."


async def get_response_context(query, context, model_name, url, headers):
    conn = aiohttp.TCPConnector(limit=512)

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            post_request_with_retries(session, url=url, headers=headers, 
                data={
                "model": model_name,
                "messages": msg,
            })
            for msg in context
        ]
        responses = await asyncio.gather(*tasks)

        fin_query = format_msgs(query+"Here is context information:",[],"\n".join(responses))

    conn = aiohttp.TCPConnector(limit=512)

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            post_request_with_retries(session, url=url, headers=headers,
                data={
                "model": model_name,
                "messages": msg,
                #"max_tokens":1,
                #"logprobs": True,
                #"top_logprobs":50,
                #"extra_body":{"guided_json": json_schema},
                ##"extra_body":{"guided_choice": ["A", "B", "C", "D"]},
                #"response_format": type_to_response_format_param(MCQ)
            })
            for msg in fin_query
        ]
        answer = await asyncio.gather(*tasks)

    return answer

def response_real_out (response, perm_q):

    ans_list= ["A", "B", "C", "D"]

    #for idx, (el,q_order) in enumerate(for_filter): 
    if response is None:
        return "", ""
    if response in ans_list:

        real_answer=[ans_list[el] for idx, el in enumerate(perm_q) if ans_list[idx]== response][0]
        return response, real_answer       
    try:
        tmp = json.loads(response)
        match = re.search('^*(A|B|C|D)(\\s|.)', tmp)

        if match and match.group(1) in ans_list:
            
            resp =match.group(1)
            
            real_answer=[ans_list[el] for idx, el in enumerate(perm_q) if ans_list[idx]== resp][0]
            #print(real_answer)
            return resp, real_answer
        else:
            return "", ""
    except:
        pass
    try:
        tt = ' '.join(response.split())
        tt=tt.split(":")[-1][:10]
        tt=tt.upper()[:20]
 
        match = re.search('(A|B|C|D)(\\s|.)',tt)

        if match and match.group(1) in ans_list:
            
            resp = match.group(1)
            real_answer=[ans_list[el] for idx, el in enumerate(perm_q) if ans_list[idx]== resp][0]
            #print(real_answer)
            return resp, real_answer
        else:
            return "", ""
    except:
        return "", ""
    
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_password_email(sender_email, sender_password, recipient_username, recipient_email, password):
    # Gmail SMTP server configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    # Email content
    subject = 'SynHealth app testing: new password'
    body = f'''
    Dear {recipient_username},

    Your password has been reset and your login credentials are:
    
    username: {recipient_username} 
    password: {password}

    You can change your password in the user settings.
    
    '''

    # Create a multipart email message
    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = recipient_email

    # Attach the email body as plain text
    message.attach(MIMEText(body, 'plain'))

    try:
        # Establish a secure connection to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        # Login to the SMTP server with the application-specific password
        server.login(sender_email, sender_password)
        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())
        # Close the SMTP connection
        server.quit()
        print('Email sent successfully.')
    except Exception as e:
        print(f'Error occurred while sending the email: {e}')


def update_vd_new_user(username, MODELS, local_dict="", orig_dict="./src/vectordb/context_proces_25_files_2025-04-11.pkl",  RAG_text="text_only", orig_pref="./src/vectordb/"):
    
    new_pref = orig_pref + f"user_data/{username}/"
    
    if not os.path.exists(new_pref ):
        os.makedirs(new_pref )
        os.makedirs(new_pref + "/papers/" )
        os.makedirs(new_pref + "/images/" )
        os.makedirs(new_pref + "/tables/" )

    if local_dict != "":
        mod_dict=local_dict
    else:
        with open(orig_dict, 'rb') as file: 
            mod_dict = pickle.load(file)

    with open( orig_pref + "prompts_used.pkl", 'rb') as file: 
        prompts = pickle.load(file)
        
    with open(new_pref + "prompts_used.pkl" , 'wb') as file: 
        pickle.dump(prompts, file)    


    dct_outcome ={}



    for idx,model in enumerate(MODELS):
        new_vd=[]
        for el in mod_dict[model["model_short"]]:
            tmp=el
            tmp.metadata["username"]=username
            if tmp.metadata["img_link"]!="":
                new_info = new_pref + "/".join(tmp.metadata["img_link"].rsplit("/",2)[1:])
                tmp.metadata["img_link"] = new_info
                 
            new_vd.append(tmp)
        dct_outcome[model["model_short"]]=new_vd
        if idx ==0:
            new_vd=[]
            for el in mod_dict[RAG_text]:
                tmp=el
                tmp.metadata["username"]=username
                new_vd.append(tmp)
            dct_outcome[RAG_text]=new_vd

    return dct_outcome

def make_tarfile(output_filename, source_dir):
    subprocess.call(["tar", "-C", source_dir, "-zcvf", output_filename, "."])

def extract_tarfile(input_filename, output_dir):
    subprocess.call(["tar", "-C", output_dir, "-xzf", input_filename, "."])

def new_user_set_files (username, input_filename="./src/vectordb/context_25_files_2025-04-11.tar.gz"):
    output_dir=f"./src/vectordb/user_data/{username}/"
    extract_tarfile(input_filename, output_dir)

def setup_initial_vector_db (username, models, local_dict="", model1 ="BAAI/bge-base-en-v1.5", port=7997, chunk_size_emb=100, batch_size=1000, EMB_DIM = 768):
    
    if local_dict != "":
        user_dct=local_dict
    else:
        user_dct = update_vd_new_user(username, models)

    for idx, model in enumerate(delte):
        if idx == 0:
            if not qdrant_client.collection_exists(collection_name=model["text_vd"]):
                qdrant_client.create_collection(
                        collection_name=model["text_vd"],
                        on_disk_payload=True,
                        vectors_config= models.VectorParams(
                            size=EMB_DIM,
                            on_disk=True, 
                            distance=models.Distance.COSINE
                            ),
                )

            print(f"Processing model {model['model_short']} for collection {model['text_vd']}")
            elements = [el.page_content for el in user_dct["text_only"]]
            embeds = get_embeddings_api(elements, model1, port, chunk_size=chunk_size_emb)
            uuid_idx=[str(uuid.uuid4()) for _ in range(len(embeds))]

            emd_list = [embeds[i:i + batch_size] for i in range(0, len(embeds), batch_size)]
            payload_list = [user_dct["text_only"][i:i + batch_size] for i in range(0, len(user_dct["text_only"]), batch_size)]
            rnd_id_list = [uuid_idx[i:i + batch_size] for i in range(0, len(uuid_idx), batch_size)]

            for embs, paylds, rnd_id_lst in zip(payload_list, emd_list, rnd_id_list):
                qdrant_client.upsert(
                collection_name=model["text_vd"],
                points=[
                    models.PointStruct(
                        id=rnd_id, 
                        vector=vector, 
                        payload={"page_content": payload.page_content, 
                                "metadata": payload.metadata }
                    ) for payload, vector, rnd_id in zip(embs, paylds, rnd_id_lst)
                    ],
                )

        print(f"Processing model {model['model_short']} for collection {model['mm_vd']}")
        elements = [el.page_content for el in user_dct[model["model_short"]]]
        embeds = get_embeddings_api(elements, model1, port, chunk_size=chunk_size_emb)
        uuid_idx=[str(uuid.uuid4()) for _ in range(len(embeds))]

        if not qdrant_client.collection_exists(collection_name=model["mm_vd"]):
            qdrant_client.create_collection(
                    collection_name=model["mm_vd"],
                    on_disk_payload=True,
                    vectors_config= models.VectorParams(
                        size=EMB_DIM,
                        on_disk=True, 
                        distance=models.Distance.COSINE
                        ),
            )
      
        emd_list = [embeds[i:i + batch_size] for i in range(0, len(embeds), batch_size)]
        payload_list = [user_dct[model["model_short"]][i:i + batch_size] for i in range(0, len(user_dct[model["model_short"]]), batch_size)]
        rnd_id_list = [uuid_idx[i:i + batch_size] for i in range(0, len(uuid_idx), batch_size)]

        for embs, paylds, rnd_id_lst in zip(payload_list, emd_list, rnd_id_list):
            qdrant_client.upsert(
            collection_name=model["mm_vd"],
            points=[
                models.PointStruct(
                    id=rnd_id, 
                    vector=vector, 
                    payload={"page_content": payload.page_content, 
                            "metadata": payload.metadata }
                ) for payload, vector, rnd_id in zip(embs, paylds, rnd_id_lst)
                ],
            )        
        if local_dict=="":
            new_user_set_files (username)


    print(f"initial vectorDB set up for use {username}.")

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import DescrStatsW
import glob
import ast

DEFAULT_PRICES_GPT = [
    {"model": "gpt-5", "price_1M_output": 10, "price_1M_input": 1.25},
    {"model": "gpt-5-mini", "price_1M_output": 2, "price_1M_input": 0.25},
    {"model": "gpt-5-nano", "price_1M_output": 0.4, "price_1M_input": 0.05},
]
DEFAULT_PRICE_DICT = {p['model']: p['price_1M_output'] for p in DEFAULT_PRICES_GPT}

def get_metric_descriptions(top_k):
    """
    Returns a description dictionary for the key metrics given a Precision@k label.
    """
    precision_label = f'P@{top_k}'
    return {
        "Cor_answer": "Average correctness rate per question",
        "Elapsed": "Average wall-clock time per question (seconds)",
        "Total_tokens": "Average total tokens consumed per question",
        precision_label: f"Precision@{top_k}: share of retrieved documents containing the reference paper",
        "Throughput": "Average tokens processed per second",
        "Cost": "USD spent per iteration/run",
        "Price-per-cost": "Cents spent per correct answer",
    }

METRIC_DESCRIPTIONS = get_metric_descriptions(5)

__all__ = [
    "DEFAULT_PRICES_GPT",
    "DEFAULT_PRICE_DICT",
    "METRIC_DESCRIPTIONS",
    "get_metric_descriptions",
    "proportion_ci",
    "mean_confidence_interval",
    "merge_data",
    "run_analysis",
    "run_ci_summary",
]

def proportion_ci(series):
    """
    Calculates the Agresti-Coull confidence interval for a proportion.
    """
    count = series.sum()
    nobs = series.count()
    if nobs == 0:
        return np.nan, np.nan
    ci_low, ci_upp = proportion_confint(count, nobs, method='agresti_coull')
    return ci_low, ci_upp

def mean_confidence_interval(series, non_negative=False):
    """
    Calculates the confidence interval for a mean.
    If non_negative is True, the lower bound of the CI is clipped at 0.
    """
    if series.count() < 2:
        return np.nan, np.nan
    ci_low, ci_upp = DescrStatsW(series.dropna()).tconfint_mean(alpha=0.05, alternative='two-sided')
    if non_negative:
        ci_low = max(0, ci_low)
    return ci_low, ci_upp

def _format_ci_cell(mean_val, low_val, upp_val, decimals=3):
    """Formats a mean/CI triplet for display."""
    if pd.isna(mean_val) or pd.isna(low_val) or pd.isna(upp_val):
        return 'N/A'
    low_val = max(0, low_val)
    value_fmt = f'{{:.{decimals}f}}'
    return f"{value_fmt.format(mean_val)}\n[{value_fmt.format(low_val)}, {value_fmt.format(upp_val)}]"

def _attach_ci_bounds(df, ci_col, low_col, upp_col):
    """
    Expands a CI column that stores tuples into two numeric columns.
    """
    if ci_col not in df.columns:
        df[low_col] = np.nan
        df[upp_col] = np.nan
        return df

    if df.empty:
        df[low_col] = pd.Series(dtype=float)
        df[upp_col] = pd.Series(dtype=float)
        return df.drop(columns=[ci_col])

    bounds = df[ci_col].apply(
        lambda val: val if isinstance(val, tuple) and len(val) == 2 else (np.nan, np.nan)
    )
    df[low_col] = bounds.apply(lambda val: val[0])
    df[upp_col] = bounds.apply(lambda val: val[1])
    return df.drop(columns=[ci_col])

def build_ci_metric_specs(precision_label):
    """
    Builds the CI metric configuration for the requested Precision@k label.
    """
    return [
        {
            "display": "Cor_answer",
            "source_col": "mean_cor_answer",
            "mean_col": "mean_cor_answer",
            "ci_col": "ci_cor_answer",
            "ci_func": proportion_ci,
            "decimals": 3,
        },
        {
            "display": "Elapsed",
            "source_col": "mean_elapsed",
            "mean_col": "mean_elapsed",
            "ci_col": "ci_elapsed",
            "ci_func": mean_confidence_interval,
            "decimals": 2,
        },
        {
            "display": "Total_tokens",
            "source_col": "mean_tokens",
            "mean_col": "mean_tokens",
            "ci_col": "ci_tokens",
            "ci_func": mean_confidence_interval,
            "decimals": 1,
        },
        {
            "display": precision_label,
            "source_col": "mean_precision",
            "mean_col": "mean_precision",
            "ci_col": "ci_precision",
            "ci_func": mean_confidence_interval,
            "decimals": 3,
        },
        {
            "display": "Throughput",
            "source_col": "mean_throughput",
            "mean_col": "mean_throughput",
            "ci_col": "ci_mean_throughput",
            "ci_func": lambda s: mean_confidence_interval(s, non_negative=True),
            "decimals": 1,
        },
        {
            "display": "Cost",
            "source_col": "sum_cost",
            "mean_col": "mean_sum_cost",
            "ci_col": "ci_mean_sum_cost",
            "ci_func": lambda s: mean_confidence_interval(s, non_negative=True),
            "decimals": 2,
        },
        {
            "display": "Price-per-cost",
            "source_col": "price_per_cost",
            "mean_col": "mean_price_per_cost",
            "ci_col": "ci_mean_price_per_cost",
            "ci_func": lambda s: mean_confidence_interval(s, non_negative=True),
            "decimals": 2,
        },
    ]

def calculate_throughput(df):
    """Calculates tokens processed per second."""
    return df['Total_tokens'] / df['Elapsed']

def calculate_latency(df):
    """Backward-compatible alias for calculate_throughput."""
    return calculate_throughput(df)

def calculate_precision_at_k(row, top_k=10):
    """
    Calculates Precision@k for the provided row.
    """
    paper_id_val = str(row['Paper_id'])
    if not paper_id_val.startswith('Paper'):
        return np.nan

    paper_id = paper_id_val.lower()
    context_papers = row['Context_papers']

    if pd.isna(context_papers) or not isinstance(context_papers, str) or not context_papers.startswith('['):
        return 0

    try:
        context_papers_list = ast.literal_eval(context_papers)
    except (ValueError, SyntaxError):
        return 0
    
    nr_el = sum([1 for el in context_papers_list if paper_id == str(el).split('_pg_')[0].lower()])
    
    return nr_el / top_k

def calculate_is_paper_id_in_context(row, top_k=10):
    """Backward-compatible alias for calculate_precision_at_k."""
    return calculate_precision_at_k(row, top_k=top_k)

def calculate_cost(df, price_dict):
    """Calculates USD cost for each row."""
    def calc_row(row):
        model = row['Model']
        total_tokens = row['Total_tokens']
        price_1M = price_dict.get(model)
        if price_1M is not None:
            return (total_tokens / 1_000_000) * price_1M
        return np.nan
    return df.apply(calc_row, axis=1)

def calculate_price(df, price_dict):
    """Backward-compatible alias for calculate_cost."""
    return calculate_cost(df, price_dict)

def create_summary_table(df, group_by, analysis_vars, price_dict=None, return_numeric=False):
    """
    Groups a DataFrame and calculates mean and confidence intervals for specified variables.
    This function performs a two-step aggregation (per-question, then final) to provide more robust CIs.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_by (list): A list of column names to group by.
        analysis_vars (dict): A dictionary where keys are variable names and values are agg types.
        price_dict (dict, optional): A dictionary for price calculation.
        return_numeric (bool): If True, return df with numeric mean, ci_low, ci_upp columns.

    Returns:
        pd.DataFrame: A summary DataFrame.
    """
    
    df_copy = df.copy()

    # Pre-calculations
    for var in analysis_vars:
        if var not in df_copy.columns:
            if var == 'Latency':
                df_copy['Latency'] = calculate_latency(df_copy)
            elif var == 'is_paper_id_in_context':
                df_copy['is_paper_id_in_context'] = df_copy.apply(calculate_is_paper_id_in_context, axis=1)
            elif var == 'Price' and price_dict:
                df_copy['Price'] = calculate_price(df_copy, price_dict)

    # Step 1: Aggregate per question
    per_question_group_by = group_by + ['Question_nr']
    vars_to_agg = list(analysis_vars.keys())
    per_question_agg_funcs = {var: 'mean' for var in vars_to_agg}
    
    cols_for_grouping = list(set(per_question_group_by + vars_to_agg))
    df_for_agg = df_copy[cols_for_grouping]
    
    per_question_df = df_for_agg.groupby(per_question_group_by, observed=True).agg(per_question_agg_funcs).reset_index()

    # Step 2: Aggregate over questions
    agg_funcs = {}
    for var, agg_type in analysis_vars.items():
        is_non_negative = var in ['Latency', 'Price']
        agg_funcs[f'mean_{var}'] = (var, 'mean')
        if agg_type == 'proportion':
            agg_funcs[f'ci_{var}'] = (var, proportion_ci)
        else: # 'mean'
            agg_funcs[f'ci_{var}'] = (var, lambda s, v=var: mean_confidence_interval(s, non_negative=(v in ['Latency', 'Price'])))

    grouped = per_question_df.groupby(group_by, observed=True)
    agg_df = grouped.agg(**agg_funcs)

    # Format output
    for var in vars_to_agg:
        agg_df[f'ci_low_{var}'], agg_df[f'ci_upp_{var}'] = zip(*agg_df[f'ci_{var}'])
    
    if return_numeric:
        # Drop the original CI tuple column and return numeric values
        return agg_df.drop(columns=[f'ci_{var}' for var in vars_to_agg])

    for var in vars_to_agg:
        mean_col = f'mean_{var}'
        ci_low_col = f'ci_low_{var}'
        ci_upp_col = f'ci_upp_{var}'
        
        agg_df[var] = agg_df[mean_col].round(3).astype(str) + ' [' + agg_df[ci_low_col].round(3).astype(str) + '-' + agg_df[ci_upp_col].round(3).astype(str) + ']'
        agg_df = agg_df.drop(columns=[mean_col, f'ci_{var}', ci_low_col, ci_upp_col])

    return agg_df

def merge_data(path):
    """
    Merges CSV files from a given path and adds an 'Iteration' column.
    """
    all_files = glob.glob(path + '*.csv')
    dfs = []
    iteration_counts = {}
    for f in all_files:
        df = pd.read_csv(f)
        if not df.empty:
            model_combo = (df['Model'].iloc[0], df['Model_ret'].iloc[0])
            iteration_counts.setdefault(model_combo, 0)
            iteration_counts[model_combo] += 1
            df['Iteration'] = iteration_counts[model_combo]
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def run_analysis(path, group_by_cols, analysis_vars_dict, price_dict):
    """
    Runs the streamlined 2-stage analysis over iterations.
    """
    # 1. Merge data
    merged_df = merge_data(path)
    if merged_df.empty:
        print("No data found in path:", path)
        return pd.DataFrame(), pd.DataFrame()

    # Set categorical order
    if 'Model' in merged_df.columns:
        # Get all unique models and order them
        all_models = merged_df['Model'].unique()
        merged_df['Model'] = pd.Categorical(merged_df['Model'], categories=sorted(all_models), ordered=True)
    if 'Model_ret' in merged_df.columns:
        all_ret_models = merged_df['Model_ret'].unique()
        merged_df['Model_ret'] = pd.Categorical(merged_df['Model_ret'], categories=sorted(all_ret_models), ordered=True)

    # Stage 1: Analysis for each iteration
    group_by_stage1 = group_by_cols + ['Iteration']
    summary_stage1_numeric = create_summary_table(merged_df, group_by_stage1, analysis_vars_dict, price_dict, return_numeric=True)

    # Stage 2: Aggregate over iterations
    agg_funcs = {}
    for var in analysis_vars_dict:
        mean_col = f'mean_{var}'
        agg_funcs[f'mean_{var}'] = (mean_col, 'mean')
        agg_funcs[f'ci_{var}'] = (mean_col, lambda s, v=var: mean_confidence_interval(s, non_negative=(v in ['Latency', 'Price'])))
    
    summary_stage2 = summary_stage1_numeric.groupby(group_by_cols, observed=True).agg(**agg_funcs)

    # Format stage 2 output
    for var in analysis_vars_dict:
        mean_col = f'mean_{var}'
        ci_col = f'ci_{var}'
        
        summary_stage2[f'ci_low_{var}'], summary_stage2[f'ci_upp_{var}'] = zip(*summary_stage2[ci_col])
        
        summary_stage2[var] = summary_stage2[mean_col].round(3).astype(str) + ' [' + summary_stage2[f'ci_low_{var}'].round(3).astype(str) + '-' + summary_stage2[f'ci_upp_{var}'].round(3).astype(str) + ']'
        
        summary_stage2 = summary_stage2.drop(columns=[mean_col, ci_col, f'ci_low_{var}', f'ci_upp_{var}'])

    # Also create a formatted version of stage 1 summary for inspection
    summary_stage1_formatted = summary_stage1_numeric.copy()
    for var in analysis_vars_dict:
        mean_col = f'mean_{var}'
        ci_low_col = f'ci_low_{var}'
        ci_upp_col = f'ci_upp_{var}'
        
        summary_stage1_formatted[var] = summary_stage1_numeric[mean_col].round(3).astype(str) + ' [' + summary_stage1_numeric[ci_low_col].round(3).astype(str) + '-' + summary_stage1_numeric[ci_upp_col].round(3).astype(str) + ']'
        summary_stage1_formatted = summary_stage1_formatted.drop(columns=[mean_col, ci_low_col, ci_upp_col])


    return summary_stage1_formatted, summary_stage2

def run_ci_summary(
    path,
    group_by_cols,
    price_dict=None,
    top_k=10,
    model_order=None,
    retriever_order=None,
    dataframe=None,
    precision_label=None,
):
    """
    Parameterized version of the CI analysis from 04_evaluations_CI_05.ipynb.

    Args:
        path (str): Folder that holds evaluation CSV files.
        group_by_cols (list[str]): Columns to group by (Iteration is appended automatically).
        price_dict (dict, optional): Mapping model -> price per 1M tokens (output). Defaults to DEFAULT_PRICE_DICT.
        top_k (int): Context window size for calculating Precision@k.
        model_order (list[str], optional): Explicit categorical order for the 'Model' column.
        retriever_order (list[str], optional): Explicit categorical order for the 'Model_ret' column.
        dataframe (pd.DataFrame, optional): Pre-loaded evaluations; skips reading from disk when provided.
        precision_label (str, optional): Display label to use for Precision@k (defaults to f'P@{top_k}').

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame):
            - per_iteration_summary: stats per run (includes Iteration column)
            - summary_table: aggregated stats across iterations with formatted CI columns
            - merged_df: raw merged evaluations
    """
    price_dict = price_dict or DEFAULT_PRICE_DICT
    base_group_cols = list(group_by_cols)
    if not base_group_cols:
        raise ValueError("group_by_cols must contain at least one column.")
    precision_label = precision_label or f'P@{top_k}'
    metric_specs = build_ci_metric_specs(precision_label)

    if dataframe is not None:
        merged_df = dataframe.copy()
    else:
        merged_df = merge_data(path)
    if merged_df.empty:
        return pd.DataFrame(), pd.DataFrame(), merged_df
    if 'Iteration' not in merged_df.columns:
        raise ValueError("Merged evaluations must include an 'Iteration' column.")

    if 'Model' in merged_df.columns:
        if model_order:
            merged_df['Model'] = pd.Categorical(merged_df['Model'], categories=model_order, ordered=True)
        else:
            merged_df['Model'] = pd.Categorical(
                merged_df['Model'], categories=sorted(merged_df['Model'].unique()), ordered=True
            )
    if 'Model_ret' in merged_df.columns:
        if retriever_order:
            merged_df['Model_ret'] = pd.Categorical(
                merged_df['Model_ret'], categories=retriever_order, ordered=True
            )
        else:
            merged_df['Model_ret'] = pd.Categorical(
                merged_df['Model_ret'], categories=sorted(merged_df['Model_ret'].unique()), ordered=True
            )

    merged_df['Throughput'] = calculate_throughput(merged_df)
    merged_df['Cost'] = calculate_cost(merged_df, price_dict)
    merged_df[precision_label] = merged_df.apply(
        lambda row: calculate_precision_at_k(row, top_k=top_k), axis=1
    )

    iteration_group_cols = list(dict.fromkeys(base_group_cols + ['Iteration']))
    per_iteration_summary = merged_df.groupby(iteration_group_cols, observed=True).agg(
        mean_cor_answer=('Cor_answer', 'mean'),
        mean_elapsed=('Elapsed', 'mean'),
        mean_tokens=('Total_tokens', 'mean'),
        mean_precision=(precision_label, 'mean'),
        mean_throughput=('Throughput', 'mean'),
        sum_cost=('Cost', 'sum'),
        sum_cor_answ=('Cor_answer', 'sum'),
    )
    per_iteration_summary['price_per_cost'] = np.where(
        per_iteration_summary['sum_cor_answ'] > 0,
        per_iteration_summary['sum_cost'] * 100 / per_iteration_summary['sum_cor_answ'],
        np.nan,
    )
    per_iteration_summary = per_iteration_summary.reset_index()

    agg_funcs = {}
    for spec in metric_specs:
        agg_funcs[spec['mean_col']] = (spec['source_col'], 'mean')
        agg_funcs[spec['ci_col']] = (spec['source_col'], spec['ci_func'])

    summary_table = per_iteration_summary.groupby(base_group_cols, observed=True).agg(**agg_funcs).reset_index()

    for spec in metric_specs:
        ci_low_col = f'ci_low_{spec["display"]}'
        ci_upp_col = f'ci_upp_{spec["display"]}'
        summary_table = _attach_ci_bounds(summary_table, spec['ci_col'], ci_low_col, ci_upp_col)
        if summary_table.empty:
            summary_table[spec['display']] = pd.Series(dtype=object)
        else:
            summary_table[spec['display']] = summary_table.apply(
                lambda row, m=spec['mean_col'], low=ci_low_col, upp=ci_upp_col, dec=spec.get('decimals', 3): _format_ci_cell(
                    row[m], row[low], row[upp], decimals=dec
                ),
                axis=1,
            )
        drop_cols = [col for col in (spec['mean_col'], ci_low_col, ci_upp_col) if col in summary_table.columns]
        if drop_cols:
            summary_table = summary_table.drop(columns=drop_cols)

    per_iteration_display = per_iteration_summary.rename(
        columns={
            'mean_cor_answer': 'Cor_answer',
            'mean_elapsed': 'Elapsed',
            'mean_tokens': 'Total_tokens',
            'mean_precision': precision_label,
            'mean_throughput': 'Throughput',
            'sum_cost': 'Cost',
            'sum_cor_answ': 'Correct_answers',
            'price_per_cost': 'Price-per-cost',
        }
    )

    return per_iteration_display, summary_table, merged_df
