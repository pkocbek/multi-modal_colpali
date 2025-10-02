from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColFlor, ColFlorProcessor
from transformers import AutoProcessor, AutoModel
from transformers.utils.import_utils import is_flash_attn_2_available
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
import base64
from io import BytesIO
import pandas as pd
import pickle
import json
from time import gmtime, strftime
from openai._client import OpenAI
import nest_asyncio
import math
from openai import OpenAI, RateLimitError,AsyncOpenAI
import asyncio
import backoff
from pydantic import BaseModel
from typing import Literal

#from colpali_engine.models import ColIdefics3, ColIdefics3Processor
#from colpali_engine.models import ColQwen2_5Omni, ColQwen2_5OmniProcessor
#from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

os.makedirs("data", exist_ok=True)
os.makedirs("papers_merge", exist_ok=True)
os.makedirs("results/evals", exist_ok=True)

def convert_pdf_to_images(pdf_dir):
    """
    Converts all PDFs in a directory to images.

    Args:
        pdf_dir (str): Path to the directory containing PDFs.

    Returns:
        dict: A dictionary where keys are file names (without extension), 
              and values are lists of images (one list per PDF).
    """
    pdf_list = [pdf for pdf in sorted(os.listdir(pdf_dir)) if pdf.endswith(".pdf")]
    all_images = {}

    for pdf_file in pdf_list:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_images = convert_from_path(pdf_path)
        all_images[pdf_file] = pdf_images  # Use file name as key
    
    return all_images

def create_document_embeddings(pdf_dir, model, processor, batch_size=2):
    """
    Converts all PDFs in a directory into embeddings with metadata.

    Args:
        pdf_dir (str): Directory containing PDF files.
        model: Pre-trained model for generating embeddings.
        processor: Preprocessor for the model (e.g., to process images).
        batch_size (int): Batch size for inference.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "embedding": The embedding tensor.
            - "doc_id": The document ID (int).
            - "page_id": The page index within the document.
            - "file_name": The name of the source PDF file.
    """
    all_images = convert_pdf_to_images(pdf_dir)
    all_embeddings_with_metadata = []

    for doc_id, (file_name, pdf_images) in enumerate(all_images.items()):
        dataloader = DataLoader(
            dataset=pdf_images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: processor.process_images(x),
        )

        page_counter = 0
        for batch in tqdm(dataloader, desc=f"Processing {file_name}"):
            with torch.no_grad():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                batch_embeddings = model(**batch)
                batch_embeddings = list(torch.unbind(batch_embeddings.to("cpu")))

                for embedding in batch_embeddings:
                    all_embeddings_with_metadata.append({
                        "embedding": embedding,
                        "doc_id": doc_id,
                        "page_id": page_counter,
                        "file_name": file_name,  # Correctly use the file name
                    })
                    page_counter += 1

    return all_embeddings_with_metadata

def get_results(query, processor, model, ds, all_images, top_k=5):
    """
    Retrieves top-k relevant images for a given query.

    Args:
        query (str): User query as a string.
        processor: Processor for pre-processing the query.
        model: Model to generate embeddings for the query.
        ds (list): List of dictionaries with "embedding", "doc_id", "page_id", and "file_name".
        all_images (dict): Dictionary of images per document.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "doc_id": Document ID.
            - "page_id": Page ID.
            - "file_name": Name of the source PDF file.
            - "image": The retrieved image (PIL.Image.Image).
            - "score": Similarity score for the image.
    """
    # Process the query and move to model's device
    query_list = query if isinstance(query, list) else [query]
    batch_queries = processor.process_queries(query_list).to(model.device)

    # Forward pass to get query embeddings
    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    # Extract embeddings from ds for scoring
    document_embeddings = torch.stack([entry["embedding"] for entry in ds])
    
    # Compute similarity scores
    scores = processor.score_multi_vector(query_embeddings, document_embeddings)
    
    all_results=[]
    for el in scores:

        score_values = el.tolist()  # Extract similarity scores as a list

        # Get top-k indices of the most relevant embeddings
        top_indices = el.topk(top_k).indices.tolist()

       # Retrieve corresponding images and metadata
        retrieved_results = []
        for idx in top_indices:
            entry = ds[idx]
            doc_id = entry["doc_id"]
            page_id = entry["page_id"]
            file_name = entry["file_name"]
            image = all_images[file_name][page_id]  # Correct lookup using file_name

            # Add score to each result
            retrieved_results.append({
                "doc_id": doc_id,
                "page_id": page_id,
                "file_name": file_name,
                "image": image,
                "score": score_values[idx],  # Add similarity score
            })
        all_results.append(retrieved_results)

    return all_results

class MCQ(BaseModel):
   answer: Literal["A", "B", "C", "D"]

def resize_base64_image(base64_string, fixed_width=1024):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    
    width = fixed_width
    img_ratio = img.size[0] / img.size[1]
    height = int(width/img_ratio)
    size_new = width, height

    # Resize the image
    resized_img =img.resize(size_new, resample=Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def img_context (context_imgs):
    messages = []
    for image in context_imgs:
        # Save the resized image to a bytes buffer

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        resized_image = resize_base64_image(img_str)
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{resized_image}"},
        }
        messages.append(image_message)
    
    out_prompt = [{"type": "text","text": "Context information:" }]
    

    return out_prompt + messages

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Decorator for exponential backoff with a maximum of 5 retries
@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
async def get_completion_with_backoff(client, gpt_model, prompt_el, MCQ):
    """
    Asynchronously sends a request to the OpenAI API with exponential backoff for rate limiting.

    Args:
        client (AsyncOpenAI): The asynchronous OpenAI client.
        gpt_model (str): The model to use for the completion.
        prompt_el (list): The prompt messages for the model.
        MCQ (Pydantic.BaseModel): The response format for the completion.

    Returns:
        dict: The parsed JSON response from the API.
    """
    completion = await client.beta.chat.completions.parse(
        model=gpt_model,
        messages=prompt_el,
        response_format=MCQ
    )
    return json.loads(completion.choices[0].message.content)

async def send_to_model_async(gpt_model, table_qa, no_context=1, topk=5, chunk=10, processor=[], model=[], ds=[], all_images=[]):
    """
    Asynchronously sends questions to a model and retrieves answers, with support for context retrieval.

    Args:
        gpt_model (str): The GPT model to use for answering questions.
        table_qa (pd.DataFrame): DataFrame with questions and answer options.
        no_context (int): If 0, retrieves context; otherwise, sends questions directly.
        topk (int): The number of top results to retrieve for context.
        chunk (int): The chunk size for processing questions.
        processor: The processor for the retrieval model.
        model: The retrieval model.
        ds: The dataset for retrieval.
        all_images: A dictionary of all images.

    Returns:
        list: A list containing the output array of answers and a list of retrieved information.
    """
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    prompt_list = []
    prompt_q = []
    for _, row in table_qa.iterrows():
        question = row['question']
        resp = ['A', 'B', 'C', 'D']
        question_string = "".join([f"{letter}. {option}" for letter, option in zip(resp, [row['A'], row['B'], row['C'], row['D']])])
        
        prompt = f"""
        You are an experienced senior researcher tasked with providing in-depth analysis.
        Use all the information at your disposal, such as uploaded files and other sources. Think about the following statement or question: {question}\n
        Below are the possible answers, where letters mark each answer. First, exclude the unlikely answer or answers, rethink, and select an output from the rest. The output is only ONE letter from the list {resp}. Check that you return only one letter; if two letters, choose one. No explanations. The answers are:\n
        {question_string}
        """
        prompt_q.append(prompt)
        prompt_list.append(question + " The answers are:" + question_string)

    prompt_llm_list = []
    info_res = []
    retrieved_results = []

    if no_context == 0:
        chunk_list = chunks(prompt_list, chunk)
        
        for chk_el in chunk_list:
            tmp = get_results(
                query=chk_el,
                processor=processor,
                model=model,
                ds=ds,
                all_images=all_images,
                top_k=topk,
            )
            retrieved_results.extend(tmp)

        print(f"Retrieved {len(retrieved_results)} results for context.")
        
        for prpmt, ret_files in zip(prompt_q, retrieved_results):
            info_el = [el["file_name"].split(".")[0] + "_pg_" + str(el["page_id"]) for el in ret_files]
            info_res.append(info_el)
    
            context_imgs = [img["image"] for img in ret_files]
            conv_imgs = img_context(context_imgs)
            prompt_all = [{"role": "user", "content": [{"type": "text", "text": prpmt}] + conv_imgs}]
            prompt_llm_list.append(prompt_all)
    else:
        prompt_llm_list = [[{"role": "user", "content": [{"type": "text", "text": prpt}]}] for prpt in prompt_q]
        info_res = [""] * len(prompt_q)

    # Create a list of tasks to run concurrently
    tasks = [get_completion_with_backoff(client, gpt_model, prompt_el, MCQ) for prompt_el in prompt_llm_list]
    
    # Run tasks concurrently and gather results
    completions = await asyncio.gather(*tasks)
    
    output_array = [comp["answer"] for comp in completions]
        
    print(f"Checking list lengths: eval_table {len(table_qa)}, retrieved list {len(retrieved_results)}.")

    return [output_array, info_res]

# Wrapper function to run the async function
def send_to_model(gpt_model, table_qa, no_context=1, topk=5, chunk=10, processor=[], model=[], ds=[], all_images=[]):
    nest_asyncio.apply()
    return asyncio.run(send_to_model_async(gpt_model, table_qa, no_context, topk, chunk, processor, model, ds, all_images))

def eval_fn(MODEL, MODEL_RET, device, qa_data, nr_iter=5, topk=5, chunk=10, out_dir= 'results/evals/', no_context = 0, ds_file="data/colpali_pdf_emb.pkl", pdf_dir='papers_merge/'):
    
    if MODEL_RET == "vidore/colpali-v1.3-merged":
        processor = ColPaliProcessor.from_pretrained(MODEL_RET)

        model = ColPali.from_pretrained(
            MODEL_RET,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    elif MODEL_RET == "ahmed-masry/ColFlor":
        model = ColFlor.from_pretrained(
            MODEL_RET,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()

        processor = ColFlorProcessor.from_pretrained(MODEL_RET)
    
    elif MODEL_RET == "vidore/colSmol-500M":
        model = ColIdefics3.from_pretrained(
            MODEL_RET,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()

        processor = ColIdefics3Processor.from_pretrained(MODEL_RET)

    elif MODEL_RET == "ibm-granite/granite-vision-3.3-2b-embedding":
        model = AutoModel.from_pretrained(
            MODEL_RET,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()

        processor = AutoProcessor.from_pretrained(MODEL_RET)
        
    elif MODEL_RET == "vidore/colqwen2.5-v0.2":
        model = ColQwen2_5.from_pretrained(
            MODEL_RET,
            torch_dtype=torch.bfloat16,
            device_map=device,  # or "mps" if on Apple Silicon
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(MODEL_RET)
    else:
        print(f"Select correct MODEL_RET, current {MODEL_RET} not correct")
        return -1

    

    if ds_file != "":
        with open(ds_file, 'rb') as fp:
            ds = pickle.load(fp)
    else:
        ds = create_document_embeddings(pdf_dir, model, processor, batch_size=4)
        with open('data/'+MODEL_RET+'_pdf_emb.pkl', 'wb') as file:
           pickle.dump(ds, file)

    all_images = convert_pdf_to_images(pdf_dir)

    for i in range(nr_iter):
        new_data = []
        print(f'Processing iteration: {i+1} for model: {MODEL}')

        out = send_to_model(MODEL, qa_data, no_context, topk, chunk, processor, model, ds, all_images)
        
        new_data = qa_data

        new_data["Model"]=MODEL
        new_data["Model_ret"]=MODEL_RET
        new_data["Answer"]=out[0]
        new_data["Context_papers"]=out[1]
        new_data['Cor_answer'] = 1*(new_data['Answer'] == new_data['Correct'])

        new_data.to_csv(out_dir +'eval_' +MODEL_RET.split("/")[-1].split("-")[0] +'_'+ MODEL + '_'+ strftime("%Y%m%d%H%M%S", gmtime()) +'.csv')

        print(f'Accuracy: {sum(new_data["Cor_answer"])/len(new_data["Answer"])}')


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    pdf_dir = 'papers_merge/'
    qa_loc ="./data/Glycans_q_a_v5.xlsx"
    qa_data = pd.read_excel(qa_loc)
    qa_data = qa_data.sample(frac=1).reset_index(drop=True)

    MODELS = ["gpt-5", "gpt-5-mini",  "gpt-5-nano"]
    MODEL_RET = ["vidore/colpali-v1.3-merged", "ahmed-masry/ColFlor", "vidore/colqwen2.5-v0.2"] 
    TOP_K = 5
    chunk = 10
    nr_iter = 5

    for model in MODELS:
        for model_ret in MODEL_RET:
            eval_fn(model, model_ret, device, qa_data, nr_iter=nr_iter, topk=TOP_K, chunk=chunk, out_dir= 'results/evals/')
