#!/usr/bin/env python3

"""
async_viz_chat.py: example python script for chatting with vision models with vLLM
"""
import argparse
import aiohttp
import asyncio
import os
from time import time
import pickle
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import random
import torch


from dotenv.main import load_dotenv
load_dotenv()

from openai.lib._parsing._completions import type_to_response_format_param

from functions import format_msgs, encode_image, retrieve_colpali, response_real_out

BENCHMARK_FILE ="./data/Glycans_q_a_v5.xlsx"
EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"

TOP_K=5



qa_data = pd.read_excel(BENCHMARK_FILE  )
qa_data = qa_data.sample(frac=1).reset_index(drop=True)

import torch
import onnxruntime as rt

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
#from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from transformers import ColPaliForRetrieval, ColPaliProcessor


#qdrant_client = QdrantClient(":memory:")
qdrant_client = QdrantClient(url="http://localhost:6333", api_key=os.environ["QDRANT_API_KEY"])

embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL_ID, providers=["CUDAExecutionProvider"])

class MCQ(BaseModel):
    answer: str = Field(description="Output is the answer of a MCQ with only one of the following categories: A, B, C or D.")
    
    @field_validator("answer")
    def answer_must_letter (cls, answer:str) ->str:
        if answer[0] not in ["A", "B", "C", "D"]:
            raise ValueError(f'Must be element of A B C D')
        return answer

class MSC_choice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"    

class MCQ_(BaseModel):
    query_answer: MSC_choice

json_schema = MCQ_.model_json_schema()

def prompt_prep (table_qa, prompts, qdrant_client, vector_db,  embeddings, top_k, type, q_perm):
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
        
        model_id = "vidore/colpali-v1.3-hf"

        cp_model = ColPaliForRetrieval.from_pretrained(
            model_id,
            torch_dtype=torch.float,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
        ).eval()

        cp_processor = ColPaliProcessor.from_pretrained(model_id)
    elif type == "":
        print(f"Inference w/o RAG will be used!")
    else:
        print(f"Error, either enter mm_RAG or colpali or '' for RAG variable")


    output_array = {}
    for (_,row), perm_el in zip(table_qa.iterrows(),q_perm):
        
        #print([row['A'],row['B'],row['C'],row['D']])
        question = row['question']

        resp = ['A','B','C','D']
        answers = [row['A'],row['B'],row['C'],row['D']]
        answers_= [answers[i] for i in perm_el]
        question_string = "\n".join([f"{letter}. {option}" for letter, option in zip(['A','B','C','D'],answers_)])
        
        
        query = f"""\
        Query: {question} The possible answers are as follows:
        {question_string}
        """
        prompt_image = prompts["img_summary_query"].format(query=query)
        prompt_text = prompts["text_summary_query"].format(query=query)
        
        if type in ["", "mm_RAG"]:
            
            if type == "":
                context = []
            else:    
                context = qdrant.similarity_search_with_score(query, top_k)
            
            q_prompt = []
            for el in context:
                if el[0].metadata["type"] in ["image"]:         
                    part_prompt = format_msgs(prompt_image,[el[0].metadata["img_link"]],"")
                elif el[0].metadata["type"] in ["text", "table"]:
                    part_prompt = format_msgs(prompt_text,[],el[0].page_content)
                else:
                    part_prompt = format_msgs(prompt_text,[],"")
                q_prompt.append(part_prompt)
            
            output_array={**output_array,**{question: q_prompt}}
        elif type == "colpali":
            #colpali
            context = retrieve_colpali (query, cp_processor, cp_model, qdrant_client, "",vector_db, top_k)

            q_prompt = []
            for el in context.points:
                part_prompt = format_msgs(prompt_image,[el.payload["img_link"]],"")
                q_prompt.append(part_prompt)

            output_array={**output_array,**{question: q_prompt}}
        else:
            print(f"Error, either enter mm_RAG or colpali or '' for RAG variable")

    return output_array


filepath_prompts = './prompts_used.pkl'

async def post_request_with_retries (session, url,headers, data, retries=4, backoff=1):
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
                return f"[error] {response.status}"


async def main():
    parser = argparse.ArgumentParser(
        description="Script that adds 5 variables from CMD"
    )
    parser.add_argument("--vllm_port", required=True, type=int)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--filepath_output", required=True, type=str)
    parser.add_argument("--vector_db", required=True, type=str)
    parser.add_argument("--type", required=True, type=str)
    #if perm_quest "Yes" order of answers will be permuted!!!
    parser.add_argument(
        "--perm_quest",
        required=False,
        type=str,
        help="Set to 'yes' to randomise answer order per question.",
    )
    args = parser.parse_args()


    vllm_port = args.vllm_port 
    model_name  = args.model_name
    filepath_output  = args.filepath_output
    vector_db  = args.vector_db
    type= args.type
    permute_answers = bool(
        args.perm_quest and args.perm_quest.lower() in ["yes", "true", "1"]
    )


    t_start=time()
    

    if model_name.startswith("gpt"):
        url = f"https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ["OPENAI_API_KEY"]}",
                "Content-Type": "application/json"}
    else:
        url = f"http://localhost:{vllm_port}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ["VLLM_API_KEY"]}",
         "Content-Type": "application/json"}     
    
    with open(filepath_prompts, "rb") as file:
        prompts  = pickle.load(file)

    q_perm=[]
    for _,row in tqdm(qa_data.iterrows()):
        answers= [row['A'],row['B'],row['C'],row['D']]
        if permute_answers:
            perm_idx= random.sample(range(len(answers)), len(answers))

            #rev_perm_idx=[j for _,j in sorted((e,i) for i,e in enumerate(perm_idx))]
        else:
            perm_idx = list(range(len(answers)))
        q_perm.append(perm_idx)


    prompts_ = prompt_prep (qa_data, prompts, qdrant_client, vector_db,  embeddings, TOP_K, type, q_perm)

    # Create a TCPConnector with a limited number of connections
    

    final_prompts=[]
    output_file=[]
    perm_idx = []
    for (_,row), perm_el in zip(qa_data.iterrows(),q_perm):
    
        #print([row['A'],row['B'],row['C'],row['D']])
        question = row['question']
                  
        resp = ['A','B','C','D']
        answers= [row['A'],row['B'],row['C'],row['D']]
        #print(answers)
        rev_perm_idx=[j for _,j in sorted((e,i) for i,e in enumerate(perm_el))]
        answers_= [answers[i] for i in perm_el]


        question_string = "\n".join([f"{letter}. {option}" for letter, option in zip(['A','B','C','D'],answers_)])


        query_final = f"""\
        Generate a JSON with the query_answer, the answer provided behind the letters: A, B, C, and D. These are the values. Additional information if provided in the Context below. If the Context is not empty, analyse it and choose from the letters. MAKE SURE your output is one of the four values stated. 
        Here is the query: {question}. Here are the choices: {question_string} 
        Context:
        """
        conn = aiohttp.TCPConnector(limit=512)

        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = [
                post_request_with_retries(session, url=url, headers=headers, 
                    data={
                    "model": model_name,
                    "messages": msg,
                })
                for msg in prompts_[question]
            ]
            responses = await asyncio.gather(*tasks)

            fin_query = format_msgs(query_final,[],"\n".join(responses))
            output_file.append({"Question_nr":row["Question_nr"], "question": question, "summary": responses, "final_query": fin_query, "quest_order": perm_el})
            
            final_prompts.append(fin_query)


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
                "response_format": type_to_response_format_param(MCQ)
            })
            for msg in final_prompts
        ]
        answers_fin = await asyncio.gather(*tasks)
    
    out_list=[]
    for quest, answer in zip(output_file, answers_fin):
        filt_resp, answer_=response_real_out(answer, quest["quest_order"])
        out_list.append({**quest,  **{"answer": answer_, "resp_init": answer[:30], "filt_resp": filt_resp}})


    timestamp =pd.Timestamp('now', tz="CET").strftime("%Y%m%d-%H%M%S")
    eval_results = {"model": model_name, "evaluation": sorted(out_list, key=lambda x: x['Question_nr']), "elapsed_time": time() - t_start, "timestamp": timestamp }
    # for prompt, response in responses:
    #     print(f"prompt: {prompt}; response: {response}")
    suffix = "_perm_q" if permute_answers else ""
    eval_results["permuted_answers"] = permute_answers

    with open(f"{filepath_output}_{timestamp}{suffix}.pkl", 'wb') as file:
        pickle.dump(eval_results, file)


t0 = time()
asyncio.run(main())
print(f"Inference taks {time() - t0} seconds")
