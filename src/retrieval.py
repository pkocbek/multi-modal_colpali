import os
import time
import uuid

import stamina
import torch
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import ColPaliForRetrieval, ColPaliProcessor


@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(qdrant_client, collection_name, points):
    """Upserts a list of points to a Qdrant collection with a retry mechanism.

    Args:
        qdrant_client (QdrantClient): An instance of the Qdrant client.
        collection_name (str): The name of the collection to upsert to.
        points (list): A list of points to upsert.

    Returns:
        bool: True if the upsert was successful, False otherwise.
    """
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


def colpali_qdrant(
    dataset,
    papers,
    doi,
    model,
    processor,
    qdrant_client,
    qdrant_collection,
    batch_size=4,
):
    """Processes a dataset of images with a ColPali model and upserts the embeddings to a Qdrant collection.

    Args:
        dataset (list): A list of dictionaries, each containing image data.
        papers (list): A list of paper filenames.
        doi (list): A list of DOIs corresponding to the papers.
        model: The ColPali model.
        processor: The ColPali processor.
        qdrant_client (QdrantClient): An instance of the Qdrant client.
        qdrant_collection (str): The name of the Qdrant collection.
        batch_size (int, optional): The batch size for processing. Defaults to 4.
    """
    with tqdm(total=len(dataset), desc="Indexing Progress") as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            images = [item["image"] for item in batch]

            with torch.no_grad():
                batch_images = processor.process_images(images).to(model.device)
                image_embeddings = model(**batch_images)

            points = []
            for j, embedding in enumerate(image_embeddings.embeddings):
                uuid_idx = [str(uuid.uuid4()) for _ in range(len(image_embeddings.embeddings))]
                points.append(
                    models.PointStruct(
                        id=uuid_idx[j],
                        vector=embedding.tolist(),
                        payload={
                            "document_name": batch[j]["filename"],
                            "document_id": str(uuid.uuid4()),
                            "document_link": [
                                d
                                for p, d in zip(papers, doi)
                                if p.split("/")[-1] == batch[j]["filename"]
                            ][0],
                            "type": "pdf_page",
                            "page_no": batch[j]["page_no"],
                            "ref": "",
                            "caption": "",
                            "img_link": batch[j]["img_link"],
                        },
                    )
                )

            try:
                qdrant_client.upsert(collection_name=qdrant_collection, points=points)
            except Exception as e:
                print(f"Error during upsert: {e}")
                continue

            pbar.update(batch_size)

    print("Indexing complete!")


def retrieve_colpali(query, processor, model, qdrant_client, username, colection_name, top_k):
    """Retrieves the top-k most relevant documents for a given query using a ColPali model.

    Args:
        query (str): The query string.
        processor: The ColPali processor.
        model: The ColPali model.
        qdrant_client (QdrantClient): An instance of the Qdrant client.
        username (str): The username for filtering.
        colection_name (str): The name of the collection to retrieve from.
        top_k (int): The number of documents to retrieve.

    Returns:
        list: A list of retrieved documents.
    """
    with torch.no_grad():
        text_embedding = processor.process_queries([query]).to(model.device)
        text_embedding = model(**text_embedding)

    token_query = text_embedding.embeddings[0].cpu().float().numpy().tolist()

    start_time = time.time()
    if username == "":
        query_result = qdrant_client.query_points(
            collection_name=colection_name,
            query=token_query,
            limit=top_k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True, rescore=True, oversampling=2.0
                )
            ),
        )
    else:
        query_result = qdrant_client.query_points(
            collection_name=colection_name,
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
                    ignore=True, rescore=True, oversampling=2.0
                )
            ),
        )

    print(f"Time taken = {(time.time() - start_time):.3f} s")
    return query_result


def get_vd_elements(qdrant_client, username, vd_name, paper_dir):
    """Retrieves a list of documents from a Qdrant collection.

    Args:
        qdrant_client (QdrantClient): An instance of the Qdrant client.
        username (str): The username for filtering.
        vd_name (str): The name of the vector database.
        paper_dir (str): The directory containing the paper files.

    Returns:
        tuple: A tuple containing the document names, links, and DOI links.
    """
    papers = [
        os.path.join(paper_dir, f)
        for f in sorted(os.listdir(paper_dir))
        if f.lower().endswith(".pdf")
    ]

    retrived_elmns = qdrant_client.scroll(
        collection_name=vd_name,
        scroll_filter=models.Filter(
            must_not=[
                models.FieldCondition(
                    key="metadata.document_name", match=models.MatchValue(value="")
                ),
            ],
        ),
        limit=100000,
        with_payload=True,
    )

    lst = []
    for el in retrived_elmns[0]:
        dct = {
            k: v
            for k, v in el.payload["metadata"].items()
            if k in ["document_name", "document_link"]
        }
        lst.append(dct)
    lst = [dict(t) for t in {tuple(d.items()) for d in lst}]
    lst = sorted(lst, key=lambda d: d["document_name"])

    dt = [el["document_name"] for el in lst]
    doi_links = [el["document_link"] for el in lst]

    links = [paper for el in dt for paper in papers if el in paper]

    return dt, links, doi_links


def get_vd_elements_colpali(qdrant_client, username, vd_name, paper_dir):
    """Retrieves a list of documents from a ColPali Qdrant collection.

    Args:
        qdrant_client (QdrantClient): An instance of the Qdrant client.
        username (str): The username for filtering.
        vd_name (str): The name of the vector database.
        paper_dir (str): The directory containing the paper files.

    Returns:
        tuple: A tuple containing the document names, links, and DOI links.
    """
    papers = [
        os.path.join(paper_dir, f)
        for f in sorted(os.listdir(paper_dir))
        if f.lower().endswith(".pdf")
    ]

    retrived_elmns = qdrant_client.scroll(
        collection_name=vd_name,
        scroll_filter=models.Filter(
            must_not=[
                models.FieldCondition(
                    key="metadata.document_name", match=models.MatchValue(value="")
                ),
            ],
            must=[
                models.FieldCondition(
                    key="metadata.username", match=models.MatchValue(value=username)
                ),
            ],
        ),
        limit=100000,
        with_payload=True,
    )

    lst = []
    for el in retrived_elmns[0]:
        dct = {k: v for k, v in el.payload.items() if k in ["document_name", "document_link"]}
        lst.append(dct)
    lst = [dict(t) for t in {tuple(d.items()) for d in lst}]
    lst = sorted(lst, key=lambda d: d["document_name"])

    dt = [el["document_name"] for el in lst]
    doi_links = [el["document_link"] for el in lst]

    links = [paper for el in dt for paper in papers if el in paper]

    return dt, links, doi_links


def qdrant_process(docs, qdrant_client, vec_db, emb_dim, embeddings, url="http://localhost:6333"):
    """Processes and upserts a list of documents to a Qdrant collection.

    Args:
        docs (list): A list of documents to process.
        qdrant_client (QdrantClient): An instance of the Qdrant client.
        vec_db (str): The name of the vector database.
        emb_dim (int): The embedding dimension.
        embeddings: The embedding model.
        url (str, optional): The URL of the Qdrant service. Defaults to "http://localhost:6333".
    """
    print(f"Processing data for colection {vec_db}.")

    if not qdrant_client.collection_exists(collection_name=vec_db):
        qdrant_client.create_collection(
            collection_name=vec_db,
            on_disk_payload=True,
            vectors_config=models.VectorParams(
                size=emb_dim, on_disk=True, distance=models.Distance.COSINE
            ),
        )

    QdrantVectorStore.from_documents(
        docs,
        embedding=embeddings,
        url=url,
        collection_name=vec_db,
        retrieval_mode=RetrievalMode.DENSE,
    )
    print(f"Processing of {len(docs)} for colection {vec_db} complete.")