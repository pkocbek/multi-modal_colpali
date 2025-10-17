import os
import uuid
from pathlib import Path

import pandas as pd
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    granite_picture_description,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.types.doc import PictureItem, TableItem, TextItem
from docling_core.types.doc.document import DoclingDocument, PictureDescriptionData, RefItem
from docling_core.types.doc.labels import DocItemLabel
from httpx import HTTPError, JSONDecodeError
from langchain_core.documents import Document
from pdf2image import convert_from_path
from PIL import Image


def doc_conv(ocr=True, check_only=False):
    """Initializes and returns a DocumentConverter object with specific pipeline options for PDF processing.

    Args:
        ocr (bool, optional): Whether to enable OCR. Defaults to True.
        check_only (bool, optional): If True, configures the pipeline for a quick check without extensive processing. Defaults to False.

    Returns:
        DocumentConverter: An instance of the DocumentConverter class.
    """
    IMAGE_RESOLUTION_SCALE = 2.0
    pipeline_options = PdfPipelineOptions()
    if ocr:
        pipeline_options.do_ocr = True
        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options
    else:
        pipeline_options.do_ocr = False

    if not check_only:
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = granite_picture_description
        pipeline_options.picture_description_options.prompt = (
            "Describe the image in four sentences. Be consise, scientific and accurate. Provide numbers if it improves the description."
        )

    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.CUDA, cuda_use_flash_attention2=True
    )

    pipeline_options.accelerator_options = accelerator_options

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
        },
    )


def check_ocr(file_name):
    """Checks if a PDF file requires OCR by analyzing its first page for text content.

    Args:
        file_name (str): The path to the PDF file.

    Returns:
        bool: True if OCR is likely required, False otherwise.
    """
    parse_first_pg = doc_conv(ocr=False, check_only=True).convert(
        source=file_name, page_range=[1, 1]
    )
    text = []

    for element, _level in parse_first_pg.document.iterate_items():
        if isinstance(element, TextItem):
            text.append(element.text)

    return len(text) == 0


def resize_image(
    image: Image.Image, min_size: int = 224, max_size: int = 1300
) -> Image.Image:
    """Resizes an image to fit within a specified minimum and maximum size range.

    Args:
        image (Image.Image): The image to resize.
        min_size (int, optional): The minimum size for the smaller dimension. Defaults to 224.
        max_size (int, optional): The maximum size for the larger dimension. Defaults to 1300.

    Returns:
        Image.Image: The resized image.
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


def data_preparation(
    conversion: list,
    vd_dir,
    vd_tokenizer,
    mm_dir="",
    only_text=False,
    page_images=True,
):
    """Prepares documents for vectorization by processing text, tables, and images.

    Args:
        conversion (list): A list of dictionaries, each containing document information.
        vd_dir (str): The directory for the vector database.
        vd_tokenizer: The tokenizer to use for chunking.
        mm_dir (str, optional): The directory for multi-modal data. Defaults to "".
        only_text (bool, optional): If True, only process text. Defaults to False.
        page_images (bool, optional): If True, generate and save page images. Defaults to True.

    Returns:
        list: A list of Document objects ready for vectorization.
    """
    all_docs = []
    for el in conversion:
        filename = el["filename"]
        filename_link = el["link"]
        documents_id = str(uuid.uuid4())

        save_dir = mm_dir if mm_dir else vd_dir

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if page_images:
            for page_no, page in el["document"].pages.items():
                page_image_filename = (
                    Path(save_dir) / f"pg_images/{filename.split(".")[0]}_{page.page_no:03d}.png"
                )
                with page_image_filename.open("wb") as fp:
                    img = resize_image(page.image.pil_image)
                    img.save(fp, format="PNG")

        if not only_text:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for element, _level in el["document"].iterate_items():
                if isinstance(element, TableItem):
                    # ... (rest of the table processing logic)
                    pass
                if isinstance(element, PictureItem):
                    # ... (rest of the picture processing logic)
                    pass

        texts: list[Document] = []
        for chunk in HybridChunker(tokenizer=vd_tokenizer).chunk(el["document"]):
            items = chunk.meta.doc_items
            if len(items) == 1 and isinstance(items[0], TableItem) and not only_text:
                continue
            ref = " ".join(map(lambda item: item.get_ref().cref, items))
            text = chunk.text
            document = Document(
                page_content=text,
                metadata={
                    "document_name": el["document"].origin.filename,
                    "document_id": documents_id,
                    "document_link": filename_link,
                    "type": "text",
                    "page_no": chunk.meta.doc_items[0].prov[0].page_no,
                    "ref": ref,
                    "caption": "",
                    "img_link": "",
                },
            )
            texts.append(document)

        if only_text:
            all_docs.extend(texts)
            print(f"For {filename}, {len(texts)} text documents were processed.")
            continue

        # ... (rest of the table and picture processing logic)

    print(f"Total number of documents processed: {len(all_docs)}.")
    return all_docs


def pdf_loader(papers, doi_links, filenames, vd_dir, vd_tolkenizer, username=""):
    """Loads and processes PDF files, preparing them for vectorization.

    Args:
        papers (list): A list of paths to the PDF files.
        doi_links (list): A list of DOI links corresponding to the papers.
        filenames (list): A list of filenames for the papers.
        vd_dir (str): The directory for the vector database.
        vd_tolkenizer: The tokenizer to use for chunking.
        username (str, optional): The username of the user. Defaults to "".

    Returns:
        tuple: A tuple containing the processed multi-modal and text-only documents.
    """
    conversion = []
    for paper, doi_paper, filename in zip(papers, doi_links, filenames):
        ocr = check_ocr(paper)
        conversion.append(
            {
                "filename": filename,
                "document": doc_conv(ocr=ocr).convert(source=paper).document,
                "link": doi_paper,
            }
        )

    mm_dir = f"./src/vectordb/user_data/{username}/" if username else ""
    pg_image = not username

    processed_multi = data_preparation(
        conversion, vd_dir, vd_tokenizer=vd_tolkenizer, mm_dir=mm_dir, only_text=False, page_images=pg_image
    )

    processed_text = data_preparation(
        conversion, vd_dir, vd_tokenizer=vd_tolkenizer, mm_dir=mm_dir, only_text=True, page_images=pg_image
    )

    return processed_multi, processed_text