# Evaluating Late-Interaction Multi-modal Models for Document Retrieval

This project provides a framework for evaluating and comparing the performance of various late-interaction multi-modal models for document retrieval and question answering. It is designed to be a flexible and extensible tool for researchers and developers working with visual document retrieval systems.

## Late-Interaction Models

Late-interaction models are a class of multi-modal models that process visual and textual information separately in the early stages and then combine them at a later stage. This approach allows for efficient retrieval and ranking of documents based on their visual and semantic content. This project is designed to be adaptable to various late-interaction models.

### Tested Models

This project has been tested with the following late-interaction models:

*   **ColPali:** [https://huggingface.co/vidore/colpali](https://huggingface.co/vidore/colpali)
*   **ColFlor:** [https://huggingface.co/ahmed-masry/ColFlor](https://huggingface.co/ahmed-masry/ColFlor)
*   **ColQwen2.5:** [https://huggingface.co/vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2)

## Workflow

The project follows a modular workflow that can be adapted to different models and datasets:

1.  **Data Preparation:** PDF documents are converted into a series of images, with each page corresponding to a single image.
2.  **Embedding Generation:** A chosen late-interaction model is used to generate vector embeddings for each page of the documents.
3.  **Retrieval and Question Answering:** For a given query, the system retrieves the most relevant document pages by comparing the query embedding with the pre-computed page embeddings. The retrieved pages are then used as context for a large language model to answer the question.
4.  **Evaluation:** The performance of the retrieval and question-answering pipeline is evaluated by comparing the generated answers with a set of ground-truth answers.

## Project Structure

*   `test_models.py`: The main script for running the evaluation pipeline from the command line.
*   `02_testing_models.ipynb`: A Jupyter notebook for running the evaluation pipeline. It includes code for loading models, processing documents, and generating answers.
*   `03_evaluations.ipynb`: A Jupyter notebook for analyzing the evaluation results. It provides tools for comparing the performance of different models and generating summary reports.
*   `data/`: This directory should contain the PDF documents for evaluation and the question-answering benchmark file.
*   `papers_merge/`: This directory should contain the PDF documents that will be processed.
*   `results/`: This directory is used to store the evaluation results.
*   `benchmark_placeholder.csv`: An example of the benchmark file structure.
*   `pyproject.toml` and `poetry.lock`: Dependency management files for the project.

## Usage

1.  **Install Dependencies:** Install the required dependencies using Poetry:
    ```bash
    poetry install
    ```
2.  **Set up Environment Variables:** Create a `.env` file in the root directory of the project and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_api_key_here"
    ```
3.  **Prepare Data:**
    *   Place your PDF documents in the `papers_merge/` directory.
    *   Create a benchmark file (e.g., `data/my_benchmark.xlsx`) with the same structure as `benchmark_placeholder.csv`.

## Creating a Benchmark Table

The benchmark table is a CSV or Excel file that contains the questions, answers, and other metadata for the evaluation. The file should have the following columns:

*   `Question_nr`: A unique identifier for each question.
*   `Paper_id`: The identifier for the paper the question is about.
*   `Nr_data_suppl`: The number of supplementary data files associated with the paper.
*   `doi`: The Digital Object Identifier of the paper.
*   `title`: The title of the paper.
*   `question`: The question to be answered.
*   `A`: The first possible answer.
*   `B`: The second possible answer.
*   `C`: The third possible answer.
*   `D`: The fourth possible answer.
*   `Correct`: The correct answer (one of 'A', 'B', 'C', or 'D').
*   `Difficulty`: The difficulty of the question (e.g., 'Easy', 'Medium', 'Hard').

You can use the `benchmark_placeholder.csv` file as a template.

4.  **Run Evaluation:**
    *   **Using the script:** Modify the `if __name__ == '__main__':` block in `test_models.py` to configure the models and data paths. Then, run the script:
        ```bash
        python test_models.py
        ```
    *   **Using the notebook:** Configure the evaluation pipeline in the `02_testing_models.ipynb` notebook by specifying the models to be evaluated and other parameters. Then, run the notebook cells.

5.  **Analyze Results:** Use the `03_evaluations.ipynb` notebook to analyze the results.

## Testing

To run the evaluation pipeline, you can use the `test_models.py` script. Before running the script, make sure you have:

1.  Installed the dependencies.
2.  Created the `.env` file with your OpenAI API key.
3.  Placed your PDF documents in the `papers_merge/` directory.
4.  Created your benchmark file in the `data/` directory.
5.  Updated the `qa_loc` variable in `test_models.py` to point to your benchmark file.

Then, you can run the script:

```bash
python test_models.py
```

The script will iterate through the specified models and save the evaluation results in the `results/evals/` directory.