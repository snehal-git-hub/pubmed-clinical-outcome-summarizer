# PubMed Clinical Outcome Summarizer

A Python-based tool that processes PubMed-style JSON records, cleans and normalizes article information, and generates concise, one-sentence clinical outcome summaries using an OpenAI-compatible LLM API powered by Llama 3.3 70B.

## Project Overview

This project automates the processing of PubMed research abstracts. It extracts relevant article information, validates records, handles inconsistent JSON field names, and sends valid abstracts to a locally hosted LLM API for clinical outcome summarization.

The application is designed to work with a locally hosted Llama 3.3 70B model through an OpenAI-compatible chat completions endpoint.

## Features

- **JSON Data Processing:** Loads PubMed records from JSON files or built-in sample data.
- **Data Cleaning:** Handles inconsistent field names, missing values, and different data formats.
- **ID Validation:** Removes invalid records and duplicate PubMed IDs.
- **Abstract Validation:** Skips records with missing abstracts.
- **Text Normalization:** Cleans whitespace and normalizes titles, abstracts, and author information.
- **LLM Integration:** Sends abstracts to an OpenAI-compatible LLM API.
- **Clinical Outcome Summarization:** Generates concise, one-sentence summaries based only on the information in the abstract.
- **Error Handling:** Handles connection failures, timeouts, HTTP errors, and invalid API responses.
- **Retry Mechanism:** Uses exponential backoff for temporary API failures.
- **JSON Output:** Saves processed summaries and processing status in a structured JSON file.

## Technologies Used

- Python 3
- Requests
- JSON
- Regular Expressions
- Logging
- OpenAI-compatible REST API
- Llama 3.3 70B

## Repository Structure

```text
pubmed-clinical-outcome-summarizer/
│
├── pubmed_summarizer.py
├── mock_api.py
├── sample_pubmed_data.json
├── pubmed_summaries.json
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    ├── 01_code.png
    ├── 02_mock_server.png
    ├── 03_execution.png
    └── 04_output.png
```

*The sample data and screenshot filenames are optional; include them only if you have created those files.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/snehal-git-hub/pubmed-clinical-outcome-summarizer.git
cd pubmed-clinical-outcome-summarizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API endpoint

The script uses the following default API endpoint:

```text
http://192.168.1.100:8000/v1/chat/completions
```

The endpoint can be changed using the `LLM_API_URL` environment variable.

For example, in Windows PowerShell:

```powershell
$env:LLM_API_URL="http://127.0.0.1:8000/v1/chat/completions"
```

The model name can also be configured using `LLM_MODEL`. Ensure that it matches the model name served by the API.

## Usage

### Run with built-in sample data

```bash
python pubmed_summarizer.py
```

### Run with an external JSON file

```bash
python pubmed_summarizer.py --input sample_pubmed_data.json --output pubmed_summaries.json
```

### Run with a custom output filename

```bash
python pubmed_summarizer.py --output results.json
```

## Sample Input

The sample dataset contains three records with different field names and data quality issues.

```json
[
  {
    "id": "PMID:349281",
    "ArticleTitle": "Efficacy of Drug X in severe cardiac failure.",
    "abstract_text": "We tested 500 patients. The drug worked in 40% of cases but caused nausea.",
    "authorList": ["Smith, J", "Doe, A"]
  },
  {
    "id": 8829102,
    "title": null,
    "Abstract": "Null hypothesis rejected. No significant improvement noted in cohort B vs placebo.",
    "authors": "Unknown"
  },
  {
    "id": "ERR_NOT_FOUND",
    "ArticleTitle": "Corrupted Entry",
    "abstract_text": null
  }
]
```

## Data Cleaning and Validation

The application performs the following operations:

1. Identifies article IDs from supported ID fields.
2. Removes the `PMID:` prefix from valid IDs.
3. Rejects invalid or missing article IDs.
4. Extracts titles and abstracts from alternative field names.
5. Normalizes author information into a list.
6. Skips records with missing abstracts.
7. Removes duplicate PubMed IDs.
8. Truncates abstracts exceeding the configured character limit.

For the sample dataset, two valid records are processed and the corrupted third record is skipped.

## API Integration and Reliability

The application sends requests to the LLM using the OpenAI-compatible chat completions format.

Reliability features include:

- Connection timeout: 5 seconds.
- Read timeout: 180 seconds.
- Maximum retry attempts: 4.
- Exponential backoff between retries.
- Retry handling for HTTP 429, 500, 502, 503, and 504 responses.
- Logging of errors and processing status.

These settings help handle temporary network interruptions and slow model responses.

## Mock API Testing

A mock API server (`mock_api.py`) is included for testing the Python application's request and response handling without requiring access to the actual Llama model server.

Start the mock server in one PowerShell terminal:

```powershell
python mock_api.py
```

In another terminal, configure the local endpoint and run the application:

```powershell
$env:LLM_API_URL="http://127.0.0.1:8000/v1/chat/completions"
python pubmed_summarizer.py
```

**Testing note:** Mock API testing verifies the application's data processing, HTTP communication, and output generation. It does not perform actual Llama 3.3 70B inference.

## Output

The application generates a JSON file named:

```text
pubmed_summaries.json
```

Each processed record contains:

- `pmid`
- `title`
- `authors`
- `clinical_outcome_summary`
- `status`

Failed API requests are recorded with an error message so that one failed article does not stop the processing of subsequent records.

## Configuration

| Parameter | Default value |
|---|---|
| Model | meta-llama/Llama-3.3-70B-Instruct |
| Connection timeout | 5 seconds |
| Read timeout | 180 seconds |
| Maximum retries | 4 |
| Backoff base | 2 seconds |
| Maximum abstract length | 6000 characters |

The model identifier must match the model served by the API endpoint.

## Hardware and Deployment

The intended deployment environment is an NVIDIA DGX Spark running a locally hosted Llama 3.3 70B model through an OpenAI-compatible inference server.

Model deployment settings, including quantization, GPU memory utilization, maximum context length, and batching, should be configured according to the available hardware and the inference server's supported model format.

The Python client is independent of the underlying inference hardware as long as the server exposes the expected API endpoint.

## Limitations

- Actual LLM inference requires access to a running model API server.
- Mock API responses are for testing and are not genuine model-generated clinical summaries.
- Summary quality depends on the information available in the original abstract.
- The application does not independently verify clinical claims or replace professional medical interpretation.

## Author

Developed as part of a technical assignment involving PubMed data processing, Python automation, and local LLM API integration.