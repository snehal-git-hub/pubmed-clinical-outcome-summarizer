
import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv(
    "LLM_API_URL",
    "http://192.168.1.100:8000/v1/chat/completions"
)

MODEL = os.getenv(
    "LLM_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct"
)

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 180

MAX_RETRIES = 4
BACKOFF_BASE = 2

MAX_ABSTRACT_CHARS = 6000

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# SAMPLE PUBMED DATA
# --------------------------------------------------

SAMPLE_DATA = [
    {
        "id": "PMID:349281",
        "ArticleTitle": "Efficacy of Drug X in severe cardiac failure.",
        "abstract_text": (
            "We tested 500 patients. The drug worked in 40% "
            "of cases but caused nausea."
        ),
        "authorList": ["Smith, J", "Doe, A"]
    },
    {
        "id": 8829102,
        "title": None,
        "Abstract": (
            "Null hypothesis rejected. No significant improvement "
            "noted in cohort B vs placebo."
        ),
        "authors": "Unknown"
    },
    {
        "id": "ERR_NOT_FOUND",
        "ArticleTitle": "Corrupted Entry",
        "abstract_text": None
    }
]


# --------------------------------------------------
# FIELD NORMALIZATION
# --------------------------------------------------

def first_nonempty(record: Dict[str, Any], keys: List[str]) -> Any:
    """Return the first non-empty value among possible field names."""

    for key in keys:
        value = record.get(key)

        if value is not None and value != "":
            return value

    return None


def normalize_text(value: Any) -> str:
    """Convert a value into clean text."""

    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(str(item) for item in value)

    text = str(value)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_id(value: Any) -> Optional[str]:
    """Normalize PubMed IDs and reject invalid identifiers."""

    if value is None:
        return None

    record_id = normalize_text(value)

    if not record_id:
        return None

    record_id = re.sub(r"^PMID:\s*", "", record_id, flags=re.IGNORECASE)

    if record_id.upper() in {
        "ERR_NOT_FOUND",
        "UNKNOWN",
        "NULL",
        "NONE",
        "N/A"
    }:
        return None

    if not record_id.isdigit():
        return None

    return record_id


def normalize_authors(value: Any) -> List[str]:
    """Normalize author information into a list."""

    if value is None:
        return []

    if isinstance(value, list):
        return [
            normalize_text(author)
            for author in value
            if normalize_text(author)
        ]

    text = normalize_text(value)

    if not text or text.lower() == "unknown":
        return []

    return [
        author.strip()
        for author in re.split(r";|\s+and\s+", text)
        if author.strip()
    ]


# --------------------------------------------------
# RECORD CLEANING
# --------------------------------------------------

def clean_record(
    record: Dict[str, Any],
    index: int
) -> Optional[Dict[str, Any]]:
    """Validate and normalize a single PubMed record."""

    if not isinstance(record, dict):
        logger.warning("Skipping record %s: invalid record format.", index)
        return None

    raw_id = first_nonempty(
        record,
        ["id", "pmid", "ID", "PMID"]
    )

    record_id = normalize_id(raw_id)

    if not record_id:
        logger.warning(
            "Skipping record %s: missing or invalid article ID.",
            index
        )
        return None

    title = normalize_text(
        first_nonempty(
            record,
            ["title", "ArticleTitle", "article_title", "Title"]
        )
    )

    abstract = normalize_text(
        first_nonempty(
            record,
            ["abstract_text", "Abstract", "abstract", "AbstractText"]
        )
    )

    authors = normalize_authors(
        first_nonempty(
            record,
            ["authorList", "authors", "Authors", "author_list"]
        )
    )

    if not abstract:
        logger.warning(
            "Skipping PMID %s: missing abstract.",
            record_id
        )
        return None

    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS]
        logger.warning(
            "PMID %s: abstract truncated to %s characters.",
            record_id,
            MAX_ABSTRACT_CHARS
        )

    return {
        "pmid": record_id,
        "title": title or "Untitled article",
        "abstract": abstract,
        "authors": authors
    }


def load_and_clean(input_file: Optional[str]) -> List[Dict[str, Any]]:
    """Load JSON data and return cleaned PubMed records."""

    if input_file:
        try:
            with open(input_file, "r", encoding="utf-8") as file:
                raw_data = json.load(file)

        except FileNotFoundError:
            logger.error("Input file not found: %s", input_file)
            raise

        except json.JSONDecodeError as error:
            logger.error("Invalid JSON file: %s", error)
            raise

    else:
        logger.info("Using built-in sample PubMed data.")
        raw_data = SAMPLE_DATA

    if not isinstance(raw_data, list):
        raise ValueError("Input JSON must contain a list of records.")

    cleaned_records = []
    seen_ids = set()

    for index, record in enumerate(raw_data, start=1):
        cleaned = clean_record(record, index)

        if cleaned is None:
            continue

        pmid = cleaned["pmid"]

        if pmid in seen_ids:
            logger.warning("Skipping duplicate PMID: %s", pmid)
            continue

        seen_ids.add(pmid)
        cleaned_records.append(cleaned)

    logger.info(
        "Cleaned %s usable records from %s input records.",
        len(cleaned_records),
        len(raw_data)
    )

    return cleaned_records


# --------------------------------------------------
# LLM REQUEST PAYLOAD
# --------------------------------------------------

def build_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build an OpenAI-compatible chat completion request."""

    system_prompt = (
        "You are a biomedical research assistant. "
        "Summarize the clinical outcome reported in the abstract "
        "in exactly one concise sentence. "
        "Use only information explicitly present in the abstract. "
        "Do not invent results, conclusions, or clinical claims. "
        "If the abstract reports no significant improvement, "
        "state that clearly."
    )

    user_prompt = (
        f"Article title: {record['title']}\n"
        f"Abstract: {record['abstract']}\n\n"
        "Provide a one-sentence summary of the clinical outcome."
    )

    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 120,
        "stream": False
    }


# --------------------------------------------------
# API SUMMARIZATION WITH RETRIES
# --------------------------------------------------

def summarize(
    session: requests.Session,
    record: Dict[str, Any]
) -> str:
    """Send an abstract to the LLM API and return its summary."""

    payload = build_payload(record)

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.post(
                API_URL,
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )

            if response.status_code in RETRY_STATUS_CODES:
                logger.warning(
                    "Temporary API error %s for PMID %s "
                    "(attempt %s/%s).",
                    response.status_code,
                    record["pmid"],
                    attempt,
                    MAX_RETRIES
                )

                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE ** attempt)
                    continue

                response.raise_for_status()

            response.raise_for_status()

            result = response.json()

            summary = result["choices"][0]["message"]["content"]

            summary = normalize_text(summary)

            if not summary:
                raise ValueError("API returned an empty summary.")

            return summary

        except (requests.Timeout, requests.ConnectionError) as error:
            logger.warning(
                "Connection/timeout error for PMID %s "
                "(attempt %s/%s): %s",
                record["pmid"],
                attempt,
                MAX_RETRIES,
                error
            )

        except requests.HTTPError as error:
            status_code = (
                error.response.status_code
                if error.response is not None
                else None
            )

            if status_code not in RETRY_STATUS_CODES:
                logger.error(
                    "Non-retryable HTTP error for PMID %s: %s",
                    record["pmid"],
                    error
                )
                raise

            logger.warning("Retryable HTTP error: %s", error)

        except (ValueError, KeyError, IndexError, TypeError) as error:
            logger.warning(
                "Invalid API response for PMID %s: %s",
                record["pmid"],
                error
            )

        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE ** attempt

            logger.info("Retrying in %s seconds...", delay)
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to summarize PMID {record['pmid']} "
        f"after {MAX_RETRIES} attempts."
    )


# --------------------------------------------------
# PROCESS RECORDS
# --------------------------------------------------

def process_records(
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Summarize all valid records sequentially."""

    results = []

    with requests.Session() as session:

        for record in records:

            logger.info(
                "Processing PMID: %s",
                record["pmid"]
            )

            try:
                summary = summarize(session, record)

                results.append({
                    "pmid": record["pmid"],
                    "title": record["title"],
                    "authors": record["authors"],
                    "clinical_outcome_summary": summary,
                    "status": "success"
                })

                logger.info(
                    "Successfully summarized PMID: %s",
                    record["pmid"]
                )

            except Exception as error:
                logger.error(
                    "Failed to process PMID %s: %s",
                    record["pmid"],
                    error
                )

                results.append({
                    "pmid": record["pmid"],
                    "title": record["title"],
                    "authors": record["authors"],
                    "clinical_outcome_summary": None,
                    "status": "failed",
                    "error": str(error)
                })

    return results


# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------

def save_results(
    results: List[Dict[str, Any]],
    output_file: str
) -> None:
    """Save summaries as formatted JSON."""

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info("Output saved to: %s", output_file)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description="PubMed Clinical Outcome Summarizer"
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Path to input PubMed JSON file."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="pubmed_summaries.json",
        help="Path to output JSON file."
    )

    args = parser.parse_args()

    try:
        records = load_and_clean(args.input)

        if not records:
            logger.warning("No valid records to process.")
            save_results([], args.output)
            return 0

        results = process_records(records)

        save_results(results, args.output)

        successful = sum(
            1 for item in results
            if item["status"] == "success"
        )

        failed = len(results) - successful

        logger.info("Processing completed.")
        logger.info("Successful: %s", successful)
        logger.info("Failed: %s", failed)

        return 0 if failed == 0 else 1

    except Exception:
        logger.exception("Program execution failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
