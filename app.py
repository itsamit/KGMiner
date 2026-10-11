
"""
KGMiner v4
============================================================

Standalone Scientific Knowledge Graph Mining Backend

Pipeline
--------

User Goal
   |
   v
LLM Search Query Generation
   |
   v
PubMed Search
   |
   v
PubMed Abstract Retrieval
   |
   v
LLM Knowledge Graph Extraction
   |
   +---- HeliosX
   |
   +---- Groq
   |
   v
Validation
   |
   v
Deterministic Normalization
   |
   v
knowledge_graph.xlsx
   |
   v
Local KG Search
   |
   +---- Deterministic Answer
   |
   +---- Optional LLM Answer Synthesis
   |
   v
Final Scientific Answer

Run:
    uvicorn app:app --reload --port 5678

Required:
    pip install fastapi uvicorn requests pandas openpyxl python-dotenv

Recommended .env:

    HELIOSX_API_KEY=your_key
    GROQ_API_KEY=your_key

Optional:

    HELIOSX_BASE_URL=https://api.heliosx.ai/v1
    GROQ_BASE_URL=https://api.groq.com/openai/v1

    HELIOSX_EXTRACTION_MODEL=google/gemini-2.5-pro
    HELIOSX_ANSWER_MODEL=google/gemini-2.5-pro

    GROQ_EXTRACTION_MODEL=openai/gpt-oss-120b
    GROQ_ANSWER_MODEL=openai/gpt-oss-120b

Important:
    API keys must remain in .env.
"""


# ============================================================================
# 0. IMPORTS
# ============================================================================

import sys
import io
import os
import re
import json
import uuid
import shutil
import logging
import threading
import time
import math
import html
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import requests
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from dotenv import load_dotenv


# ============================================================================
# 1. UTF-8 OUTPUT
# ============================================================================

try:

    if sys.stdout.encoding != "utf-8":

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace"
        )

    if sys.stderr.encoding != "utf-8":

        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace"
        )

except Exception:

    pass


# ============================================================================
# 2. LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "[%(levelname)s] "
        "%(name)s: "
        "%(message)s"
    ),
)

logger = logging.getLogger("kgminer")


# ============================================================================
# 3. ENVIRONMENT
# ============================================================================

load_dotenv()


def env_int(
    name,
    default
):

    try:

        return int(
            os.getenv(
                name,
                str(default)
            )
        )

    except Exception:

        return default


def env_float(
    name,
    default
):

    try:

        return float(
            os.getenv(
                name,
                str(default)
            )
        )

    except Exception:

        return default


def env_bool(
    name,
    default=False
):

    value = os.getenv(
        name,
        str(default)
    ).strip().lower()

    return value in (
        "1",
        "true",
        "yes",
        "y",
        "on"
    )


# ============================================================================
# 4. API KEY COLLECTION
# ============================================================================

def collect_keys(
    prefixes
):

    found = []

    for prefix in prefixes:

        base = os.getenv(
            prefix,
            ""
        ).strip()

        if base:

            found.append(
                base
            )

        for i in range(
            1,
            101
        ):

            value = os.getenv(
                f"{prefix}_{i}",
                ""
            ).strip()

            if value:

                found.append(
                    value
                )

    unique = []

    for key in found:

        if key and key not in unique:

            unique.append(
                key
            )

    return unique


GROQ_KEYS = collect_keys([
    "GROQ_API_KEY"
])


HELIOSX_KEYS = collect_keys([
    "HELIOSX_API_KEY",
    "HELIUS_API_KEY"
])


# ============================================================================
# 5. PROVIDER CONFIGURATION
# ============================================================================

GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
).rstrip("/")


HELIOSX_BASE_URL = os.getenv(
    "HELIOSX_BASE_URL",
    "https://api.heliosx.ai/v1"
).rstrip("/")


# ---------------------------------------------------------------------------
# Candidate models
#
# These are fallback candidates.
# The application first tries to discover models available to the key.
# ---------------------------------------------------------------------------

GROQ_EXTRACTION_MODELS = [
    os.getenv(
        "GROQ_EXTRACTION_MODEL",
        "openai/gpt-oss-120b"
    ),
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]


GROQ_ANSWER_MODELS = [
    os.getenv(
        "GROQ_ANSWER_MODEL",
        "openai/gpt-oss-120b"
    ),
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]


HELIOSX_EXTRACTION_MODELS = [
    os.getenv(
        "HELIOSX_EXTRACTION_MODEL",
        "google/gemini-2.5-pro"
    ),
    "anthropic/claude-opus-4.6",
    "openai/gpt-4.1",
]


HELIOSX_ANSWER_MODELS = [
    os.getenv(
        "HELIOSX_ANSWER_MODEL",
        "google/gemini-2.5-pro"
    ),
    "anthropic/claude-opus-4.6",
    "openai/gpt-4.1",
]


REQUEST_TIMEOUT = env_int(
    "LLM_REQUEST_TIMEOUT",
    120
)


LLM_RETRIES = env_int(
    "LLM_RETRIES",
    2
)


EXTRACTION_TEMPERATURE = env_float(
    "EXTRACTION_TEMPERATURE",
    0.0
)


ANSWER_TEMPERATURE = env_float(
    "ANSWER_TEMPERATURE",
    0.2
)


# If True, answer synthesis will use the LLM.
# If False, deterministic KG answers are returned.
ENABLE_LLM_ANSWER = env_bool(
    "ENABLE_LLM_ANSWER",
    True
)


# ============================================================================
# 6. PROVIDER CLIENT
# ============================================================================

class LLMProvider:

    def __init__(
        self,
        name,
        keys,
        base_url,
        models
    ):

        self.name = name

        self.keys = keys

        self.base_url = base_url

        self.models = models

        self.lock = threading.Lock()

        self.key_index = 0

        self.cooldowns = {}

        self.available_models = None

        self.models_checked_at = 0


    # ---------------------------------------------------------------------

    def available(self):

        return bool(
            self.keys
        )


    # ---------------------------------------------------------------------

    def _next_key(self):

        if not self.keys:

            return None

        with self.lock:

            now = time.time()

            for _ in range(
                len(self.keys)
            ):

                key = self.keys[
                    self.key_index
                ]

                self.key_index = (
                    self.key_index + 1
                ) % len(self.keys)

                cooldown = self.cooldowns.get(
                    key,
                    0
                )

                if cooldown <= now:

                    return key

            return self.keys[
                self.key_index
            ]


    # ---------------------------------------------------------------------

    def _cooldown_key(
        self,
        key,
        seconds=30
    ):

        with self.lock:

            self.cooldowns[
                key
            ] = (
                time.time() +
                seconds
            )


    # ---------------------------------------------------------------------

    def discover_models(
        self,
        force=False
    ):

        if not self.keys:

            return []


        if (
            not force
            and
            self.available_models is not None
            and
            time.time() -
            self.models_checked_at < 300
        ):

            return self.available_models


        key = self._next_key()

        if not key:

            return []


        url = (
            self.base_url +
            "/models"
        )


        headers = {
            "Authorization":
                f"Bearer {key}"
        }


        try:

            logger.info(
                "Discovering models "
                "provider=%s",
                self.name
            )

            response = requests.get(
                url,
                headers=headers,
                timeout=30
            )


            if response.status_code != 200:

                logger.warning(
                    "%s /models failed "
                    "HTTP %s: %s",
                    self.name,
                    response.status_code,
                    response.text[:500]
                )

                return []


            data = response.json()


            models = []

            for item in data.get(
                "data",
                []
            ):

                if not isinstance(
                    item,
                    dict
                ):

                    continue

                model_id = item.get(
                    "id"
                )

                if model_id:

                    models.append(
                        str(model_id)
                    )


            self.available_models = (
                list(
                    dict.fromkeys(
                        models
                    )
                )
            )

            self.models_checked_at = (
                time.time()
            )


            logger.info(
                "%s available models=%d",
                self.name,
                len(
                    self.available_models
                )
            )


            return self.available_models


        except Exception as exc:

            logger.warning(
                "%s model discovery "
                "error: %s",
                self.name,
                exc
            )

            return []


    # ---------------------------------------------------------------------

    def candidate_models(self):

        discovered = (
            self.discover_models()
        )


        # If the provider exposes /models,
        # prefer models actually available
        # to this key.

        if discovered:

            preferred = [
                model
                for model in self.models
                if model in discovered
            ]


            # Remaining discovered text models.
            #
            # Do not blindly use audio/embedding/
            # moderation models.

            usable = []

            for model in discovered:

                lower = model.lower()

                if any(
                    term in lower
                    for term in [
                        "embedding",
                        "whisper",
                        "moderation",
                        "guard",
                        "tts",
                        "speech",
                        "image"
                    ]
                ):

                    continue

                usable.append(
                    model
                )


            remaining = [
                model
                for model in usable
                if model not in preferred
            ]


            candidates = (
                preferred +
                remaining
            )


            if candidates:

                return candidates


        return list(
            dict.fromkeys(
                self.models
            )
        )


    # ---------------------------------------------------------------------

    def _send_chat(
        self,
        key,
        model,
        messages,
        temperature,
        max_tokens,
        response_format
    ):

        url = (
            self.base_url +
            "/chat/completions"
        )


        headers = {

            "Authorization":
                f"Bearer {key}",

            "Content-Type":
                "application/json",
        }


        payload = {

            "model":
                model,

            "messages":
                messages,

            "temperature":
                temperature,

            "max_tokens":
                max_tokens,
        }


        if response_format is not None:

            payload[
                "response_format"
            ] = response_format


        return requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )


    # ---------------------------------------------------------------------

    def chat(
        self,
        messages,
        temperature=0.2,
        max_tokens=4096,
        response_format=None,
        model=None
    ):

        if not self.keys:

            raise RuntimeError(
                f"{self.name}: "
                f"no API keys configured"
            )


        if model:

            models = [
                model
            ]

        else:

            models = (
                self.candidate_models()
            )


        if not models:

            raise RuntimeError(
                f"{self.name}: "
                f"no candidate models"
            )


        last_error = None


        for current_model in models:

            for attempt in range(
                LLM_RETRIES + 1
            ):

                key = self._next_key()

                if not key:

                    break


                logger.info(
                    "LLM request "
                    "provider=%s "
                    "model=%s "
                    "key=...%s",
                    self.name,
                    current_model,
                    key[-4:]
                )


                try:

                    response = self._send_chat(
                        key,
                        current_model,
                        messages,
                        temperature,
                        max_tokens,
                        response_format
                    )


                    # ====================================================
                    # SUCCESS
                    # ====================================================

                    if response.status_code == 200:

                        data = response.json()


                        choices = data.get(
                            "choices",
                            []
                        )


                        if not choices:

                            raise RuntimeError(
                                f"{self.name}: "
                                f"empty choices"
                            )


                        content = (
                            choices[0]
                            .get(
                                "message",
                                {}
                            )
                            .get(
                                "content",
                                ""
                            )
                        )


                        if content:

                            logger.info(
                                "LLM success "
                                "provider=%s "
                                "model=%s",
                                self.name,
                                current_model
                            )

                            return content


                        raise RuntimeError(
                            f"{self.name}: "
                            f"empty response"
                        )


                    # ====================================================
                    # RESPONSE FORMAT NOT SUPPORTED
                    # ====================================================

                    if (
                        response.status_code
                        in (400, 422)
                        and
                        response_format is not None
                    ):

                        logger.warning(
                            "%s model=%s "
                            "rejected response_format. "
                            "Retrying without structured output.",
                            self.name,
                            current_model
                        )


                        fallback = self._send_chat(
                            key,
                            current_model,
                            messages,
                            temperature,
                            max_tokens,
                            None
                        )


                        if (
                            fallback.status_code
                            == 200
                        ):

                            data = (
                                fallback.json()
                            )

                            choices = (
                                data.get(
                                    "choices",
                                    []
                                )
                            )

                            if choices:

                                content = (
                                    choices[0]
                                    .get(
                                        "message",
                                        {}
                                    )
                                    .get(
                                        "content",
                                        ""
                                    )
                                )

                                if content:

                                    return content


                    # ====================================================
                    # RATE LIMIT
                    # ====================================================

                    if response.status_code == 429:

                        logger.warning(
                            "%s rate limited "
                            "model=%s",
                            self.name,
                            current_model
                        )


                        self._cooldown_key(
                            key,
                            30
                        )


                        last_error = RuntimeError(
                            f"{self.name} "
                            f"HTTP 429"
                        )


                        time.sleep(
                            min(
                                5 * (
                                    attempt + 1
                                ),
                                15
                            )
                        )


                        continue


                    # ====================================================
                    # AUTH / BILLING
                    # ====================================================

                    if response.status_code in (
                        401,
                        402,
                        403
                    ):

                        logger.error(
                            "%s authentication/"
                            "billing error HTTP %s",
                            self.name,
                            response.status_code
                        )


                        self._cooldown_key(
                            key,
                            300
                        )


                        last_error = RuntimeError(
                            f"{self.name} "
                            f"HTTP "
                            f"{response.status_code}: "
                            f"{response.text[:500]}"
                        )


                        break


                    # ====================================================
                    # MODEL NOT FOUND
                    # ====================================================

                    if response.status_code == 404:

                        logger.warning(
                            "%s model unavailable: "
                            "%s",
                            self.name,
                            current_model
                        )


                        last_error = RuntimeError(
                            f"{self.name} "
                            f"model "
                            f"{current_model} "
                            f"unavailable"
                        )


                        # Do not retry this model.
                        break


                    # ====================================================
                    # OTHER ERROR
                    # ====================================================

                    last_error = RuntimeError(
                        f"{self.name} HTTP "
                        f"{response.status_code}: "
                        f"{response.text[:500]}"
                    )


                    logger.warning(
                        "%s request failed: %s",
                        self.name,
                        last_error
                    )


                    time.sleep(
                        min(
                            2 * (
                                attempt + 1
                            ),
                            10
                        )
                    )


                except requests.RequestException as exc:

                    last_error = exc


                    logger.warning(
                        "%s network error: %s",
                        self.name,
                        exc
                    )


                    time.sleep(
                        min(
                            2 * (
                                attempt + 1
                            ),
                            10
                        )
                    )


        raise RuntimeError(
            f"{self.name}: "
            f"all models/keys failed. "
            f"Last error: {last_error}"
        )


# ============================================================================
# 7. PROVIDERS
# ============================================================================

groq_provider = LLMProvider(
    name="groq",
    keys=GROQ_KEYS,
    base_url=GROQ_BASE_URL,
    models=GROQ_ANSWER_MODELS
)


heliosx_provider = LLMProvider(
    name="heliosx",
    keys=HELIOSX_KEYS,
    base_url=HELIOSX_BASE_URL,
    models=HELIOSX_ANSWER_MODELS
)


# HeliosX first, Groq second.
PROVIDERS = [
    heliosx_provider,
    groq_provider
]


# ============================================================================
# 8. LLM FALLBACK
# ============================================================================

def llm_call(
    messages,
    purpose="answer",
    temperature=0.2,
    max_tokens=4096,
    response_format=None
):

    if purpose == "extraction":

        provider_configs = [

            (
                heliosx_provider,
                HELIOSX_EXTRACTION_MODELS
            ),

            (
                groq_provider,
                GROQ_EXTRACTION_MODELS
            )
        ]

    else:

        provider_configs = [

            (
                heliosx_provider,
                HELIOSX_ANSWER_MODELS
            ),

            (
                groq_provider,
                GROQ_ANSWER_MODELS
            )
        ]


    errors = []


    for provider, configured_models in (
        provider_configs
    ):

        if not provider.available():

            errors.append(
                f"{provider.name}: "
                f"no API keys"
            )

            continue


        # Temporarily use purpose-specific models.
        old_models = provider.models

        provider.models = (
            configured_models
        )


        try:

            return provider.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )


        except Exception as exc:

            logger.warning(
                "Provider %s failed: %s",
                provider.name,
                exc
            )


            errors.append(
                f"{provider.name}: "
                f"{exc}"
            )


        finally:

            provider.models = old_models


    raise RuntimeError(
        "All LLM providers failed: "
        +
        " | ".join(
            errors
        )
    )


# ============================================================================
# 9. STATUS
# ============================================================================

def provider_status():

    return {

        "groq": {

            "configured":
                bool(GROQ_KEYS),

            "keys":
                len(GROQ_KEYS),

            "models":
                GROQ_ANSWER_MODELS,

            "base_url":
                GROQ_BASE_URL
        },

        "heliosx": {

            "configured":
                bool(HELIOSX_KEYS),

            "keys":
                len(HELIOSX_KEYS),

            "models":
                HELIOSX_ANSWER_MODELS,

            "base_url":
                HELIOSX_BASE_URL
        }
    }


# ============================================================================
# 10. PUBMED
# ============================================================================

PUBMED_BASE = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils"
)


def clean_xml_text(
    value
):

    if not value:

        return ""

    value = re.sub(
        r"<[^>]+>",
        " ",
        str(value)
    )

    value = html.unescape(
        value
    )

    # Excel/XML 1.0 permits TAB, LF and CR, but rejects most C0 controls.
    # Remove those characters here as well as at final workbook export.
    value = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        "",
        value
    )

    # Some biomedical text sources replace the Greek beta with a control
    # character. Recover the common beta-carotene corruption when possible.
    value = re.sub(
        r"(?i)(?:[\x00-\x1F])+-carotene",
        "β-carotene",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ---------------------------------------------------------------------------

def search_pubmed(
    query,
    retmax=100
):

    url = (
        PUBMED_BASE +
        "/esearch.fcgi"
    )


    params = {

        "db":
            "pubmed",

        "term":
            query,

        "retmode":
            "json",

        "retmax":
            retmax
    }


    response = requests.get(
        url,
        params=params,
        timeout=60
    )


    response.raise_for_status()


    data = response.json()


    return (
        data
        .get(
            "esearchresult",
            {}
        )
        .get(
            "idlist",
            []
        )
    )


# ---------------------------------------------------------------------------

def fetch_pubmed_articles(
    pmids
):

    if not pmids:

        return []


    url = (
        PUBMED_BASE +
        "/efetch.fcgi"
    )


    params = {

        "db":
            "pubmed",

        "id":
            ",".join(pmids),

        "retmode":
            "xml"
    }


    response = requests.get(
        url,
        params=params,
        timeout=120
    )


    response.raise_for_status()


    xml = response.text


    articles = []


    blocks = re.findall(
        r"<PubmedArticle>(.*?)"
        r"</PubmedArticle>",
        xml,
        flags=re.DOTALL
    )


    for block in blocks:

        pmid_match = re.search(
            r"<PMID[^>]*>"
            r"(.*?)"
            r"</PMID>",
            block,
            flags=re.DOTALL
        )


        title_match = re.search(
            r"<ArticleTitle>"
            r"(.*?)"
            r"</ArticleTitle>",
            block,
            flags=re.DOTALL
        )


        journal_match = re.search(
            r"<Journal>"
            r".*?"
            r"<Title>"
            r"(.*?)"
            r"</Title>",
            block,
            flags=re.DOTALL
        )


        abstract_parts = re.findall(
            r"<AbstractText"
            r"[^>]*>"
            r"(.*?)"
            r"</AbstractText>",
            block,
            flags=re.DOTALL
        )


        if not abstract_parts:

            continue


        pmid = (
            clean_xml_text(
                pmid_match.group(1)
            )
            if pmid_match
            else ""
        )


        title = (
            clean_xml_text(
                title_match.group(1)
            )
            if title_match
            else ""
        )


        journal = (
            clean_xml_text(
                journal_match.group(1)
            )
            if journal_match
            else ""
        )


        abstract = " ".join(
            clean_xml_text(
                x
            )
            for x in abstract_parts
        )


        articles.append({

            "PMID":
                pmid,

            "Title":
                title,

            "Journal":
                journal,

            "Abstract":
                abstract
        })


    return articles


# ============================================================================
# 11. SEARCH QUERY GENERATION
# ============================================================================

SEARCH_QUERY_SYSTEM = """
You are a biomedical literature search specialist.

Generate 3-5 complementary PubMed search queries for the user's
scientific research question.

The queries must help retrieve papers containing:
- biological mechanisms
- pathways
- enzymes
- genes
- organisms
- metabolites
- production relationships
- engineering strategies
- experimental evidence

Rules:
1. Use PubMed-compatible syntax.
2. Do not invent organisms, genes or molecules.
3. Use synonyms when useful.
4. Keep queries focused.
5. Return ONLY JSON.

Format:

{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}
"""


def generate_search_queries(
    goal
):

    prompt = f"""
Scientific research question:

{goal}

Generate 3-5 complementary PubMed queries.
"""


    try:

        response = llm_call(
            [
                {
                    "role":
                        "system",
                    "content":
                        SEARCH_QUERY_SYSTEM
                },

                {
                    "role":
                        "user",
                    "content":
                        prompt
                }
            ],

            purpose="extraction",

            temperature=0.0,

            max_tokens=1000,

            response_format={
                "type":
                    "json_object"
            }
        )


        data = json.loads(
            response
        )


        queries = data.get(
            "queries",
            []
        )


        queries = [
            str(q).strip()
            for q in queries
            if str(q).strip()
        ]


        if queries:

            return list(
                dict.fromkeys(
                    queries
                )
            )[:5]


    except Exception as exc:

        logger.warning(
            "Query generation failed: %s",
            exc
        )


    # Deterministic fallback.
    return [
        goal,
        f'"{goal}"'
    ]


# ============================================================================
# 12. CHUNKING
# ============================================================================

def chunk_text(
    text,
    max_chars=8500
):

    text = str(
        text
    ).strip()


    if len(text) <= max_chars:

        return [
            text
        ]


    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )


    chunks = []

    current = ""


    for sentence in sentences:

        if (
            len(current)
            +
            len(sentence)
            +
            1
            <= max_chars
        ):

            current += (
                (
                    " "
                    if current
                    else ""
                )
                +
                sentence
            )

        else:

            if current:

                chunks.append(
                    current
                )

            current = sentence


    if current:

        chunks.append(
            current
        )


    return chunks


# ============================================================================
# 13. ONTOLOGY
# ============================================================================

VALID_RELATIONS = {

    "activates",

    "inhibits",

    "produces",

    "is_variant_of",

    "encodes",

    "is_host_for",

    "increases",

    "decreases",

    "integrated_in",

    "has_capability",

    "is_a",

    "has_metric",

    "improves",

    "affects",

    "depends_on",

    "enables",

    "prevents",

    "contains",

    "exhibits",

    "related_to"
}


RELATION_ALIASES = {

    "activate":
        "activates",

    "activated":
        "activates",

    "inhibit":
        "inhibits",

    "inhibited":
        "inhibits",

    "produce":
        "produces",

    "produced":
        "produces",

    "production":
        "produces",

    "encode":
        "encodes",

    "encoded":
        "encodes",

    "host":
        "is_host_for",

    "host_for":
        "is_host_for",

    "increase":
        "increases",

    "increased":
        "increases",

    "decrease":
        "decreases",

    "decreased":
        "decreases",

    "integrated":
        "integrated_in",

    "capability":
        "has_capability",

    "type_of":
        "is_a",

    "type":
        "is_a",

    "metric":
        "has_metric",

    "improve":
        "improves",

    "affect":
        "affects",

    "depend_on":
        "depends_on",

    "enable":
        "enables",

    "prevent":
        "prevents",

    "contain":
        "contains",

    "exhibit":
        "exhibits",

    "related":
        "related_to"
}


# ============================================================================
# 14. TRIPLE CLEANING
# ============================================================================

TRIPLE_REGEX = re.compile(
    r"""
    \(
        \s*
        ([^,\n]+?)
        \s*,\s*
        ([^,\n]+?)
        \s*,\s*
        ([^)]+?)
        \s*
    \)
    """,
    flags=re.VERBOSE
)


def normalize_relation(
    relation
):

    relation = str(
        relation
    ).strip().lower()


    relation = re.sub(
        r"\s+",
        "_",
        relation
    )


    relation = relation.replace(
        "-",
        "_"
    )


    relation = RELATION_ALIASES.get(
        relation,
        relation
    )


    if relation in VALID_RELATIONS:

        return relation


    if "produce" in relation:

        return "produces"


    if "inhibit" in relation:

        return "inhibits"


    if "activat" in relation:

        return "activates"


    if "encod" in relation:

        return "encodes"


    if "increase" in relation:

        return "increases"


    if "decrease" in relation:

        return "decreases"


    if "metric" in relation:

        return "has_metric"


    if "host" in relation:

        return "is_host_for"


    if "variant" in relation:

        return "is_variant_of"


    if "integrat" in relation:

        return "integrated_in"


    if "capab" in relation:

        return "has_capability"


    if "improv" in relation:

        return "improves"


    if "affect" in relation:

        return "affects"


    if "depend" in relation:

        return "depends_on"


    if "enable" in relation:

        return "enables"


    if "prevent" in relation:

        return "prevents"


    if "contain" in relation:

        return "contains"


    if "exhibit" in relation:

        return "exhibits"


    if (
        relation in
        (
            "type",
            "typeof",
            "type_of"
        )
    ):

        return "is_a"


    return "related_to"


# ---------------------------------------------------------------------------

def clean_entity(
    value
):

    value = str(
        value
    ).strip()

    # Normalize malformed/control characters before graph construction.
    value = clean_xml_text(value)


    value = value.strip(
        "\"' "
    )


    value = re.sub(
        r"\s+",
        " ",
        value
    )


    value = value.strip(
        ".,;:"
    )


    return value


# ---------------------------------------------------------------------------

def clean_object(
    value
):

    value = clean_entity(
        value
    )


    value = re.sub(
        r"\b(\w+)\s+\1\b",
        r"\1",
        value,
        flags=re.IGNORECASE
    )


    return value


# ---------------------------------------------------------------------------

def parse_triples(
    text
):

    triples = []


    for match in TRIPLE_REGEX.finditer(
        str(text or "")
    ):

        s = clean_entity(
            match.group(1)
        )

        r = normalize_relation(
            match.group(2)
        )

        o = clean_object(
            match.group(3)
        )


        if not s or not r or not o:

            continue


        if s.lower() == o.lower():

            continue


        triples.append(
            (
                s,
                r,
                o
            )
        )


    return normalize_triple_set(
        triples
    )


# ============================================================================
# 15. JSON TRIPLE PARSER
# ============================================================================

TRIPLE_SCHEMA = {

    "type":
        "json_schema",

    "json_schema": {

        "name":
            "knowledge_graph_triples",

        "schema": {

            "type":
                "object",

            "properties": {

                "triples": {

                    "type":
                        "array",

                    "items": {

                        "type":
                            "object",

                        "properties": {

                            "subject": {
                                "type":
                                    "string"
                            },

                            "relation": {
                                "type":
                                    "string"
                            },

                            "object": {
                                "type":
                                    "string"
                            }
                        },

                        "required": [
                            "subject",
                            "relation",
                            "object"
                        ],

                        "additionalProperties":
                            False
                    }
                }
            },

            "required": [
                "triples"
            ],

            "additionalProperties":
                False
        }
    }
}


# ---------------------------------------------------------------------------

def parse_json_triples(
    text
):

    if not text:

        return []


    data = None


    try:

        data = json.loads(
            text
        )

    except Exception:

        # Remove markdown fences.
        cleaned = re.sub(
            r"```json|```",
            "",
            str(text),
            flags=re.IGNORECASE
        ).strip()


        try:

            data = json.loads(
                cleaned
            )

        except Exception:

            match = re.search(
                r"\{.*\}",
                cleaned,
                flags=re.DOTALL
            )


            if match:

                try:

                    data = json.loads(
                        match.group(0)
                    )

                except Exception:

                    data = None


    if not isinstance(
        data,
        dict
    ):

        return []


    triples = []


    for item in data.get(
        "triples",
        []
    ):

        if not isinstance(
            item,
            dict
        ):

            continue


        s = clean_entity(
            item.get(
                "subject",
                ""
            )
        )


        r = normalize_relation(
            item.get(
                "relation",
                ""
            )
        )


        o = clean_object(
            item.get(
                "object",
                ""
            )
        )


        if not s or not r or not o:

            continue


        if s.lower() == o.lower():

            continue


        triples.append(
            (
                s,
                r,
                o
            )
        )


    return normalize_triple_set(
        triples
    )


# ============================================================================
# 16. EXTRACTION PROMPT
# ============================================================================

EXTRACTION_SYSTEM = """
You are KGMiner, a biomedical and biological knowledge graph extraction
engine.

Extract ONLY relationships explicitly supported by the supplied scientific
abstract.

Do NOT use outside knowledge.

Do NOT infer a pathway merely because two compounds are known to be in the
same pathway.

Do NOT invent enzymes, genes, organisms or reactions.

ONTOLOGY:

activates
inhibits
produces
is_variant_of
encodes
is_host_for
increases
decreases
integrated_in
has_capability
is_a
has_metric
improves
affects
depends_on
enables
prevents
contains
exhibits
related_to

IMPORTANT:

If the abstract says:

"strain X produced 39.5 g/L beta-carotene"

extract:

(strain X, produces, beta-carotene)
(strain X, has_metric, "39.5 g/L")

NOT:

(strain X, produces, 39.5 g/L beta-carotene)

If the abstract says:

"overexpression of crtY increased beta-carotene production"

extract:

(crtY, increases, beta-carotene production)

If the abstract explicitly states:

"strain X is a host for beta-carotene production"

extract:

(strain X, is_host_for, beta-carotene production)

Use exact terminology from the abstract whenever possible.

Return ONLY:

{
  "triples": [
    {
      "subject": "...",
      "relation": "...",
      "object": "..."
    }
  ]
}
"""


# ============================================================================
# 17. EXTRACT FROM ONE ABSTRACT
# ============================================================================

def extract_triples_from_text(
    abstract,
    goal
):

    chunks = chunk_text(
        abstract
    )


    collected = set()


    for chunk_index, chunk in enumerate(
        chunks
    ):

        user_prompt = f"""
Research question:

{goal}

Scientific abstract:

{chunk}

Extract all explicitly supported biological, biochemical, engineering,
production and quantitative relationships.

This is extraction chunk {chunk_index + 1} of {len(chunks)}.

Return ONLY JSON.
"""


        try:

            response = llm_call(
                [
                    {
                        "role":
                            "system",

                        "content":
                            EXTRACTION_SYSTEM
                    },

                    {
                        "role":
                            "user",

                        "content":
                            user_prompt
                    }
                ],

                purpose="extraction",

                temperature=0.0,

                max_tokens=5000,

                response_format=TRIPLE_SCHEMA
            )


            triples = parse_json_triples(
                response
            )


            if not triples:

                triples = parse_triples(
                    response
                )


            collected.update(
                triples
            )


        except Exception as exc:

            logger.warning(
                "Extraction chunk %d failed: %s",
                chunk_index + 1,
                exc
            )


    return sorted(
        collected
    )


# ============================================================================
# 18. VALIDATION
# ============================================================================

VALIDATION_SYSTEM = """
You are a strict biomedical knowledge graph validator.

Given a scientific abstract and candidate triples:

1. Remove unsupported triples.
2. Correct relation names.
3. Keep only relationships explicitly stated or directly supported.
4. Preserve numerical values as has_metric.
5. Add a missing relationship only if it is explicitly present in the
   abstract.
6. Do not use outside knowledge.

Allowed relations:

activates
inhibits
produces
is_variant_of
encodes
is_host_for
increases
decreases
integrated_in
has_capability
is_a
has_metric
improves
affects
depends_on
enables
prevents
contains
exhibits
related_to

Return ONLY JSON.
"""


def validate_triples(
    abstract,
    goal,
    candidate_triples
):

    if not candidate_triples:

        return []


    candidate_text = "\n".join(

        f"({s}, {r}, {o})"

        for s, r, o
        in candidate_triples
    )


    user_prompt = f"""
Research question:

{goal}

ABSTRACT:

{abstract}

CANDIDATE TRIPLES:

{candidate_text}

Validate these triples against the abstract.

Return ONLY JSON.
"""


    try:

        response = llm_call(
            [
                {
                    "role":
                        "system",

                    "content":
                        VALIDATION_SYSTEM
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt
                }
            ],

            purpose="extraction",

            temperature=0.0,

            max_tokens=5000,

            response_format=TRIPLE_SCHEMA
        )


        triples = parse_json_triples(
            response
        )


        if triples:

            return triples


    except Exception as exc:

        logger.warning(
            "Validation failed: %s",
            exc
        )


    return candidate_triples


# ============================================================================
# 19. TRIPLE NORMALIZATION
# ============================================================================

def normalize_triple_set(
    triples
):

    normalized = set()


    for triple in triples:

        if not isinstance(
            triple,
            (list, tuple)
        ):

            continue


        if len(triple) != 3:

            continue


        s, r, o = triple


        s = clean_entity(
            s
        )


        r = normalize_relation(
            r
        )


        o = clean_object(
            o
        )


        if not s or not r or not o:

            continue


        if s.lower() == o.lower():

            continue


        normalized.add(
            (
                s,
                r,
                o
            )
        )


    return sorted(
        normalized,
        key=lambda x: (
            x[0].lower(),
            x[1].lower(),
            x[2].lower()
        )
    )


# ============================================================================
# 20. TRIPLE → SENTENCE
# ============================================================================

def triple_to_sentence(
    triple
):

    s, r, o = triple


    relation_map = {

        "activates":
            "activates",

        "inhibits":
            "inhibits",

        "produces":
            "produces",

        "is_variant_of":
            "is a variant of",

        "encodes":
            "encodes",

        "is_host_for":
            "serves as a host for",

        "increases":
            "increases",

        "decreases":
            "decreases",

        "integrated_in":
            "is integrated in",

        "has_capability":
            "has the capability for",

        "is_a":
            "is a type of",

        "has_metric":
            "has a reported metric of",

        "improves":
            "improves",

        "affects":
            "affects",

        "depends_on":
            "depends on",

        "enables":
            "enables",

        "prevents":
            "prevents",

        "contains":
            "contains",

        "exhibits":
            "exhibits",

        "related_to":
            "is related to"
    }


    natural = relation_map.get(
        r,
        r.replace(
            "_",
            " "
        )
    )


    return (
        f"{s} "
        f"{natural} "
        f"{o}."
    )


# ============================================================================
# 21. ARTICLE PROCESSING
# ============================================================================

def summarize_article(
    abstract
):

    try:

        return llm_call(
            [
                {
                    "role":
                        "system",

                    "content":
                        """
You are a scientific research analyst.

Summarize the supplied abstract in 1-2 precise
sentences.

Use ONLY information present in the abstract.
"""
                },

                {
                    "role":
                        "user",

                    "content":
                        abstract
                }
            ],

            purpose="answer",

            temperature=0.0,

            max_tokens=500
        ).strip()


    except Exception:

        return (
            "Summary unavailable."
        )


# ---------------------------------------------------------------------------

def process_article(
    article,
    goal
):

    abstract = str(
        article.get(
            "Abstract",
            ""
        )
    ).strip()


    if not abstract:

        return {

            **article,

            "Answer to Question 1":
                "No abstract available.",

            "Answer to Question 2":
                "No relationships found.",

            "Stability Score Q2":
                0.0,

            "Extraction Status":
                "no_abstract"
        }


    # ---------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------

    candidate_triples = (
        extract_triples_from_text(
            abstract,
            goal
        )
    )


    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validated_triples = (
        validate_triples(
            abstract,
            goal,
            candidate_triples
        )
    )


    validated_triples = (
        normalize_triple_set(
            validated_triples
        )
    )


    # ---------------------------------------------------------
    # Stability
    # ---------------------------------------------------------

    if candidate_triples:

        intersection = len(
            set(candidate_triples)
            &
            set(validated_triples)
        )


        union = len(
            set(candidate_triples)
            |
            set(validated_triples)
        )


        stability = (
            intersection / union
            if union
            else 1.0
        )

    else:

        stability = 0.0


    # ---------------------------------------------------------
    # Triple text
    # ---------------------------------------------------------

    if validated_triples:

        triple_text = "; ".join(

            f"({s}, {r}, {o})"

            for s, r, o
            in validated_triples
        )

        status = "success"

    else:

        triple_text = (
            "No relationships found."
        )

        status = "no_triples"


    # ---------------------------------------------------------
    # Summary
    #
    # Do NOT make summary failure affect extraction.
    # ---------------------------------------------------------

    summary = summarize_article(
        abstract
    )


    return {

        **article,

        "Answer to Question 1":
            summary,

        "Answer to Question 2":
            triple_text,

        "Stability Score Q2":
            round(
                stability,
                4
            ),

        "Extraction Status":
            status
    }


# ============================================================================
# 22. SOURCE-AWARE TRIPLE EXTRACTION
# ============================================================================

def extract_triples_with_sources(
    df
):

    rows = []


    if (
        "Answer to Question 2"
        not in df.columns
    ):

        return rows


    for _, row in df.iterrows():

        answer = str(
            row.get(
                "Answer to Question 2",
                ""
            )
        )


        pmid = str(
            row.get(
                "PMID",
                ""
            )
        )


        title = str(
            row.get(
                "Title",
                ""
            )
        )


        journal = str(
            row.get(
                "Journal",
                ""
            )
        )


        triples = parse_triples(
            answer
        )


        for s, r, o in triples:

            rows.append({

                "PMID":
                    pmid,

                "Title":
                    title,

                "Journal":
                    journal,

                "Subject":
                    s,

                "Relation":
                    r,

                "Object":
                    o
            })


    # Remove duplicates while preserving source.
    unique = {}

    for row in rows:

        key = (

            row["PMID"].lower(),

            row["Subject"].lower(),

            row["Relation"].lower(),

            row["Object"].lower()
        )


        unique[key] = row


    return list(
        unique.values()
    )


# ============================================================================
# 23. TOKENIZATION
# ============================================================================

STOPWORDS = {

    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "for",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "how",
    "does",
    "did",
    "do",
    "by",
    "from",
    "this",
    "that",
    "these",
    "those",
    "about",
    "pathway",
    "pathways",
    "relationship",
    "relationships",
    "role",
    "roles",
    "mechanism",
    "mechanisms"
}


def normalize_token(
    token
):

    token = token.lower().strip()


    # Simple biomedical normalization.
    replacements = {

        "produced":
            "produce",

        "produces":
            "produce",

        "production":
            "produce",

        "producing":
            "produce",

        "increased":
            "increase",

        "increases":
            "increase",

        "increasing":
            "increase",

        "decreased":
            "decrease",

        "decreases":
            "decrease",

        "inhibits":
            "inhibit",

        "inhibited":
            "inhibit",

        "activates":
            "activate",

        "activated":
            "activate"
    }


    return replacements.get(
        token,
        token
    )


# ---------------------------------------------------------------------------

def tokenize(
    text
):

    words = re.findall(
        r"[A-Za-z0-9_\-\u0370-\u03FF]+",
        str(text).lower()
    )


    tokens = []


    for word in words:

        if word in STOPWORDS:

            continue


        word = normalize_token(
            word
        )


        if word:

            tokens.append(
                word
            )


    return tokens


# ============================================================================
# 24. LOCAL KNOWLEDGE GRAPH
# ============================================================================

class LocalKnowledgeGraph:

    def __init__(
        self,
        triples,
        source_rows=None
    ):

        self.triples = (
            normalize_triple_set(
                triples
            )
        )


        self.inverted = defaultdict(
            set
        )


        self.entity_to_triples = (
            defaultdict(set)
        )


        self.sources = defaultdict(
            list
        )


        for index, triple in enumerate(
            self.triples
        ):

            s, r, o = triple


            text = (
                s +
                " " +
                r +
                " " +
                o
            )


            tokens = tokenize(
                text
            )


            for token in tokens:

                self.inverted[
                    token
                ].add(
                    index
                )


            self.entity_to_triples[
                s.lower()
            ].add(index)


            self.entity_to_triples[
                o.lower()
            ].add(index)


        if source_rows:

            for row in source_rows:

                triple = (

                    clean_entity(
                        row.get(
                            "Subject",
                            ""
                        )
                    ),

                    normalize_relation(
                        row.get(
                            "Relation",
                            ""
                        )
                    ),

                    clean_object(
                        row.get(
                            "Object",
                            ""
                        )
                    )
                )


                self.sources[
                    triple
                ].append(
                    row
                )


    # ---------------------------------------------------------------------

    def search(
        self,
        query,
        top_k=50
    ):

        q_tokens = tokenize(
            query
        )


        if not q_tokens:

            return []


        candidate_indexes = set()


        # ---------------------------------------------------------
        # Token-based candidate retrieval
        # ---------------------------------------------------------

        for token in q_tokens:

            candidate_indexes.update(
                self.inverted.get(
                    token,
                    set()
                )
            )


        # ---------------------------------------------------------
        # If no candidates were found,
        # perform substring retrieval.
        # ---------------------------------------------------------

        if not candidate_indexes:

            query_lower = (
                str(query)
                .lower()
            )


            for index, triple in enumerate(
                self.triples
            ):

                triple_text = (
                    " ".join(
                        triple
                    ).lower()
                )


                if any(
                    token in triple_text
                    for token in q_tokens
                ):

                    candidate_indexes.add(
                        index
                    )


        scored = []


        for index in candidate_indexes:

            triple = self.triples[
                index
            ]


            s, r, o = triple


            triple_tokens = set(
                tokenize(
                    " ".join(
                        triple
                    )
                )
            )


            entity_tokens = set(
                tokenize(
                    s +
                    " " +
                    o
                )
            )


            relation_tokens = set(
                tokenize(
                    r
                )
            )


            q_set = set(
                q_tokens
            )


            overlap = len(
                q_set &
                triple_tokens
            )


            entity_overlap = len(
                q_set &
                entity_tokens
            )


            relation_overlap = len(
                q_set &
                relation_tokens
            )


            if overlap == 0:

                continue


            # Base score.
            score = (
                overlap /
                math.sqrt(
                    max(
                        len(q_set),
                        1
                    )
                    *
                    max(
                        len(triple_tokens),
                        1
                    )
                )
            )


            # Stronger entity match.
            score += (
                0.25 *
                entity_overlap
            )


            # Relation match.
            score += (
                0.20 *
                relation_overlap
            )


            # Exact entity-name occurrence.
            q_lower = (
                str(query)
                .lower()
            )


            if (
                s.lower()
                in
                q_lower
            ):

                score += 0.50


            if (
                o.lower()
                in
                q_lower
            ):

                score += 0.50


            scored.append(
                (
                    triple,
                    score
                )
            )


        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )


        return scored[
            :top_k
        ]


# ============================================================================
# 25. EXCEL GRAPH LOADING
# ============================================================================

def load_source_rows(
    df
):

    rows = []


    if "Answer to Question 2" not in df.columns:

        return rows


    for _, row in df.iterrows():

        triples = parse_triples(
            row.get(
                "Answer to Question 2",
                ""
            )
        )


        for s, r, o in triples:

            rows.append({

                "PMID":
                    str(
                        row.get(
                            "PMID",
                            ""
                        )
                    ),

                "Title":
                    str(
                        row.get(
                            "Title",
                            ""
                        )
                    ),

                "Journal":
                    str(
                        row.get(
                            "Journal",
                            ""
                        )
                    ),

                "Subject":
                    s,

                "Relation":
                    r,

                "Object":
                    o
            })


    return rows


# ---------------------------------------------------------------------------

def extract_triples_from_dataframe(
    df
):

    rows = load_source_rows(
        df
    )


    triples = [

        (
            row["Subject"],
            row["Relation"],
            row["Object"]
        )

        for row in rows
    ]


    return normalize_triple_set(
        triples
    )


# ============================================================================
# 26. GRAPH CACHE
# ============================================================================

_graph_cache = {}

_graph_lock = threading.Lock()


def get_graph(
    goal_id,
    df
):

    with _graph_lock:

        cached = _graph_cache.get(
            goal_id
        )


        if cached is not None:

            return cached


    triples = (
        extract_triples_from_dataframe(
            df
        )
    )


    source_rows = (
        load_source_rows(
            df
        )
    )


    graph = SimpleGraph(
        triples
    )


    searcher = LocalKnowledgeGraph(
        triples,
        source_rows
    )


    result = (
        graph,
        searcher
    )


    with _graph_lock:

        _graph_cache[
            goal_id
        ] = result


    return result


# ============================================================================
# 27. SIMPLE GRAPH
# ============================================================================

class SimpleGraph:

    def __init__(
        self,
        triples
    ):

        self.triples = triples

        self.nodes = set()


        for s, r, o in triples:

            self.nodes.add(
                s
            )

            self.nodes.add(
                o
            )


    def number_of_nodes(self):

        return len(
            self.nodes
        )


    def number_of_edges(self):

        return len(
            self.triples
        )


# ============================================================================
# 28. SOURCE-AWARE DETERMINISTIC ANSWER
# ============================================================================

def deterministic_answer(
    query,
    search_results,
    searcher
):

    if not search_results:

        return (
            "No matching relationships were found "
            "in the current knowledge graph for this "
            "query."
        )


    # ---------------------------------------------------------
    # Group results by relationship type.
    # ---------------------------------------------------------

    groups = defaultdict(
        list
    )


    for triple, score in search_results:

        groups[
            triple[1]
        ].append(
            (
                triple,
                score
            )
        )


    answer_parts = []


    answer_parts.append(
        "### Knowledge graph results"
    )


    answer_parts.append(
        f"The knowledge graph contains "
        f"{len(search_results)} relevant "
        f"relationships for **{query}**."
    )


    # ---------------------------------------------------------
    # Production relationships first.
    # ---------------------------------------------------------

    priority = [

        "produces",

        "increases",

        "decreases",

        "activates",

        "inhibits",

        "encodes",

        "is_host_for",

        "has_metric",

        "enables",

        "improves",

        "affects",

        "depends_on",

        "contains",

        "exhibits",

        "is_a",

        "is_variant_of",

        "integrated_in",

        "has_capability",

        "prevents",

        "related_to"
    ]


    ordered_relations = []


    for relation in priority:

        if relation in groups:

            ordered_relations.append(
                relation
            )


    for relation in groups:

        if relation not in ordered_relations:

            ordered_relations.append(
                relation
            )


    # ---------------------------------------------------------
    # Render groups.
    # ---------------------------------------------------------

    for relation in ordered_relations:

        items = groups[
            relation
        ]


        if relation == "produces":

            answer_parts.append(
                "\n**Production relationships**"
            )


        elif relation == "has_metric":

            answer_parts.append(
                "\n**Reported measurements**"
            )


        elif relation == "increases":

            answer_parts.append(
                "\n**Increased relationships**"
            )


        elif relation == "decreases":

            answer_parts.append(
                "\n**Decreased relationships**"
            )


        else:

            answer_parts.append(
                "\n**"
                +
                relation.replace(
                    "_",
                    " "
                ).title()
                +
                "**"
            )


        seen = set()


        for triple, score in items:

            if triple in seen:

                continue


            seen.add(
                triple
            )


            sentence = triple_to_sentence(
                triple
            )


            sources = searcher.sources.get(
                triple,
                []
            )


            if sources:

                pmids = []


                for source in sources:

                    pmid = str(
                        source.get(
                            "PMID",
                            ""
                        )
                    ).strip()


                    if pmid and pmid not in pmids:

                        pmids.append(
                            pmid
                        )


                if pmids:

                    sentence += (
                        " "
                        "[PMID: "
                        +
                        ", ".join(
                            pmids[:5]
                        )
                        +
                        "]"
                    )


            answer_parts.append(
                "- "
                +
                sentence
            )


    answer_parts.append(
        "\n**Conclusion:** "
        "The statements above are restricted "
        "to relationships stored in the local "
        "knowledge graph."
    )


    return "\n".join(
        answer_parts
    )


# ============================================================================
# 29. LLM ANSWER SYNTHESIS
# ============================================================================

ANSWER_SYSTEM = """
You are KGMiner, a scientific knowledge-graph question answering assistant.

You MUST answer ONLY from the supplied knowledge graph evidence.

Do not use outside knowledge.

Do not invent biological mechanisms.

Do not convert a correlation into causation.

Do not infer a complete pathway if the evidence contains only individual
relationships.

Preserve numerical measurements exactly.

Distinguish:
- production
- increased production
- inhibition
- activation
- encoding
- host relationships
- quantitative metrics

If the evidence does not establish the requested mechanism, explicitly say
that the knowledge graph does not establish it.

Use concise scientific language.

Do not expose raw triple syntax.

Always provide a conclusion.
"""


def llm_answer(
    query,
    search_results,
    graph,
    chat_history=None
):

    evidence_lines = []


    for triple, score in search_results[
        :50
    ]:

        evidence_lines.append(
            "- "
            +
            triple_to_sentence(
                triple
            )
        )


    evidence = "\n".join(
        evidence_lines
    )


    history_text = ""


    if chat_history:

        history_parts = []


        for item in chat_history[
            -4:
        ]:

            history_parts.append(

                "Previous question: "
                +
                str(
                    item.get(
                        "query",
                        ""
                    )
                )
                +
                "\nPrevious answer: "
                +
                str(
                    item.get(
                        "answer",
                        ""
                    )
                )[:1000]
            )


        history_text = (
            "\n\nCONVERSATION CONTEXT:\n"
            +
            "\n\n".join(
                history_parts
            )
        )


    prompt = f"""
QUESTION:

{query}

KNOWLEDGE GRAPH:

{graph.number_of_nodes()} entities
{graph.number_of_edges()} relationships

EVIDENCE:

{evidence}

{history_text}

Answer the question scientifically using ONLY the evidence.
"""


    return llm_call(

        [
            {
                "role":
                    "system",

                "content":
                    ANSWER_SYSTEM
            },

            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ],

        purpose="answer",

        temperature=ANSWER_TEMPERATURE,

        max_tokens=4000
    )


# ============================================================================
# 30. FINAL ANSWER
# ============================================================================

def answer_query(
    query,
    search_results,
    graph,
    searcher,
    chat_history=None
):

    if not search_results:

        return deterministic_answer(
            query,
            search_results,
            searcher
        )


    # ---------------------------------------------------------
    # Always create deterministic evidence first.
    # ---------------------------------------------------------

    deterministic = (
        deterministic_answer(
            query,
            search_results,
            searcher
        )
    )


    # ---------------------------------------------------------
    # LLM answer is optional.
    # ---------------------------------------------------------

    if not ENABLE_LLM_ANSWER:

        return deterministic


    try:

        return llm_answer(
            query,
            search_results,
            graph,
            chat_history
        )


    except Exception as exc:

        logger.warning(
            "LLM answer synthesis failed: %s",
            exc
        )


        # Never return an empty/dead-end response.
        return deterministic


# ============================================================================
# 31. GOAL STORAGE
# ============================================================================

GOALS_DIR = Path(
    os.getenv(
        "GOALS_DIR",
        "goals"
    )
)


GOALS_DIR.mkdir(
    exist_ok=True
)


def goal_dir(
    goal_id
):

    return (
        GOALS_DIR /
        goal_id
    )


def config_path(
    goal_id
):

    return (
        goal_dir(goal_id) /
        "config.json"
    )


def excel_path(
    goal_id
):

    return (
        goal_dir(goal_id) /
        "knowledge_graph.xlsx"
    )


def chat_path(
    goal_id
):

    return (
        goal_dir(goal_id) /
        "query_history.json"
    )


def read_config(
    goal_id
):

    path = config_path(
        goal_id
    )


    if not path.exists():

        raise FileNotFoundError


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as fh:

        return json.load(
            fh
        )


def write_config(
    goal_id,
    config
):

    directory = goal_dir(
        goal_id
    )


    directory.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        config_path(goal_id),
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            config,
            fh,
            indent=2,
            ensure_ascii=False
        )


# ============================================================================
# 32. CHAT HISTORY
# ============================================================================

CHAT_QUERY_LIMIT = 5


def read_chats(
    goal_id
):

    path = chat_path(
        goal_id
    )


    if not path.exists():

        return []


    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as fh:

            return json.load(
                fh
            )


    except Exception:

        return []


# ---------------------------------------------------------------------------

def write_chats(
    goal_id,
    chats
):

    with open(
        chat_path(goal_id),
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            chats,
            fh,
            indent=2,
            ensure_ascii=False
        )


# ============================================================================
# 33. PIPELINE
# ============================================================================

def excel_safe_value(value):
    """Return a value that is safe for Excel/XML 1.0 worksheets."""

    if value is None:
        return None

    if isinstance(value, str):
        text = value

        # Recover common corrupted Greek beta in beta-carotene terms.
        text = re.sub(
            r"(?i)(?:[\x00-\x1F])+-carotene",
            "β-carotene",
            text
        )

        # Remove XML 1.0-illegal C0 controls. Keep TAB/LF/CR.
        text = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
            "",
            text
        )

        # Excel has a maximum cell length of 32,767 characters.
        if len(text) > 32767:
            text = text[:32767]

        return text

    return value


def excel_safe_dataframe(df):
    """Sanitize every cell, not only KG columns, before openpyxl sees it."""

    if df is None:
        return pd.DataFrame()

    out = df.copy()

    for column in out.columns:
        out[column] = out[column].map(excel_safe_value)

    return out


def excel_safe_rows(rows):
    """Sanitize dictionaries used to build Excel sheets."""

    safe_rows = []

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        safe_rows.append({
            key: excel_safe_value(value)
            for key, value in row.items()
        })

    return safe_rows


def save_knowledge_graph_excel(path, df_final, source_rows, metadata):
    """Write the workbook safely using a temporary file and validate it."""

    from openpyxl import load_workbook

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_df = excel_safe_dataframe(df_final)
    safe_source_rows = excel_safe_rows(source_rows)
    safe_metadata = excel_safe_dataframe(metadata)

    temporary = path.with_name(
        path.stem + ".tmp.xlsx"
    )

    try:
        if temporary.exists():
            temporary.unlink()

        with pd.ExcelWriter(
            temporary,
            engine="openpyxl"
        ) as writer:
            safe_df.to_excel(
                writer,
                sheet_name="Knowledge Graph",
                index=False
            )

            pd.DataFrame(safe_source_rows).to_excel(
                writer,
                sheet_name="Triples",
                index=False
            )

            safe_metadata.to_excel(
                writer,
                sheet_name="Metadata",
                index=False
            )

        # Re-open the finished archive before replacing the previous file.
        workbook = load_workbook(temporary, read_only=True)
        workbook.close()

        temporary.replace(path)

    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except Exception:
                pass


def run_pipeline(
    goal_id,
    goal_text
):

    try:

        cfg = read_config(
            goal_id
        )


        def update(
            status=None,
            progress=None,
            **extra
        ):

            if status is not None:

                cfg[
                    "status"
                ] = status


            if progress is not None:

                cfg[
                    "progress"
                ] = progress


            cfg.update(
                extra
            )


            write_config(
                goal_id,
                cfg
            )


        # ================================================================
        # PHASE 1: SEARCH QUERIES
        # ================================================================

        update(

            status="searching",

            progress=5,

            phase_detail=
                "Generating PubMed search queries..."
        )


        queries = generate_search_queries(
            goal_text
        )


        update(

            progress=10,

            search_queries=
                queries,

            phase_detail=
                f"Generated "
                f"{len(queries)} "
                f"search queries."
        )


        # ================================================================
        # PHASE 2: PUBMED
        # ================================================================

        all_pmids = set()


        for index, query in enumerate(
            queries
        ):

            logger.info(
                "[%s] PubMed query: %s",
                goal_id[:8],
                query
            )


            try:

                pmids = search_pubmed(
                    query,
                    retmax=100
                )


                all_pmids.update(
                    pmids
                )


            except Exception as exc:

                logger.warning(
                    "PubMed search failed: %s",
                    exc
                )


            progress = (
                10
                +
                int(
                    15 *
                    (
                        index + 1
                    )
                    /
                    max(
                        len(queries),
                        1
                    )
                )
            )


            update(

                progress=
                    progress,

                pubmed_ids_count=
                    len(all_pmids),

                phase_detail=
                    "PubMed search "
                    f"{index + 1}/"
                    f"{len(queries)}"
            )


        pmids = sorted(
            all_pmids
        )


        # ================================================================
        # PHASE 3: FETCH ARTICLES
        # ================================================================

        update(

            progress=27,

            phase_detail=
                f"Fetching "
                f"{len(pmids)} "
                f"PubMed articles..."
        )


        articles = fetch_pubmed_articles(
            pmids
        )


        df = pd.DataFrame(
            articles
        )


        update(

            progress=35,

            articles_fetched=
                len(df),

            phase_detail=
                f"Fetched "
                f"{len(df)} "
                f"articles with abstracts."
        )


        if df.empty:

            update(

                status="complete",

                progress=100,

                stats={

                    "articles":
                        0,

                    "triples":
                        0,

                    "entities":
                        0
                },

                error_message=
                    "No PubMed articles with abstracts were found."
            )


            pd.DataFrame().to_excel(
                excel_path(
                    goal_id
                ),
                index=False
            )


            return


        # ================================================================
        # PHASE 4: EXTRACTION
        # ================================================================

        update(

            status="extracting",

            progress=40,

            total_count=
                len(df),

            processed_count=0,

            phase_detail=
                "Extracting biological relationships..."
        )


        results = []


        for index, row in df.iterrows():

            article = row.to_dict()


            logger.info(

                "[%s] Processing "
                "%d/%d PMID=%s",

                goal_id[:8],

                index + 1,

                len(df),

                article.get(
                    "PMID"
                )
            )


            try:

                result = process_article(
                    article,
                    goal_text
                )


            except Exception as exc:

                logger.exception(
                    "Article processing failed"
                )


                result = {

                    **article,

                    "Answer to Question 1":
                        "Extraction failed.",

                    "Answer to Question 2":
                        "No relationships found.",

                    "Stability Score Q2":
                        0.0,

                    "Extraction Status":
                        "failed"
                }


            results.append(
                result
            )


            progress = (
                40
                +
                int(
                    40 *
                    (
                        index + 1
                    )
                    /
                    len(df)
                )
            )


            update(

                progress=
                    progress,

                processed_count=
                    index + 1,

                phase_detail=
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(df)} articles"
            )


        df_final = pd.DataFrame(
            results
        )


        # ================================================================
        # PHASE 5: DEDUPLICATION
        # ================================================================

        update(

            status="normalizing",

            progress=82,

            phase_detail=
                "Deduplicating articles..."
        )


        if "PMID" in df_final.columns:

            df_final = (

                df_final

                .drop_duplicates(
                    subset=[
                        "PMID"
                    ]
                )

                .reset_index(
                    drop=True
                )
            )


        # ================================================================
        # PHASE 6: SOURCE-AWARE TRIPLES
        # ================================================================

        update(

            progress=87,

            phase_detail=
                "Building source-aware knowledge graph..."
        )


        source_rows = (
            extract_triples_with_sources(
                df_final
            )
        )


        triples = [

            (
                row["Subject"],
                row["Relation"],
                row["Object"]
            )

            for row in source_rows
        ]


        triples = (
            normalize_triple_set(
                triples
            )
        )


        entities = set()


        for s, r, o in triples:

            entities.add(
                s
            )

            entities.add(
                o
            )


        stats = {

            "articles":
                len(df_final),

            "triples":
                len(triples),

            "entities":
                len(entities),

            "source_triple_rows":
                len(source_rows)
        }


        # ================================================================
        # PHASE 7: SAVE EXCEL
        # ================================================================

        update(

            progress=94,

            phase_detail=
                "Saving knowledge graph Excel file..."
        )


        metadata = pd.DataFrame({

            "Parameter": [
                "Goal",
                "Date",
                "Articles",
                "Unique Triples",
                "Entities",
                "Source Triple Rows",
                "LLM Providers",
                "HeliosX Models",
                "Groq Models"
            ],

            "Value": [
                goal_text,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                stats["articles"],
                stats["triples"],
                stats["entities"],
                stats["source_triple_rows"],
                "HeliosX + Groq",
                ", ".join(HELIOSX_EXTRACTION_MODELS),
                ", ".join(GROQ_EXTRACTION_MODELS)
            ]
        })

        save_knowledge_graph_excel(
            excel_path(goal_id),
            df_final,
            source_rows,
            metadata
        )

        logger.info(
            "Knowledge graph Excel saved: %s | articles=%d triples=%d entities=%d",
            excel_path(goal_id),
            stats["articles"],
            stats["triples"],
            stats["entities"]
        )


        # ================================================================
        # CACHE CLEAR
        # ================================================================

        with _graph_lock:

            _graph_cache.pop(
                goal_id,
                None
            )


        update(

            status="complete",

            progress=100,

            stats=
                stats,

            phase_detail=
                "Knowledge graph generation complete."
        )


        logger.info(

            "[%s] COMPLETE "
            "articles=%d "
            "triples=%d "
            "entities=%d",

            goal_id[:8],

            stats[
                "articles"
            ],

            stats[
                "triples"
            ],

            stats[
                "entities"
            ]
        )


    except Exception as exc:

        logger.exception(
            "[%s] Pipeline failed",
            goal_id[:8]
        )


        try:

            cfg = read_config(
                goal_id
            )


            cfg[
                "status"
            ] = "error"


            cfg[
                "error_message"
            ] = str(
                exc
            )


            write_config(
                goal_id,
                cfg
            )


        except Exception:

            pass


# ============================================================================
# 34. FASTAPI
# ============================================================================

app = FastAPI(

    title=
        "KGMiner Knowledge Graph Query API",

    version=
        "4.0.0"
)


app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


# ============================================================================
# 35. PYDANTIC MODELS
# ============================================================================

class GoalCreate(
    BaseModel
):

    name: str


class QueryRequest(
    BaseModel
):

    query: str

    top_k: int = 50

    chat_id: str | None = None


# ============================================================================
# 36. API STATUS
# ============================================================================

@app.get(
    "/api/status"
)
def api_status():

    return {

        "status":
            "ok",

        "providers":
            provider_status(),

        "message":
            "KGMiner v4 backend is running."
    }


# ============================================================================
# 37. MODEL DISCOVERY
# ============================================================================

@app.get(
    "/api/models"
)
def api_models():

    result = {}


    for provider in [

        heliosx_provider,

        groq_provider

    ]:

        if not provider.available():

            result[
                provider.name
            ] = {

                "configured":
                    False,

                "models":
                    [],

                "message":
                    "No API key configured."
            }


            continue


        try:

            models = (
                provider.discover_models(
                    force=True
                )
            )


            result[
                provider.name
            ] = {

                "configured":
                    True,

                "models":
                    models,

                "configured_fallback_models":
                    provider.models
            }


        except Exception as exc:

            result[
                provider.name
            ] = {

                "configured":
                    True,

                "models":
                    [],

                "error":
                    str(exc)
            }


    return result


# ============================================================================
# 38. TEST LLM
# ============================================================================

@app.get(
    "/api/test-llm"
)
def test_llm():

    prompt = [

        {
            "role":
                "system",

            "content":
                "You are a test assistant. "
                "Return a short response."
        },

        {
            "role":
                "user",

            "content":
                "Say: KGMiner LLM connection successful."
        }
    ]


    try:

        response = llm_call(

            prompt,

            purpose="answer",

            temperature=0.0,

            max_tokens=100
        )


        return {

            "status":
                "success",

            "response":
                response
        }


    except Exception as exc:

        return {

            "status":
                "error",

            "error":
                str(exc)
        }


# ============================================================================
# 39. GOALS
# ============================================================================

@app.get(
    "/api/goals"
)
def list_goals():

    goals = []


    if not GOALS_DIR.exists():

        return goals


    for directory in sorted(
        GOALS_DIR.iterdir()
    ):

        path = (
            directory /
            "config.json"
        )


        if not path.exists():

            continue


        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as fh:

                goals.append(
                    json.load(
                        fh
                    )
                )


        except Exception:

            continue


    return goals


# ---------------------------------------------------------------------------

@app.post(
    "/api/goals",
    status_code=201
)
def create_goal(
    body: GoalCreate
):

    goal_id = str(
        uuid.uuid4()
    )


    config = {

        "id":
            goal_id,

        "name":
            body.name,

        "status":
            "pending",

        "progress":
            0,

        "created_at":
            datetime.utcnow().isoformat(),

        "stats": {

            "articles":
                0,

            "triples":
                0,

            "entities":
                0
        },

        "error_message":
            None
    }


    write_config(
        goal_id,
        config
    )


    thread = threading.Thread(

        target=
            run_pipeline,

        args=(

            goal_id,

            body.name
        ),

        daemon=True
    )


    thread.start()


    return config


# ---------------------------------------------------------------------------

@app.get(
    "/api/goals/{goal_id}"
)
def get_goal(
    goal_id: str
):

    try:

        return read_config(
            goal_id
        )


    except FileNotFoundError:

        raise HTTPException(

            status_code=404,

            detail=
                "Goal not found"
        )


# ============================================================================
# 40. GRAPH DATA
# ============================================================================

@app.get(
    "/api/goals/{goal_id}/data"
)
def get_goal_data(

    goal_id: str,

    page: int = Query(
        1,
        ge=1
    ),

    page_size: int = Query(
        50,
        ge=1,
        le=500
    )
):

    path = excel_path(
        goal_id
    )


    if not path.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "Knowledge graph file not found."
        )


    try:

        df = pd.read_excel(

            path,

            sheet_name=
                "Knowledge Graph"
        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=str(
                exc
            )
        )


    total = len(
        df
    )


    start = (
        page - 1
    ) * page_size


    end = (
        start +
        page_size
    )


    page_df = df.iloc[
        start:end
    ]


    return {

        "total":
            total,

        "page":
            page,

        "page_size":
            page_size,

        "total_pages":
            (
                total +
                page_size -
                1
            )
            //
            page_size,

        "data":
            page_df
            .fillna("")
            .to_dict(
                orient="records"
            )
    }


# ============================================================================
# 41. QUERY KNOWLEDGE GRAPH
# ============================================================================

@app.post(
    "/api/goals/{goal_id}/query"
)
def query_goal(

    goal_id: str,

    body: QueryRequest
):

    path = excel_path(
        goal_id
    )


    if not path.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "Knowledge graph file not found."
        )


    try:

        df = pd.read_excel(

            path,

            sheet_name=
                "Knowledge Graph"
        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=str(
                exc
            )
        )


    if df.empty:

        return {

            "answer":
                "The knowledge graph is empty.",

            "results_count":
                0,

            "top_triples":
                []
        }


    # ---------------------------------------------------------
    # Build graph
    # ---------------------------------------------------------

    graph, searcher = get_graph(

        goal_id,

        df
    )


    # ---------------------------------------------------------
    # Chat history
    # ---------------------------------------------------------

    chats = read_chats(
        goal_id
    )


    chat_id = body.chat_id

    chat = None


    if chat_id:

        for item in chats:

            if (
                item.get("id")
                ==
                chat_id
            ):

                chat = item

                break


    if (
        chat
        and
        len(
            chat.get(
                "messages",
                []
            )
        )
        >=
        CHAT_QUERY_LIMIT
    ):

        return {

            "answer":
                f"This chat has reached "
                f"the limit of "
                f"{CHAT_QUERY_LIMIT} queries.",

            "results_count":
                0,

            "top_triples":
                [],

            "chat_id":
                chat_id,

            "limit_reached":
                True
        }


    history = (

        chat.get(
            "messages",
            []
        )

        if chat

        else []
    )


    # ---------------------------------------------------------
    # LOCAL SEARCH
    # ---------------------------------------------------------

    results = searcher.search(

        body.query,

        top_k=max(

            1,

            min(
                body.top_k,
                200
            )
        )
    )


    # ---------------------------------------------------------
    # ANSWER
    # ---------------------------------------------------------

    answer = answer_query(

        body.query,

        results,

        graph,

        searcher,

        history
    )


    # ---------------------------------------------------------
    # API RESULT
    # ---------------------------------------------------------

    result = {

        "answer":
            answer,

        "results_count":
            len(
                results
            ),

        "top_triples": [

            {

                "triple":
                    triple,

                "score":
                    round(
                        score,
                        4
                    ),

                "sources":
                    searcher.sources.get(
                        triple,
                        []
                    )[:5]
            }

            for triple, score
            in results[:10]
        ]
    }


    # ---------------------------------------------------------
    # SAVE CHAT
    # ---------------------------------------------------------

    message = {

        "query":
            body.query,

        "answer":
            answer,

        "timestamp":
            datetime.now().isoformat()
    }


    if chat:

        chat[
            "messages"
        ].append(
            message
        )


    else:

        chat_id = str(
            uuid.uuid4()
        )[:8]


        chat = {

            "id":
                chat_id,

            "title":
                body.query[:60],

            "created":
                datetime.now().isoformat(),

            "messages":
                [
                    message
                ]
        }


        chats.append(
            chat
        )


    try:

        write_chats(
            goal_id,
            chats
        )


    except Exception as exc:

        logger.warning(
            "Could not save chat: %s",
            exc
        )


    result[
        "chat_id"
    ] = chat["id"]


    result[
        "messages_count"
    ] = len(
        chat[
            "messages"
        ]
    )


    result[
        "limit_reached"
    ] = (

        len(
            chat[
                "messages"
            ]
        )
        >=
        CHAT_QUERY_LIMIT
    )


    return result


# ============================================================================
# 42. CHATS
# ============================================================================

@app.get(
    "/api/goals/{goal_id}/chats"
)
def list_chats(
    goal_id: str
):

    chats = read_chats(
        goal_id
    )


    return [

        {

            "id":
                chat["id"],

            "title":
                chat["title"],

            "created":
                chat["created"],

            "message_count":
                len(
                    chat.get(
                        "messages",
                        []
                    )
                ),

            "limit_reached":
                len(
                    chat.get(
                        "messages",
                        []
                    )
                )
                >=
                CHAT_QUERY_LIMIT
        }

        for chat in chats
    ]


# ---------------------------------------------------------------------------

@app.get(
    "/api/goals/{goal_id}/chats/{chat_id}"
)
def get_chat(

    goal_id: str,

    chat_id: str
):

    chats = read_chats(
        goal_id
    )


    for chat in chats:

        if (
            chat["id"]
            ==
            chat_id
        ):

            return chat


    raise HTTPException(

        status_code=404,

        detail=
            "Chat not found"
    )


# ---------------------------------------------------------------------------

@app.delete(
    "/api/goals/{goal_id}/chats/{chat_id}"
)
def delete_chat(

    goal_id: str,

    chat_id: str
):

    chats = read_chats(
        goal_id
    )


    chats = [

        c

        for c in chats

        if c["id"] != chat_id
    ]


    write_chats(
        goal_id,
        chats
    )


    return {

        "status":
            "deleted"
    }


# ---------------------------------------------------------------------------

@app.delete(
    "/api/goals/{goal_id}/chats"
)
def clear_chats(
    goal_id: str
):

    path = chat_path(
        goal_id
    )


    if path.exists():

        path.unlink()


    return {

        "status":
            "cleared"
    }


# ============================================================================
# 43. GRAPH VISUALIZATION
# ============================================================================

@app.get(
    "/api/goals/{goal_id}/graph"
)
def get_goal_graph(

    goal_id: str,

    limit: int = Query(
        500,
        ge=10,
        le=5000
    )
):

    path = excel_path(
        goal_id
    )


    if not path.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "Knowledge graph file not found."
        )


    df = pd.read_excel(

        path,

        sheet_name=
            "Knowledge Graph"
    )


    triples = (
        extract_triples_from_dataframe(
            df
        )
    )


    node_counts = Counter()


    edges = []


    seen_edges = set()


    for s, r, o in triples:

        node_counts[
            s
        ] += 1


        node_counts[
            o
        ] += 1


        key = (

            s.lower(),

            r.lower(),

            o.lower()
        )


        if key in seen_edges:

            continue


        seen_edges.add(
            key
        )


        edges.append({

            "from":
                s,

            "to":
                o,

            "label":
                r
        })


    sorted_nodes = sorted(

        node_counts.items(),

        key=
            lambda x: x[1],

        reverse=True
    )


    selected_nodes = {

        node

        for node, count
        in sorted_nodes[
            :limit
        ]
    }


    selected_edges = [

        edge

        for edge in edges

        if (

            edge["from"]
            in
            selected_nodes

            or

            edge["to"]
            in
            selected_nodes
        )

    ][:limit]


    node_set = set()


    for edge in selected_edges:

        node_set.add(
            edge["from"]
        )

        node_set.add(
            edge["to"]
        )


    nodes = [

        {

            "id":
                node,

            "label":
                node,

            "count":
                node_counts[node]
        }

        for node in node_set
    ]


    relation_counts = Counter(

        edge["label"]

        for edge in selected_edges
    )


    return {

        "nodes":
            nodes,

        "edges":
            selected_edges,

        "stats": {

            "total_nodes":
                len(
                    node_counts
                ),

            "total_edges":
                len(
                    edges
                ),

            "displayed_nodes":
                len(
                    nodes
                ),

            "displayed_edges":
                len(
                    selected_edges
                ),

            "relation_distribution":
                dict(
                    relation_counts.most_common(
                        15
                    )
                )
        }
    }


# ============================================================================
# 44. DOWNLOAD
# ============================================================================

@app.get(
    "/api/goals/{goal_id}/download"
)
def download_goal(
    goal_id: str
):

    path = excel_path(
        goal_id
    )


    if not path.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "Knowledge graph file not found."
        )


    return FileResponse(

        path=str(
            path
        ),

        media_type=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),

        filename=
            f"knowledge_graph_"
            f"{goal_id[:8]}.xlsx"
    )


# ============================================================================
# 45. DELETE GOAL
# ============================================================================

@app.delete(
    "/api/goals/{goal_id}"
)
def delete_goal(
    goal_id: str
):

    directory = goal_dir(
        goal_id
    )


    if not directory.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "Goal not found"
        )


    with _graph_lock:

        _graph_cache.pop(
            goal_id,
            None
        )


    shutil.rmtree(
        directory
    )


    return {

        "detail":
            "Goal deleted"
    }


# ============================================================================
# 46. FRONTEND
# ============================================================================

@app.get("/")
def serve_index():

    index_path = Path(
        "static/index.html"
    )


    if not index_path.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "static/index.html not found"
        )


    return FileResponse(
        str(
            index_path
        )
    )


if Path(
    "static"
).exists():

    app.mount(

        "/static",

        StaticFiles(
            directory="static"
        ),

        name="static"
    )

