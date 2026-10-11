import os
import pandas as pd
import re
import time
from datetime import datetime, timedelta
# Import Groq specific client and errors
from groq import Groq, APIStatusError as GroqAPIStatusError, RateLimitError as GroqRateLimitError, BadRequestError as GroqBadRequestError
# Import Cerebras specific client and errors
from cerebras.cloud.sdk import Cerebras, APIError, RateLimitError as CerebrasRateLimitError 
from Bio import Entrez
from sentence_transformers import SentenceTransformer, util
import sys
import threading
import queue
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# IMPORTANT: API Keys are now strictly loaded from environment variables for security.
# Ensure you have your .env file or environment variables set.

# Groq API Key (1 key)
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY"),
]

# Cerebras API Keys (5 keys)
CEREBRAS_API_KEYS = [
    os.getenv("CEREBRAS_API_KEY_1"),
    os.getenv("CEREBRAS_API_KEY_2"),
    os.getenv("CEREBRAS_API_KEY_3"),
    os.getenv("CEREBRAS_API_KEY_4"),
    os.getenv("CEREBRAS_API_KEY_5"),
]
# Filter out None values in case some variables are not set
GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k]
CEREBRAS_API_KEYS = [k for k in CEREBRAS_API_KEYS if k]

# Define your email to use with NCBI Entrez
Entrez.email = os.getenv("ENTREZ_EMAIL") # <<<<<  ADD YOUR EMAIL HERE  >>>>>


# --- LLM Model Configurations ---
# Unified structure to hold model details including provider and specific parameters
# Priorities are crucial for fallback order across providers and models.

LLM_MODELS_CONFIG = {
    # Groq Models (using actual model names)
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "params": {"temperature": 1, "max_completion_tokens": 8192, "top_p": 1, "reasoning_effort": "medium"},
        "priority": 1 # Lower number means higher priority
    },
    "llama-3.1-8b-instant": { 
        "provider": "groq",
        "params": {"temperature": 1, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 2
    },
    "llama-3.3-70b-versatile": { 
        "provider": "groq",
        "params": {"temperature": 1, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 3
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": { # This was not in your Groq examples, assuming it's correct from original list
        "provider": "groq",
        "params": {"temperature": 1, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 4
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "provider": "groq",
        "params": {"temperature": 0.7, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 5
    },
    "moonshotai/kimi-k2-instruct-0905": { # This was not in your Groq examples, assuming it's correct from original list
        "provider": "groq",
        "params": {"temperature": 0.7, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 6
    },
    "qwen3-32b": { # Assuming "groq-qwen3-32b" should be just "qwen3-32b" based on patterns
        "provider": "groq",
        "params": {"temperature": 0.7, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 7
    },
    "groq/compound": { # Assuming this name is correct
        "provider": "groq",
        "params": {"temperature": 0.7, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 8
    },
    "groq/compound-mini": { # Assuming this name is correct
        "provider": "groq",
        "params": {"temperature": 0.7, "max_completion_tokens": 1024, "top_p": 1},
        "priority": 9
    },
    # REMOVED: llama-prompt-guard-2-86m and llama-prompt-guard-2-22m
    # These are safety classifiers (output "safe"/"unsafe"), NOT generative models.

    # Cerebras Models (using actual model names)
    "qwen-3-235b-a22b-instruct-2507": { 
        "provider": "cerebras",
        "params": {"temperature": 0.7, "max_completion_tokens": 20000, "top_p": 0.8},
        "priority": 12
    },
    "qwen-3-235b-a22b-thinking-2507": { # This was not in your Cerebras examples, assuming it's correct from original list
        "provider": "cerebras",
        "params": {"temperature": 0.6, "max_completion_tokens": 65536, "top_p": 0.95},
        "priority": 13
    },
    "qwen-3-32b": { # This was not in your Cerebras examples, assuming it's correct from original list
        "provider": "cerebras",
        "params": {"temperature": 0.6, "max_completion_tokens": 16382, "top_p": 0.95},
        "priority": 14
    },
    "gpt-oss-120b": { # This was not in your Cerebras examples, assuming it's correct from original list
        "provider": "cerebras",
        "params": {"temperature": 1, "max_completion_tokens": 65536, "top_p": 1, "reasoning_effort": "medium"},
        "priority": 15
    },
}

# Consolidate models into a single prioritized list based on 'priority' field
PRIORITIZED_LLM_MODELS = sorted(LLM_MODELS_CONFIG.keys(), key=lambda model_name: LLM_MODELS_CONFIG[model_name]["priority"])

# Rate limiting variables for each provider (adjust based on actual provider limits)
MAX_REQUESTS_PER_MINUTE_GROQ = 20 # Example, adjust as per Groq limits
MAX_REQUESTS_PER_MINUTE_CEREBRAS = 60 # Example, adjust as per Cerebras limits
MINUTE_WINDOW = 60  # seconds

MAX_RETRIES_PER_QUESTION = 3 # Number of *overall attempts* for a question across available models
MAX_FAILED_ABSTRACT_QUESTIONS = 1

processed_abstracts_count = 0
global_lock = threading.Lock()
CHECKPOINT_FILE = "checkpoint_results.pkl"  # FIX ISSUE 33: Periodic checkpoint

def chunk_abstract(abstract: str, max_chars: int = 2000) -> list:
    """Simple sentence-based chunking to avoid LLM attention drop on long abstracts."""
    if not abstract or len(abstract) <= max_chars:
        return [abstract]
    
    # FIX ISSUE 15: Improved sentence splitting that handles scientific abbreviations
    # Handles: "et al.", "E. coli", "sp.", "spp.", "vs.", "Fig.", "approx.", single-letter abbrevs
    sentences = re.split(
        r'(?<!\w\.\w.)'      # not mid-abbreviation like "U.S.A."
        r'(?<![A-Z][a-z]\.)'  # not after "Dr.", "Mr.", etc.
        r'(?<!et al\.)'       # not after "et al."
        r'(?<!\bsp\.)'        # not after "sp."
        r'(?<!\bspp\.)'       # not after "spp."
        r'(?<!\bvs\.)'        # not after "vs."
        r'(?<!\bFig\.)'       # not after "Fig."
        r'(?<!\b[A-Z]\.)'     # not after single capital letter like "E." in "E. coli"
        r'(?<=\.|\?)\s',
        abstract
    )
    chunks = []
    current_chunk = ""
    for sent in sentences:
        # FIX ISSUE 16: avoid leading space on first sentence
        if current_chunk:
            candidate = current_chunk + " " + sent
        else:
            candidate = sent
        if len(candidate) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent
        else:
            current_chunk = candidate
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- PubMed Search Functions (unchanged) ---

def search_pubmed(keyword: str) -> list:
    search_term = keyword 
    print(f"Searching PubMed for '{keyword}' to get total count...")
    try:
        handle_count = Entrez.esearch(db="pubmed", term=search_term, retmax=1)
        record_count = Entrez.read(handle_count)
        handle_count.close()
        total_articles = int(record_count["Count"])
        
        if total_articles == 0:
            print(f"No articles found for '{keyword}'.")
            return []

        print(f"Found {total_articles} articles for '{keyword}'. Fetching all IDs...")
        
        # Safety cap to prevent memory explosion on generic queries
        MAX_SAFE_ARTICLES = 100000 
        if total_articles > MAX_SAFE_ARTICLES:
            print(f"[Warning] Total articles ({total_articles}) exceeds safety limit ({MAX_SAFE_ARTICLES}). Fetching first {MAX_SAFE_ARTICLES} only.")
            total_articles = MAX_SAFE_ARTICLES
            
        BATCH_SIZE = 10000
        id_list = []
        for start in range(0, total_articles, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, total_articles - start)  # FIX ISSUE 8: correct last batch size
            print(f"Fetching batch {start} to {start + current_batch_size}...")

            # FIX ISSUE 10: retry failed batches up to 3 times
            for retry in range(3):
                try:
                    handle = Entrez.esearch(db="pubmed", term=search_term, retstart=start, retmax=current_batch_size)
                    record = Entrez.read(handle)
                    handle.close()
                    id_list.extend(record["IdList"])
                    break  # Success, exit retry loop
                except Exception as batch_err:
                    print(f"[Warning] Batch {start} failed (attempt {retry+1}/3): {batch_err}")
                    if retry < 2:
                        time.sleep(2 * (retry + 1))
                    else:
                        print(f"[Error] Batch {start} permanently failed. Continuing with collected IDs.")

            time.sleep(0.5)  # Be nice to NCBI API

        return id_list
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []






def fetch_article_details(id_list: list) -> list:
    """FIX ISSUE 11: Batch efetch into chunks of 200 IDs to avoid HTTP 414 / timeouts.
       FIX ISSUE 12: Skip articles with empty abstracts.
       FIX ISSUE 13: Store PMID for traceability."""
    if not id_list:
        return []

    EFETCH_BATCH_SIZE = 200  # NCBI recommended max per request
    articles = []
    total = len(id_list)
    print(f"Fetching details for {total} articles in batches of {EFETCH_BATCH_SIZE}...")

    for batch_start in range(0, total, EFETCH_BATCH_SIZE):
        batch_ids = id_list[batch_start:batch_start + EFETCH_BATCH_SIZE]
        ids_str = ','.join(batch_ids)
        print(f"  Fetching batch {batch_start}-{batch_start + len(batch_ids)} of {total}...")

        try:
            handle = Entrez.efetch(db="pubmed", id=ids_str, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for pubmed_article in records.get('PubmedArticle', []):
                article = {}
                medline_citation = pubmed_article.get('MedlineCitation', {})
                article_data = medline_citation.get('Article', {})

                # FIX ISSUE 13: Store PMID
                pmid = str(medline_citation.get('PMID', 'N/A'))
                article['PMID'] = pmid

                article['Title'] = article_data.get('ArticleTitle', 'N/A')

                abstract_text_elements = article_data.get('Abstract', {}).get('AbstractText', [])
                if isinstance(abstract_text_elements, list):
                    abstract_text = ' '.join(str(elem) for elem in abstract_text_elements)
                else:
                    abstract_text = str(abstract_text_elements)
                article['Abstract'] = abstract_text

                article['Journal'] = article_data.get('Journal', {}).get('Title', 'N/A')

                # FIX ISSUE 12: Skip empty abstracts
                if not abstract_text or abstract_text.strip() == '' or abstract_text == 'N/A':
                    continue

                articles.append(article)

            time.sleep(0.4)  # Rate limit for NCBI
        except Exception as e:
            print(f"[Warning] Failed to fetch batch starting at {batch_start}: {e}. Continuing...")
            continue

    print(f"Fetched {len(articles)} articles with non-empty abstracts (from {total} IDs).")
    return articles

# --- LLM API Interaction Functions (Rewritten for multiple providers) ---

class LLMClientManager:
    """
    Manages a single LLM client (Groq or Cerebras) with its own rate limiter
    and tracks model availability/cooldowns for its associated provider.
    Each thread will get an instance of this.
    """
    def __init__(self, provider: str, api_key: str, max_requests_per_minute: int, minute_window: int):
        self.provider = provider
        self.api_key = api_key
        self.max_requests_per_minute = max_requests_per_minute
        self.minute_window = minute_window
        self.request_times = [] # For overall client rate limiting
        self.lock = threading.Lock()
        self.model_cooldowns = {} # {model_name: datetime_until_cooldown}

        if self.provider == "groq":
            self.client = Groq(api_key=api_key)
            self.error_types = {
                "RateLimitError": GroqRateLimitError,
                "BadRequestError": GroqBadRequestError, # e.g. token limits, malformed request
                "APIStatusError": GroqAPIStatusError, # e.g. 5xx server errors
                "GenericError": Exception
            }
        elif self.provider == "cerebras":
            self.client = Cerebras(api_key=api_key)
            self.error_types = {
                "RateLimitError": CerebrasRateLimitError, 
                "BadRequestError": APIError, 
                "APIStatusError": APIError, 
                "GenericError": Exception
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def wait_for_client_limit(self):
        """Ensures the overall client doesn't exceed its rate limit.
        FIX ISSUE 24: Calculate wait time inside lock, but sleep OUTSIDE lock."""
        time_to_wait = 0
        with self.lock:
            current_time = datetime.now()
            self.request_times[:] = [req_time for req_time in self.request_times
                                     if current_time - req_time < timedelta(seconds=self.minute_window)]

            if len(self.request_times) >= self.max_requests_per_minute:
                oldest_request_time = self.request_times[0] if self.request_times else current_time
                time_to_wait = self.minute_window - (current_time - oldest_request_time).total_seconds() + 1

        # Sleep OUTSIDE lock so other threads aren't blocked
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        with self.lock:
            self.request_times.append(datetime.now())

    def get_model_cooldown_status(self, model_name: str) -> bool:
        """Check if a model is currently on global cooldown for this client."""
        with self.lock:
            cooldown_until = self.model_cooldowns.get(model_name, datetime.min)
            return datetime.now() < cooldown_until

    def set_model_cooldown(self, model_name: str, cooldown_duration_seconds: int):
        """Set a model on cooldown for this client."""
        with self.lock:
            self.model_cooldowns[model_name] = datetime.now() + timedelta(seconds=cooldown_duration_seconds)

    def generate_completion(self, model_name: str, messages: list, **kwargs):
        """
        Calls the appropriate client's completion method with model-specific parameters.
        **kwargs will contain model-specific parameters like temperature, max_completion_tokens etc.
        """
        if self.provider == "groq":
            # Groq uses chat.completions.create
            # Ensure stream=False for simple completion (kwargs might contain stream=True from user example)
            kwargs.pop("stream", None) # Remove stream if present
            return self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs # Pass all model-specific params
            )
        elif self.provider == "cerebras":
            # Cerebras also uses chat.completions.create
            kwargs.pop("stream", None) # Remove stream if present
            return self.client.chat.completions.create(
                messages=messages, # Cerebras has messages first
                model=model_name,
                **kwargs # Pass all model-specific params
            )
        else:
            raise RuntimeError(f"Unknown provider {self.provider} for completion generation.")


def generate_search_queries(user_goal: str, llm_client_manager, model_priority_list: list) -> list:
    # llm_client_manager can be a single manager OR a list of managers for cross-provider fallback
    import hashlib
    import json
    
    # Check cache first
    query_hash = hashlib.md5(user_goal.encode()).hexdigest()
    cache_file = f"query_cache_{query_hash}.json"
    concept_cache_file = f"concept_cache_{query_hash}.json"
    
    if os.path.exists(cache_file):
        print(f"Loading cached queries from {cache_file}...")
        try:
            with open(cache_file, "r") as f:
                queries = json.load(f)
            print(f"Loaded {len(queries)} cached queries.")
            return queries
        except Exception as e:
            print(f"Error reading cache: {e}. Regenerating...")

    print(f"\n--- Generative Step: creating deterministic queries for '{user_goal}' ---")
    
    concepts = None
    if os.path.exists(concept_cache_file):
        print(f"Loading cached concepts from {concept_cache_file}...")
        try:
            with open(concept_cache_file, "r") as f:
                concepts = json.load(f)
        except Exception as e:
            print(f"Error reading concept cache: {e}")

    if not concepts:
        system_prompt = (
            "You are an expert Scientific Information Specialist. "
            "Your task IS NOT to generate queries. "
            "Your task is to EXTRACT the core concepts from the User's Research Goal into a structured JSON format. "
            "Identify and synonymize the following categories:\n"
            "1. 'compound': The main chemical, drug, or molecule.\n"
            "2. 'organism': The target species or host.\n"
            "3. 'process': The action or biological process.\n"
            "4. 'other': Any other specific constraints.\n\n"
            "Output strictly valid JSON: {\"compound\": [], \"organism\": [], \"process\": [], \"other\": []}. "
        )
        
        # Use temperature=0 for deterministic output
        # Try each client manager if one is a list (cross-provider fallback)
        client_managers = llm_client_manager if isinstance(llm_client_manager, list) else [llm_client_manager]
        response = None
        for mgr in client_managers:
            response = ask_llm_with_fallback(mgr, model_priority_list, "", f"Research Goal: {user_goal}", system_prompt, temperature=0, json_mode=True)
            if response and "Error:" not in response:
                break
        if not response:
            response = ""
        
        try:
            # cleanup json
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
            concepts = json.loads(json_str)
            print(f"Extracted Concepts: {concepts}")
            
            # Save concept cache
            with open(concept_cache_file, "w") as f:
                json.dump(concepts, f)
        except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as e:
            print(f"[Error] Failed to parse concepts ({type(e).__name__}: {e}): {response}")
            # Fallback baseline query even if LLM fails
            return [user_goal]

    queries = []
    
    # helper to build OR group
    def build_or_group(terms, tag="[tiab]"):
        if not terms: return ""
        # Quote terms with spaces, apply tag
        cleaned = []
        for t in terms:
            if '[' in t: target = t # already tagged
            else: target = f'"{t}"{tag}' if ' ' in t else f'{t}{tag}'
            cleaned.append(target)
        return f"({' OR '.join(cleaned)})"

    # 1. BASELINE (Broadest possible safety net - No Tags, relies on ATM)
    # Filter out generic noise words that don't help PubMed search
    noise_words = {"study", "research", "investigation", "analysis", "review", "effect", "role", "impact"}
    base_terms = []
    if concepts.get('compound'): base_terms.append(concepts['compound'][0])
    if concepts.get('organism'): base_terms.append(concepts['organism'][0])
    if concepts.get('process'):
        proc_term = concepts['process'][0]
        if proc_term.lower() not in noise_words:
            base_terms.append(proc_term)

    if base_terms:
        queries.append(" AND ".join(base_terms))
    else:
        # Absolute safety net
        queries.append(user_goal)

    # 2. BROAD (Compound + Organism) - Good for high recall
    comp_grp = build_or_group(concepts.get('compound', []))
    org_grp = build_or_group(concepts.get('organism', []))
    
    if comp_grp and org_grp:
        queries.append(f"{comp_grp} AND {org_grp}")

    # 3. SPECIFIC (Compound + Organism + Process)
    proc_grp = build_or_group(concepts.get('process', []))
    if comp_grp and org_grp and proc_grp:
        queries.append(f"{comp_grp} AND {org_grp} AND {proc_grp}")
        
    # 4. TARGETED (Add 'other' or 'engineering' context if exists)
    other_grp = build_or_group(concepts.get('other', []))
    if comp_grp and org_grp and proc_grp and other_grp:
        queries.append(f"{comp_grp} AND {org_grp} AND {proc_grp} AND {other_grp}")

    # Deduplicate preserving order
    unique_queries = []
    seen = set()
    for q in queries:
        if q not in seen:
            unique_queries.append(q)
            seen.add(q)
            
    print(f"Generated {len(unique_queries)} deterministic queries:")
    for q in unique_queries:
        print(f" - {q}")
    
    # Save to cache
    try:
        with open(cache_file, "w") as f:
            json.dump(unique_queries, f)
        print(f"Saved generated queries to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save query cache: {e}")
        
    return unique_queries


def ask_llm_with_fallback(
    llm_client_manager: LLMClientManager, 
    llm_models_priority_list: list, 
    abstract: str, 
    question: str, 
    system_prompt: str,
    temperature=None,
    json_mode=False,
    multi_pass=False,
    user_goal=""
) -> str:
    """
    Interacts with the LLM API with an optional multi-pass extraction layer.
    """
    if multi_pass:
        all_triples = set()

        # FIX ISSUE 20: Both passes now use the SAME ontology so triples are consistent
        ONTOLOGY_INSTRUCTION = (
            "RELATION ONTOLOGY (use ONLY these relations): "
            "activates, inhibits, produces, is_variant_of, encodes, is_host_for, "
            "increases, decreases, integrated_in, has_capability, is_a, has_metric. "
            "NUMERIC DATA: Separate metrics into (Entity, has_metric, \"Value\"). "
        )

        # Pass 1: Initial Extraction (now uses ontology from the start)
        p1_prompt = (
            f"You are a Knowledge Graph Engineer. Extract ALL biological relationships from this abstract chunk. "
            f"Be exhaustive: Include causal, mechanistic, associative, implied, or weak interactions. "
            f"{ONTOLOGY_INSTRUCTION}"
            f"Output ONLY as raw triples: (subject, relation, object). "
            f"Cover genes, proteins, pathways, diseases, compounds, processes, etc. Goal context: {user_goal}."
        )
        res1 = _single_llm_call(llm_client_manager, llm_models_priority_list, abstract, question, p1_prompt, temperature=0)
        all_triples.update(re.findall(r'\(([^,]+?),\s*([^,]+?),\s*([^\)]+?)\)', res1))

        # Pass 2: Overlooked Scan (same ontology, focused on missed items)
        p2_prompt = (
            f"Re-scan for overlooked interactions and engineering details. Focus on {user_goal}. "
            f"{ONTOLOGY_INSTRUCTION}"
            f"Output ONLY additional triples not already identified."
        )
        res2 = _single_llm_call(llm_client_manager, llm_models_priority_list, abstract, f"Already found: {res1}\n\nScan again: {question}", p2_prompt, temperature=0)
        all_triples.update(re.findall(r'\(([^,]+?),\s*([^,]+?),\s*([^\)]+?)\)', res2))

        # Pass 3: Validation
        found_str = "; ".join([f"({s}, {r}, {o})" for s,r,o in all_triples])
        p3_prompt = f"Given the abstract chunk and these triples: [{found_str}], list ONLY missing biological relationships as triples. {ONTOLOGY_INSTRUCTION}"
        res3 = _single_llm_call(llm_client_manager, llm_models_priority_list, abstract, question, p3_prompt, temperature=0)
        all_triples.update(re.findall(r'\(([^,]+?),\s*([^,]+?),\s*([^\)]+?)\)', res3))
        
        if not all_triples:
            return "No relationships found."
        return "; ".join([f"({s.strip()}, {r.strip()}, {o.strip()})" for s,r,o in sorted(all_triples)])

    # Standard Single Pass
    return _single_llm_call(llm_client_manager, llm_models_priority_list, abstract, question, system_prompt, temperature, json_mode)

def _single_llm_call(llm_client_manager, llm_models_priority_list, abstract, question, system_prompt, temperature=None, json_mode=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\nAbstract: {str(abstract)}"}
    ]
    if json_mode: messages[0]["content"] += " Output must be valid JSON."

    # FIX ISSUE 22: Track permanent failures (BadRequest) separately from transient ones (RateLimit)
    # Transient failures are reset each attempt, permanent ones persist
    permanently_failed_models = set()

    for attempt in range(MAX_RETRIES_PER_QUESTION):
        # Only exclude permanently failed + currently-on-cooldown models
        available_models = [model_name for model_name in llm_models_priority_list
                            if LLM_MODELS_CONFIG.get(model_name, {}).get("provider") == llm_client_manager.provider and
                               model_name not in permanently_failed_models and
                               not llm_client_manager.get_model_cooldown_status(model_name)]
        
        if not available_models:
            with global_lock:
                sys.stderr.write(f"\n[Warning] Thread {threading.get_ident()} - Client {llm_client_manager.provider} (key ending {llm_client_manager.api_key[-4:]}) - No available models for this question on attempt {attempt+1}. All models either failed or on global cooldown for this client.\n")
                sys.stderr.flush()
            if attempt < MAX_RETRIES_PER_QUESTION - 1:
                time.sleep(5 * (attempt + 1)) # Wait before the next overall attempt for the question
            continue # Go to next overall attempt for this question

        for model_name in available_models:
            model_config = LLM_MODELS_CONFIG[model_name]
            
            try:
                llm_client_manager.wait_for_client_limit() # Wait for overall client rate limit

                completion_params = model_config["params"].copy()
                
                # Override temp if needed
                if temperature is not None: completion_params["temperature"] = temperature

                chat_completion = llm_client_manager.generate_completion(
                    model_name=model_name,
                    messages=messages,
                    **completion_params
                )
                
                if chat_completion and chat_completion.choices and chat_completion.choices[0].message:
                    content = chat_completion.choices[0].message.content
                    # Remove <think>...</think> blocks common in reasoning models
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    return content
                else:
                    raise ValueError("Empty or malformed completion response.")
            
            except llm_client_manager.error_types["RateLimitError"] as e:
                with global_lock:
                    sys.stderr.write(f"\n[RateLimitError] Thread {threading.get_ident()} - Client {llm_client_manager.provider} (key ending {llm_client_manager.api_key[-4:]}) - Model {model_name} hit rate limit. Moving to next model.\n")
                    sys.stderr.flush()

                # FIX ISSUE 22: RateLimit is transient — use cooldown, don't permanently blacklist
                # FIX ISSUE 23: Safe Retry-After header extraction
                try:
                    resp = getattr(e, 'response', None)
                    retry_after_header = resp.headers.get("Retry-After") if resp and hasattr(resp, 'headers') else None
                    wait_time = int(retry_after_header) if retry_after_header else 30
                except (AttributeError, ValueError, TypeError):
                    wait_time = 30
                llm_client_manager.set_model_cooldown(model_name, wait_time)
                continue

            except llm_client_manager.error_types["BadRequestError"] as e:
                with global_lock:
                    sys.stderr.write(f"\n[BadRequestError] Thread {threading.get_ident()} - Client {llm_client_manager.provider} (key ending {llm_client_manager.api_key[-4:]}) - Model {model_name}: {e}. Permanently skipping.\n")
                    sys.stderr.flush()
                permanently_failed_models.add(model_name)  # Permanent: token limit, malformed request
                continue

            except llm_client_manager.error_types["APIStatusError"] as e:
                with global_lock:
                    sys.stderr.write(f"\n[APIStatusError] Thread {threading.get_ident()} - Client {llm_client_manager.provider} (key ending {llm_client_manager.api_key[-4:]}) - Model {model_name}: {getattr(e, 'status_code', 'N/A')}. Using cooldown.\n")
                    sys.stderr.flush()
                llm_client_manager.set_model_cooldown(model_name, 15)  # Transient: server error
                continue

            except Exception as e:
                with global_lock:
                    sys.stderr.write(f"\n[Error] Thread {threading.get_ident()} - Client {llm_client_manager.provider} (key ending {llm_client_manager.api_key[-4:]}) - Model {model_name}: {e}\n")
                    sys.stderr.flush()
                permanently_failed_models.add(model_name)
                continue
        
        # If we reached here, all available models for this client and question were tried and failed.
        # Wait before the next *overall* attempt for the question.
        if attempt < MAX_RETRIES_PER_QUESTION - 1:
            with global_lock:
                sys.stderr.write(f"\n[Info] Thread {threading.get_ident()} - All models failed for question on abstract (Client {llm_client_manager.provider}, key ending {llm_client_manager.api_key[-4:]}) in attempt {attempt+1}. Retrying in {5 * (attempt + 1)}s...\n")
                sys.stderr.flush()
            time.sleep(5 * (attempt + 1))

    return "Error: Could not generate response after multiple retries across models."

# FIX ISSUE 21/2: Cross-provider fallback wrapper
def ask_llm_cross_provider(
    all_client_managers: list,
    llm_models_priority_list: list,
    abstract: str, question: str, system_prompt: str,
    temperature=None, json_mode=False, multi_pass=False, user_goal=""
) -> str:
    """Tries the primary client first, then falls back to other providers."""
    for mgr in all_client_managers:
        result = ask_llm_with_fallback(
            mgr, llm_models_priority_list, abstract, question, system_prompt,
            temperature=temperature, json_mode=json_mode, multi_pass=multi_pass, user_goal=user_goal
        )
        if result and "Error:" not in result:
            return result
    return "Error: All providers exhausted."


# --- Entity Normalization Functions (Unchanged) ---

VALID_ONTOLOGY_RELATIONS = {
    "activates", "inhibits", "produces", "is_variant_of", "encodes",
    "is_host_for", "is_produced_by", "increases", "decreases",
    "integrated_in", "has_capability", "is_a", "has_metric",
}

def extract_causal_entities(text_column_values: list, warn_invalid_relations: bool = True) -> list:
    """Extract unique entities (subjects + objects) from triple strings.
    Also validates relations against the ontology and warns about unknowns."""
    entities = []
    invalid_relations = set()
    pattern = r'\(([^,]+?),\s*([^,]+?),\s*([^\)]+?)\)'

    for value in text_column_values:
        if not isinstance(value, str):
            continue
        matches = re.findall(pattern, value)
        for match in matches:
            subject = match[0].strip()
            relation = match[1].strip()
            obj = match[2].strip()

            entities.append(subject)
            entities.append(obj)

            # Validate relation against ontology
            if relation.lower() not in VALID_ONTOLOGY_RELATIONS:
                invalid_relations.add(relation)

    if warn_invalid_relations and invalid_relations:
        print(f"[Warning] Found {len(invalid_relations)} non-ontology relations: {sorted(invalid_relations)[:10]}{'...' if len(invalid_relations) > 10 else ''}")
        print("  These will survive into the final output. Consider adding them to RELATION_MAPPING.")

    return sorted(list(set(entities)))

def find_similar_phrases(entities: list, similarity_threshold: float = 0.85) -> dict:
    """FIX ISSUE 25: Use batch cosine similarity matrix instead of O(n^2) pairwise loop.
       FIX ISSUE 26: Prefer longer (more descriptive) entity as canonical."""
    if not entities:
        print("No entities to process for similarity.")
        return {}

    print(f"Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
    try:
        t2vmodel = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        return {}

    print(f"Generating embeddings for {len(entities)} entities...")
    embeddings = t2vmodel.encode(entities, convert_to_tensor=True, show_progress_bar=False)

    similar_phrases = {}
    assigned = set()

    # Process similarity in batches to avoid OOM on large entity sets
    BATCH_SIZE = 2000  # Max rows per batch — keeps memory under ~64MB per batch
    n = len(entities)
    print(f"Computing similarity (batch mode, {n} entities)...")

    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        # Compare this batch against ALL entities from batch_start onward
        sim_block = util.pytorch_cos_sim(embeddings[batch_start:batch_end], embeddings[batch_start:])

        for local_i in range(batch_end - batch_start):
            global_i = batch_start + local_i
            if global_i in assigned:
                continue
            # Only check j > global_i to avoid duplicate pairs
            for local_j in range(local_i + 1, sim_block.shape[1]):
                global_j = batch_start + local_j
                if global_j in assigned:
                    continue
                if sim_block[local_i][local_j].item() > similarity_threshold:
                    # Prefer longer entity as canonical (more descriptive)
                    if len(entities[global_i]) >= len(entities[global_j]):
                        canonical, duplicate = entities[global_i], entities[global_j]
                        assigned.add(global_j)
                    else:
                        canonical, duplicate = entities[global_j], entities[global_i]
                        assigned.add(global_i)
                    similar_phrases[duplicate] = canonical

        if batch_start + BATCH_SIZE < n:
            sys.stdout.write(f"\r  Similarity progress: {min(batch_end, n)}/{n}")
            sys.stdout.flush()

    if n > BATCH_SIZE:
        sys.stdout.write("\n")

    print(f"Found {len(similar_phrases)} phrases to normalize.")
    return similar_phrases

def apply_entity_normalization(df: pd.DataFrame, column_name: str, similar_phrases: dict, exclusion_keyword: str = None) -> pd.DataFrame:
    """FIX ISSUE 27: Resolve transitive chains (A→B, B→C becomes A→C, B→C).
       FIX ISSUE 28: Removed hardcoded 'Yarrowia' exclusion; now None by default."""
    if not similar_phrases:
        print("No normalization rules to apply.")
        return df

    # Resolve transitive chains
    def resolve_canonical(entity, mapping, visited=None):
        if visited is None:
            visited = set()
        if entity in visited:
            return entity  # Break circular reference
        visited.add(entity)
        if entity in mapping:
            return resolve_canonical(mapping[entity], mapping, visited)
        return entity

    resolved_phrases = {k: resolve_canonical(v, similar_phrases) for k, v in similar_phrases.items()}
    similar_phrases = resolved_phrases

    df_copy = df.copy()
    total_rows = len(df_copy)
    current_row_processed = 0

    # FIX ISSUE C: Sort once outside the loop, not per-row
    sorted_similar_phrases = sorted(similar_phrases.items(), key=lambda item: len(item[0]), reverse=True)

    # FIX ISSUE D: Pre-compile all regex patterns once
    compiled_patterns = []
    for similar, original in sorted_similar_phrases:
        escaped = re.escape(similar)
        pattern = re.compile(r'(?<![a-zA-Z0-9_])' + escaped + r'(?![a-zA-Z0-9_])')
        compiled_patterns.append((pattern, original, similar))

    print(f"Applying normalization to column '{column_name}' ({len(compiled_patterns)} rules)...")
    for index, row in df_copy.iterrows():
        cell_value = str(row[column_name])
        modified_cell_value = cell_value

        for pattern, replacement, similar_text in compiled_patterns:
            if exclusion_keyword and (exclusion_keyword in modified_cell_value or exclusion_keyword in similar_text):
                continue
            modified_cell_value = pattern.sub(replacement, modified_cell_value)

        df_copy.at[index, column_name] = modified_cell_value

        current_row_processed += 1
        if current_row_processed % 50 == 0 or current_row_processed == total_rows:
            percentage = (current_row_processed / total_rows) * 100
            sys.stdout.write(f"\rColumn modification progress: {percentage:.2f}% ({current_row_processed}/{total_rows})")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("Column modification complete.")
    return df_copy

RELATION_MAPPING = {
    # Only map NON-ONTOLOGY synonyms to ontology terms.
    # Ontology terms (activates, inhibits, produces, increases, decreases, etc.) are left as-is.

    # Activations / Enhancements → "activates"
    "induces": "activates", "enhances": "activates", "stimulates": "activates",
    "upregulates": "activates", "promotes": "activates",
    "facilitates": "activates", "positive_regulator": "activates",
    "triggers": "activates", "initiates": "activates",

    # Inhibitions / Reductions → "inhibits"
    "reduces": "inhibits", "suppresses": "inhibits", "downregulates": "inhibits",
    "blocks": "inhibits", "interferes": "inhibits",
    "negative_regulator": "inhibits", "diminishes": "inhibits",
    "represses": "inhibits", "attenuates": "inhibits",

    # Production / Yield → "produces" or "is_host_for"
    "produced": "produces", "synthesized": "produces", "secreted": "produces",
    "generates": "produces", "yields": "produces",
    "used as host for production of": "is_host_for", "produced_in": "is_produced_by",

    # Genetic / Functional
    # REMOVED: "encodes": "encodes" (self-mapping, no-op)
    "is_a_variant_of": "is_variant_of", "integrated_into": "integrated_in",
    "has metabolic capability": "has_capability",

    # FIX ISSUE F: Common scientific relations LLMs frequently output
    "degrades": "activates",  # degradation is a biological activity
    "metabolizes": "activates",
    "catalyzes": "activates",
    "binds": "activates",  # binding typically implies functional interaction
    "regulates": "activates",  # generic regulation → activates (conservative)
    "converts": "produces",  # conversion = production of new compound
    "transports": "activates",
    "modifies": "activates",

    # Taxonomy → "is_a"
    "is a": "is_a",
    "belongs to": "is_a",
    "classified as": "is_a",

    # Titer / Metrics
    "increases titer by": "increases", "enhanced production": "activates",
}

def apply_relation_normalization(df: pd.DataFrame, column_name: str, mapping: dict) -> pd.DataFrame:
    df_copy = df.copy()
    total_rows = len(df_copy)

    # FIX ISSUE H: Sort once outside the loop
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)

    # FIX ISSUE I: Pre-compile all regex patterns
    compiled_rel_patterns = []
    for similar, canonical in sorted_mapping:
        pattern = re.compile(
            r'\(\s*([^,]+?)\s*,\s*' + re.escape(similar) + r'\s*,\s*([^)]+?)\s*\)',
            re.IGNORECASE
        )
        compiled_rel_patterns.append((pattern, canonical))

    print(f"Applying relation normalization to column '{column_name}' ({len(compiled_rel_patterns)} rules)...")
    for row_num, (index, row) in enumerate(df_copy.iterrows()):
        cell = str(row[column_name])

        for pattern, canonical in compiled_rel_patterns:
            cell = pattern.sub(
                lambda m: f"({m.group(1).strip()}, {canonical}, {m.group(2).strip()})",
                cell
            )

        # FIX ISSUE J: Case-insensitive deduplication of triples
        triples_raw = re.findall(r'\(([^,]+?),\s*([^,]+?),\s*([^\)]+?)\)', cell)
        if triples_raw:
            seen = set()
            unique_triples = []
            for s, r, o in triples_raw:
                key = (s.strip().lower(), r.strip().lower(), o.strip().lower())
                if key not in seen:
                    seen.add(key)
                    unique_triples.append((s.strip(), r.strip(), o.strip()))
            cell = "; ".join([f"({s}, {r}, {o})" for s, r, o in sorted(unique_triples)])

        df_copy.at[index, column_name] = cell

        # FIX ISSUE K: Progress indicator
        if (row_num + 1) % 50 == 0 or row_num + 1 == total_rows:
            sys.stdout.write(f"\rRelation normalization: {row_num + 1}/{total_rows}")
            sys.stdout.flush()

    sys.stdout.write("\n")
    print("Relation normalization complete.")
    return df_copy


