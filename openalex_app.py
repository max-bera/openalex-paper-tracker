"""
OpenAlex Paper Tracker — Streamlit App

A web interface for:
  1. Searching academic papers on OpenAlex by keyword (company/instrument name)
  2. Looking up metadata for a list of paper titles uploaded as CSV

Usage:
    pip install streamlit requests pandas
    streamlit run openalex_app.py

Then share the URL (default http://localhost:8501) with colleagues.
"""

import streamlit as st
import requests
import pandas as pd
import time
import re
import io
from datetime import date, timedelta
from difflib import SequenceMatcher

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenAlex Paper Tracker",
    page_icon="📚",
    layout="wide",
)

BASE_URL = "https://api.openalex.org"

ALL_FIELDS = [
    "Materials Science", "Chemistry", "Chemical Engineering",
    "Biochemistry, Genetics and Molecular Biology", "Medicine",
    "Engineering", "Physics and Astronomy",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Agricultural and Biological Sciences",
    "Immunology and Microbiology", "Neuroscience", "Health Professions",
    "Multidisciplinary", "Computer Science", "Mathematics",
    "Environmental Science", "Earth and Planetary Sciences",
    "Social Sciences", "Arts and Humanities", "Psychology",
    "Economics, Econometrics and Finance",
    "Business, Management and Accounting", "Decision Sciences",
    "Energy", "Nursing", "Veterinary", "Dentistry",
]

DEFAULT_FIELDS = [
    "Materials Science", "Chemistry", "Chemical Engineering",
    "Biochemistry, Genetics and Molecular Biology", "Medicine",
    "Engineering", "Physics and Astronomy",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Agricultural and Biological Sciences",
    "Immunology and Microbiology", "Neuroscience", "Health Professions",
    "Multidisciplinary",
]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_api_key():
    """
    Resolve the OpenAlex API key.

    Priority:
      1. Sidebar text input (st.session_state['openalex_api_key'])
      2. Streamlit secrets (OPENALEX_API_KEY in .streamlit/secrets.toml)

    OpenAlex made API keys mandatory on 2026-02-13 and removed the old
    email "polite pool", so a key is now required for every request.
    Free keys: https://openalex.org/settings/api
    """
    key = (st.session_state.get("openalex_api_key") or "").strip()
    if key:
        return key
    try:
        return (st.secrets.get("OPENALEX_API_KEY", "") or "").strip()
    except Exception:
        # No secrets.toml present
        return ""


def openalex_get(endpoint, params=None, polite_email=""):
    # `polite_email` is retained for call-signature compatibility but is no
    # longer used: OpenAlex removed the mailto polite pool on 2026-02-13.
    params = dict(params or {})
    api_key = get_api_key()
    if api_key:
        params["api_key"] = api_key

    headers = {"User-Agent": "PaperTracker/1.0"}
    url = f"{BASE_URL}/{endpoint}"
    last_status = None

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            last_status = resp.status_code

            # Auth / credit problems: surface a clear message, don't retry blindly.
            if resp.status_code in (401, 403, 429):
                if not api_key:
                    raise RuntimeError(
                        "OpenAlex requires an API key since 13 Feb 2026 (the email "
                        "'polite pool' was removed), and none was found. Add "
                        "OPENALEX_API_KEY to your Streamlit secrets, or paste a key in "
                        "the sidebar. Get a free key at "
                        "https://openalex.org/settings/api"
                    )
                detail = ""
                try:
                    detail = resp.json().get("message", "")
                except Exception:
                    detail = resp.text[:200]
                raise RuntimeError(
                    f"OpenAlex returned HTTP {resp.status_code} — rate limit or daily "
                    f"budget exceeded. {detail}".strip()
                )

            resp.raise_for_status()
            return resp.json()
        except RuntimeError:
            # Our own explanatory errors above — propagate immediately.
            raise
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"OpenAlex request failed after 3 attempts "
                    f"(last HTTP status: {last_status}): {e}"
                )


def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def normalize_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name.lower().replace('.', ' ')).strip()


def title_similarity(a: str, b: str) -> float:
    """Normalized similarity between two titles (0–100)."""
    a_clean = re.sub(r'[^\w\s]', '', a.lower()).strip()
    b_clean = re.sub(r'[^\w\s]', '', b.lower()).strip()
    return round(SequenceMatcher(None, a_clean, b_clean).ratio() * 100, 1)


def matches_excluded_author(author_name: str, excluded_authors: list) -> bool:
    if not excluded_authors:
        return False
    norm_author = normalize_name(author_name)
    author_parts = norm_author.split()
    if not author_parts:
        return False
    author_surname = author_parts[-1]
    author_given = author_parts[:-1]

    for excluded in excluded_authors:
        norm_excluded = normalize_name(excluded)
        excl_parts = norm_excluded.split()
        if not excl_parts:
            continue
        excl_surname = excl_parts[-1]
        excl_given = excl_parts[:-1]
        if author_surname != excl_surname:
            continue
        if not excl_given:
            return True
        if len(excl_given) > len(author_given):
            continue
        all_match = True
        for i, excl_part in enumerate(excl_given):
            if i >= len(author_given):
                all_match = False
                break
            if len(excl_part) == 1:
                if not author_given[i].startswith(excl_part):
                    all_match = False
                    break
            else:
                if author_given[i] != excl_part:
                    all_match = False
                    break
        if all_match:
            return True
    return False


def check_referenced_authors(work, search_term, polite_email):
    ref_ids = work.get("referenced_works", [])
    if not ref_ids:
        return {"cited_match_count": 0, "cited_match_authors": ""}
    ref_ids_short = [rid.split("/")[-1] for rid in ref_ids[:100]]
    matching_authors = []
    for i in range(0, len(ref_ids_short), 50):
        chunk = ref_ids_short[i:i + 50]
        id_filter = "|".join(chunk)
        try:
            data = openalex_get("works", {
                "filter": f"openalex:{id_filter}",
                "select": "id,display_name,authorships",
                "per_page": 50,
            }, polite_email)
            for ref_work in data.get("results", []):
                for auth in ref_work.get("authorships", []):
                    name = auth.get("author", {}).get("display_name", "")
                    if search_term.lower() in name.lower():
                        ref_title = ref_work.get("display_name", "")[:50]
                        matching_authors.append(f"{name} (in: {ref_title})")
            time.sleep(0.15)
        except Exception:
            pass
    return {
        "cited_match_count": len(matching_authors),
        "cited_match_authors": "; ".join(matching_authors),
    }


def extract_work_metadata(work: dict) -> dict:
    """Extract a flat metadata dict from an OpenAlex work object."""
    title = work.get("display_name", "") or ""

    authors = []
    author_institutions = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        orcid_url = author.get("orcid") or ""
        # Format as "Name [ORCID]" when available
        if orcid_url:
            orcid_id = orcid_url.replace("https://orcid.org/", "")
            authors.append(f"{name} [{orcid_id}]")
        else:
            authors.append(name)
        for inst in authorship.get("institutions", []):
            inst_name = inst.get("display_name", "")
            if inst_name:
                author_institutions.append(inst_name)

    abstract_idx = work.get("abstract_inverted_index")
    abstract_text = reconstruct_abstract(abstract_idx) if abstract_idx else ""

    topics = work.get("topics", [])
    primary_topic = topics[0] if topics else {}
    topic_name = primary_topic.get("display_name", "")
    subfield = primary_topic.get("subfield", {}).get("display_name", "")
    field = primary_topic.get("field", {}).get("display_name", "")

    primary_loc = work.get("primary_location", {}) or {}
    source = primary_loc.get("source", {}) or {}
    journal = source.get("display_name", "")
    oa = work.get("open_access", {}) or {}

    return {
        "openalex_id": work.get("id", ""),
        "doi": work.get("doi", ""),
        "title": title,
        "abstract": abstract_text[:300] + ("…" if len(abstract_text) > 300 else ""),
        "publication_date": work.get("publication_date", ""),
        "authors": "; ".join(authors),
        "institutions": "; ".join(sorted(set(author_institutions))),
        "journal": journal,
        "field": field,
        "subfield": subfield,
        "topic": topic_name,
        "cited_by_count": work.get("cited_by_count", 0),
        "has_fulltext": work.get("has_fulltext", False),
        "is_oa": oa.get("is_oa", False),
    }


# ── Machine (instrument) detection ─────────────────────────────────────────────

# Ordered longest-first so "Piuma Chiaro" is matched before "Piuma" or "Chiaro"
INSTRUMENT_NAMES = ["Piuma Chiaro", "Piuma", "Chiaro", "Pavone", "Cuore"]
VICINITY_CHARS = 300  # characters each side of the search term to scan


def _scan_text_for_instruments(text_lower: str) -> list:
    """Return list of instrument names found in text (word-boundary matched)."""
    found = []
    for instr in INSTRUMENT_NAMES:
        pattern = r'\b' + re.escape(instr.lower()) + r'\b'
        if re.search(pattern, text_lower):
            if instr in ("Piuma", "Chiaro") and "Piuma Chiaro" in found:
                continue
            found.append(instr)
    return found


def detect_machine_local(title: str, abstract: str, search_term: str) -> str:
    """
    Look for instrument names in title+abstract only (no API calls).

    1. If the search term itself IS an instrument, return it when present
       in title/abstract.
    2. Vicinity window (±300 chars) around the search term.
    3. Full title + abstract.

    Returns semicolon-joined instruments or "".
    """
    text = f"{title}  {abstract}"
    text_lower = text.lower()
    term_lower = search_term.lower()

    # Shortcut: search term is itself an instrument
    for instr in INSTRUMENT_NAMES:
        if term_lower == instr.lower():
            if instr.lower() in title.lower() or instr.lower() in abstract.lower():
                return instr
            return ""

    # Tier 1: Vicinity windows
    windows = []
    start = 0
    while True:
        idx = text_lower.find(term_lower, start)
        if idx == -1:
            break
        win_start = max(0, idx - VICINITY_CHARS)
        win_end = min(len(text), idx + len(term_lower) + VICINITY_CHARS)
        windows.append(text_lower[win_start:win_end])
        start = idx + 1

    if windows:
        found = _scan_text_for_instruments(" ".join(windows))
        if found:
            return "; ".join(found)

    # Tier 2: Full title + abstract
    found = _scan_text_for_instruments(text_lower)
    if found:
        return "; ".join(found)

    return ""


def batch_detect_machines(work_ids: list, polite_email: str) -> dict:
    """
    For a list of OpenAlex work IDs, search each instrument name against
    OpenAlex's fulltext index to see which papers mention which instruments.

    Returns {openalex_id: "Instrument; ..."} for IDs with matches.
    """
    if not work_ids:
        return {}

    # Short IDs for the filter (strip URL prefix)
    short_ids = [wid.split("/")[-1] if "/" in wid else wid for wid in work_ids]

    # {short_id: set of instrument names}
    hits = {}

    for instr in INSTRUMENT_NAMES:
        # Query in chunks of 50 (OpenAlex OR-filter limit)
        for i in range(0, len(short_ids), 50):
            chunk = short_ids[i:i + 50]
            id_filter = "|".join(chunk)
            try:
                data = openalex_get("works", {
                    "search": instr,
                    "filter": f"openalex:{id_filter}",
                    "select": "id",
                    "per_page": 50,
                }, polite_email)
                for result in data.get("results", []):
                    rid = result.get("id", "")
                    short = rid.split("/")[-1] if "/" in rid else rid
                    if short not in hits:
                        hits[short] = set()
                    hits[short].add(instr)
                time.sleep(0.15)
            except Exception:
                pass

    # Build final strings, applying compound-name priority
    result = {}
    for short, instruments in hits.items():
        # Find the full ID
        full_id = next((wid for wid in work_ids
                        if wid.endswith(short)), short)
        # If "Piuma Chiaro" matched, drop individual "Piuma" / "Chiaro"
        if "Piuma Chiaro" in instruments:
            instruments.discard("Piuma")
            instruments.discard("Chiaro")
        # Preserve the canonical order from INSTRUMENT_NAMES
        ordered = [i for i in INSTRUMENT_NAMES if i in instruments]
        result[full_id] = "; ".join(ordered)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORD SEARCH pipeline
# ══════════════════════════════════════════════════════════════════════════════

def discover_field_ids(polite_email):
    data = openalex_get("fields", {"per_page": 50}, polite_email)
    all_fields = data.get("results", [])
    mapping = {}
    for field in all_fields:
        mapping[field["display_name"]] = field["id"].split("/")[-1]
    return mapping


def search_works(search_term, date_from, date_to, field_filter, polite_email,
                 progress_callback=None):
    field_id_values = "|".join(field_filter.values())
    date_part = f"from_publication_date:{date_from},to_publication_date:{date_to}"
    combined_filter = f"{date_part},topics.field.id:{field_id_values}"

    all_results = []
    page = 1
    per_page = 100
    total = None

    while True:
        data = openalex_get("works", {
            "search": search_term,
            "filter": combined_filter,
            "per_page": per_page,
            "page": page,
        }, polite_email)

        meta = data.get("meta", {})
        results = data.get("results", [])
        if total is None:
            total = meta.get("count", 0)

        all_results.extend(results)

        if progress_callback:
            progress_callback(min(len(all_results), total), total)

        if len(all_results) >= total or not results:
            break
        page += 1
        time.sleep(0.15)

    return all_results, total


def parse_results(works, search_term, excluded_authors, excluded_terms,
                  check_references, polite_email, progress_callback=None):
    rows = []
    signal_classes = [
        "TITLE_MATCH", "ABSTRACT_MATCH",
        "TITLE_OR_ABSTRACT_MATCH (also author)",
        "FULLTEXT_ONLY (potential instrument mention)",
    ]

    for i, work in enumerate(works):
        title = work.get("display_name", "") or ""
        term_lower = search_term.lower()

        # Authors
        authors = []
        author_institutions = []
        term_is_author = False
        is_excluded_author = False
        matching_author_names = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            orcid_url = author.get("orcid") or ""
            if orcid_url:
                orcid_id = orcid_url.replace("https://orcid.org/", "")
                authors.append(f"{name} [{orcid_id}]")
            else:
                authors.append(name)
            if term_lower in name.lower():
                term_is_author = True
                matching_author_names.append(name)
                if matches_excluded_author(name, excluded_authors):
                    is_excluded_author = True
            for inst in authorship.get("institutions", []):
                inst_name = inst.get("display_name", "")
                if inst_name:
                    author_institutions.append(inst_name)

        # Abstract
        abstract_idx = work.get("abstract_inverted_index")
        abstract_text = reconstruct_abstract(abstract_idx) if abstract_idx else ""
        term_in_abstract = term_lower in abstract_text.lower()
        term_in_title = term_lower in title.lower()

        found_in = []
        if term_in_title:
            found_in.append("title")
        if term_in_abstract:
            found_in.append("abstract")
        if term_is_author:
            found_in.append("author")
        if not found_in:
            found_in.append("fulltext_only")

        # Excluded terms check
        has_excluded_term = False
        matched_excluded_term = ""
        for ex_term in excluded_terms:
            if ex_term.lower() in title.lower() or ex_term.lower() in abstract_text.lower():
                has_excluded_term = True
                matched_excluded_term = ex_term
                break

        # Reference check
        cited_match_count = 0
        cited_match_authors = ""
        if check_references and "fulltext_only" in found_in:
            ref_info = check_referenced_authors(work, search_term, polite_email)
            cited_match_count = ref_info["cited_match_count"]
            cited_match_authors = ref_info["cited_match_authors"]

        # Classify
        if has_excluded_term:
            classification = f"EXCLUDED_TERM ({matched_excluded_term})"
        elif is_excluded_author:
            classification = "EXCLUDED_AUTHOR"
        elif term_is_author and not term_in_title and not term_in_abstract:
            classification = "AUTHOR_FALSE_POSITIVE"
        elif "fulltext_only" in found_in and cited_match_count > 0:
            classification = "REFERENCE_ONLY"
        elif term_in_title and not term_is_author:
            classification = "TITLE_MATCH"
        elif term_in_abstract and not term_is_author:
            classification = "ABSTRACT_MATCH"
        elif (term_in_title or term_in_abstract) and term_is_author:
            classification = "TITLE_OR_ABSTRACT_MATCH (also author)"
        elif "fulltext_only" in found_in and cited_match_count == 0:
            classification = "FULLTEXT_ONLY (potential instrument mention)"
        else:
            classification = "AMBIGUOUS"

        # Metadata
        topics = work.get("topics", [])
        primary_topic = topics[0] if topics else {}
        topic_name = primary_topic.get("display_name", "")
        subfield = primary_topic.get("subfield", {}).get("display_name", "")
        field = primary_topic.get("field", {}).get("display_name", "")

        primary_loc = work.get("primary_location", {}) or {}
        source = primary_loc.get("source", {}) or {}
        journal = source.get("display_name", "")
        oa = work.get("open_access", {}) or {}

        # Detect instrument name in title + abstract (vicinity → full)
        machine = detect_machine_local(title, abstract_text, search_term)

        rows.append({
            "openalex_id": work.get("id", ""),
            "doi": work.get("doi", ""),
            "title": title,
            "abstract": abstract_text[:300] + ("…" if len(abstract_text) > 300 else ""),
            "publication_date": work.get("publication_date", ""),
            "authors": "; ".join(authors),
            "institutions": "; ".join(sorted(set(author_institutions))),
            "journal": journal,
            "field": field,
            "subfield": subfield,
            "topic": topic_name,
            "machine": machine,
            "cited_by_count": work.get("cited_by_count", 0),
            "has_fulltext": work.get("has_fulltext", False),
            "is_oa": oa.get("is_oa", False),
            "found_in": ", ".join(found_in),
            "matching_authors": "; ".join(matching_author_names),
            "cited_match_count": cited_match_count,
            "cited_match_authors": cited_match_authors,
            "classification": classification,
        })

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(i + 1, len(works))

    df = pd.DataFrame(rows)

    # ── Batch fulltext search for papers with no machine detected yet ─────
    if not df.empty:
        no_machine = df[df["machine"] == ""]
        if not no_machine.empty:
            fulltext_hits = batch_detect_machines(
                no_machine["openalex_id"].tolist(), polite_email,
            )
            if fulltext_hits:
                df["machine"] = df.apply(
                    lambda r: r["machine"] if r["machine"]
                    else fulltext_hits.get(r["openalex_id"], ""),
                    axis=1,
                )
    return df, signal_classes


# ══════════════════════════════════════════════════════════════════════════════
#  TITLE LOOKUP pipeline
# ══════════════════════════════════════════════════════════════════════════════

def lookup_single_title(query_title: str, polite_email: str) -> dict:
    """
    Search OpenAlex for a single title using title.search filter.
    Returns the best match with similarity score, or a NOT_FOUND row.
    """
    try:
        # Sanitize title: remove quotes and characters that break the filter syntax
        safe_title = query_title.replace('"', '').replace('\\', '')
        data = openalex_get("works", {
            "filter": f'title.search:"{safe_title}"',
            "per_page": 5,
        }, polite_email)
    except Exception as e:
        return {
            "query_title": query_title,
            "match_status": f"API_ERROR ({e})",
            "similarity": 0.0,
        }

    results = data.get("results", [])
    if not results:
        return {
            "query_title": query_title,
            "match_status": "NOT_FOUND",
            "similarity": 0.0,
        }

    # Score each candidate and pick the best
    best_work = None
    best_sim = -1.0
    for work in results:
        returned_title = work.get("display_name", "") or ""
        sim = title_similarity(query_title, returned_title)
        if sim > best_sim:
            best_sim = sim
            best_work = work

    meta = extract_work_metadata(best_work)
    meta["query_title"] = query_title
    meta["similarity"] = best_sim
    meta["match_status"] = ""  # will be set after threshold check
    return meta


def lookup_titles(input_df: pd.DataFrame, title_col: str,
                  polite_email: str, sim_threshold: float = 75,
                  progress_callback=None) -> pd.DataFrame:
    """
    Look up each title in input_df on OpenAlex.
    Returns a DataFrame with all original columns plus OpenAlex metadata.
    """
    rows = []
    # Columns from the original CSV (excluding the title column itself,
    # which will appear as query_title)
    orig_cols = [c for c in input_df.columns if c != title_col]

    for i, (_, orig_row) in enumerate(input_df.iterrows()):
        raw_title = orig_row[title_col]
        title = str(raw_title).strip() if pd.notna(raw_title) else ""
        if not title:
            continue

        row = lookup_single_title(title, polite_email)

        # Classify based on threshold
        if row["match_status"] == "NOT_FOUND" or row["match_status"].startswith("API_ERROR"):
            pass
        elif row["similarity"] >= sim_threshold:
            row["match_status"] = "MATCHED"
        else:
            row["match_status"] = "LOW_CONFIDENCE"

        # Carry over all original columns (prefix with "input_" if they
        # collide with an OpenAlex column)
        for col in orig_cols:
            out_col = f"input_{col}" if col in row else col
            row[out_col] = orig_row[col]

        rows.append(row)
        time.sleep(0.15)

        if progress_callback:
            progress_callback(i + 1, len(input_df))

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def show_editable_table(source_df, display_cols, key_prefix):
    """Show a data_editor with an Include checkbox. Returns the edited df."""
    if source_df.empty:
        st.info("No papers in this category.")
        return source_df

    edit_df = source_df.reset_index(drop=True).copy()
    edit_df.insert(0, "Include", True)

    col_config = {
        "Include": st.column_config.CheckboxColumn(
            "Include",
            help="Uncheck to exclude from CSV download",
            default=True,
        ),
    }
    if "doi" in display_cols:
        col_config["doi"] = st.column_config.LinkColumn("DOI")
    if "similarity" in display_cols:
        col_config["similarity"] = st.column_config.ProgressColumn(
            "Similarity %",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        )

    edited = st.data_editor(
        edit_df[["Include"] + display_cols],
        use_container_width=True,
        hide_index=True,
        key=f"{key_prefix}_editor",
        column_config=col_config,
        disabled=display_cols,
    )
    return edited


def get_included_csv(edited_df, full_df):
    """Return CSV bytes for rows where Include is True, with all columns."""
    if edited_df.empty or "Include" not in edited_df.columns:
        return full_df.to_csv(index=False).encode("utf-8"), len(full_df)
    mask = edited_df["Include"].values
    included = full_df.reset_index(drop=True).loc[mask]
    return included.to_csv(index=False).encode("utf-8"), len(included)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("📚 OpenAlex Paper Tracker")

# ── OpenAlex API key (shared by both tabs) ─────────────────────────────────────
# Required since 13 Feb 2026. The key is read from this input first, then from
# app secrets (OPENALEX_API_KEY), via get_api_key().
with st.sidebar:
    st.header("🔑 OpenAlex API")
    st.text_input(
        "API key",
        type="password",
        key="openalex_api_key",
        help="Required since 13 Feb 2026. Get a free key at "
             "openalex.org/settings/api",
    )
    if get_api_key():
        st.caption("✅ A key is set (input or app secrets).")
    else:
        st.caption(
            "⚠️ No key found. Paste one above, or add `OPENALEX_API_KEY` to "
            "`.streamlit/secrets.toml`."
        )

mode_keyword, mode_lookup = st.tabs([
    "🔍 Keyword Search",
    "📄 Title Lookup",
])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1: KEYWORD SEARCH
# ──────────────────────────────────────────────────────────────────────────────

with mode_keyword:
    st.caption(
        "Search for academic papers mentioning a keyword (company name, instrument, "
        "technique) and filter out false positives from author names and reference lists."
    )

    with st.expander("⚙️ Search configuration", expanded=True):
        kw_col1, kw_col2 = st.columns([2, 1])
        with kw_col1:
            search_term = st.text_input(
                "Search term",
                value="Pavone",
                help="Company name, instrument name, or keyword to search for.",
            )
        with kw_col2:
            # OpenAlex removed the email "polite pool" on 2026-02-13; the key
            # is now set in the sidebar (or app secrets). Variable kept as ""
            # so existing call sites that pass `polite_email` still work.
            polite_email = ""
            st.markdown("**Authentication**")
            st.caption("🔑 Set your OpenAlex API key in the sidebar.")

        dc1, dc2 = st.columns(2)
        with dc1:
            date_from = st.date_input(
                "From date", value=date(2025, 1, 1),
                min_value=date(2000, 1, 1), max_value=date.today(),
            )
        with dc2:
            date_to = st.date_input(
                "To date", value=date.today(),
                min_value=date(2000, 1, 1), max_value=date.today(),
            )

        target_fields = st.multiselect(
            "Scientific fields", options=ALL_FIELDS, default=DEFAULT_FIELDS,
            help="Only papers in these fields will be returned.",
        )

        fc1, fc2 = st.columns(2)
        with fc1:
            excluded_terms_text = st.text_area(
                "Excluded terms (one per line)", value="",
                help="Papers containing these terms in title/abstract are filtered out.",
            )
            excluded_terms = [
                t.strip() for t in excluded_terms_text.strip().splitlines() if t.strip()
            ]
        with fc2:
            excluded_authors_text = st.text_area(
                "Excluded authors (one per line)", value="",
                help='e.g. "S. Pavone" or "Francesco Pavone".',
            )
            excluded_authors = [
                a.strip() for a in excluded_authors_text.strip().splitlines() if a.strip()
            ]

        check_refs = st.checkbox(
            "Check reference lists (slower, fewer false positives)", value=True,
        )

    run_search = st.button("🔍 Run keyword search", type="primary")

    # ── Execute keyword search ────────────────────────────────────────────
    if run_search:
        if not search_term:
            st.error("Please enter a search term.")
            st.stop()
        if not target_fields:
            st.error("Please select at least one scientific field.")
            st.stop()
        if date_from > date_to:
            st.error("'From date' must be ≤ 'To date'.")
            st.stop()

        with st.status("Discovering OpenAlex field IDs…", expanded=False):
            all_field_map = discover_field_ids(polite_email)
            field_filter = {
                name: fid for name, fid in all_field_map.items()
                if name in target_fields
            }
            st.write(f"Matched {len(field_filter)} of {len(target_fields)} fields.")

        if not field_filter:
            st.error("No OpenAlex fields matched your selection.")
            st.stop()

        search_progress = st.progress(0, text="Searching OpenAlex…")

        def update_search_progress(current, total):
            pct = current / max(total, 1)
            search_progress.progress(pct, text=f"Fetched {current} of {total} papers…")

        works, total_count = search_works(
            search_term, date_from.isoformat(), date_to.isoformat(),
            field_filter, polite_email,
            progress_callback=update_search_progress,
        )
        search_progress.progress(1.0, text=f"Done — {total_count} papers found.")

        if not works:
            st.warning("No papers found for this query.")
            st.stop()

        classify_status = st.status(
            f"Classifying {len(works)} papers…", expanded=False)

        def update_classify_progress(current, total):
            classify_status.update(label=f"Classifying… {current}/{total}")

        df_kw, signal_classes = parse_results(
            works, search_term, excluded_authors, excluded_terms, check_refs,
            polite_email, progress_callback=update_classify_progress,
        )
        classify_status.update(label="Classification complete ✓", state="complete")

        st.session_state["kw_results_df"] = df_kw
        st.session_state["kw_signal_classes"] = signal_classes

    # ── Display keyword search results ────────────────────────────────────
    if "kw_results_df" in st.session_state:
        df = st.session_state["kw_results_df"]
        signal_classes = st.session_state["kw_signal_classes"]

        signal_df = df[df["classification"].isin(signal_classes)].copy()
        noise_df = df[~df["classification"].isin(signal_classes)].copy()

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total papers", len(df))
        m2.metric("Potential matches", len(signal_df))
        m3.metric("Filtered out", len(noise_df))
        snr = len(signal_df) / max(len(noise_df), 1)
        m4.metric("Signal / noise", f"{snr:.2f}")

        with st.expander("Classification breakdown", expanded=True):
            class_counts = df["classification"].value_counts().reset_index()
            class_counts.columns = ["Classification", "Count"]
            st.bar_chart(class_counts, x="Classification", y="Count")

        kw_display_cols = [
            "title", "authors", "publication_date", "journal", "field",
            "machine", "found_in", "classification", "doi",
        ]

        tab_signal, tab_all, tab_noise = st.tabs([
            f"✅ Potential matches ({len(signal_df)})",
            f"📋 All results ({len(df)})",
            f"🚫 Filtered out ({len(noise_df)})",
        ])

        with tab_signal:
            edited_signal = show_editable_table(signal_df, kw_display_cols, "kw_signal")
        with tab_all:
            edited_all = show_editable_table(df, kw_display_cols, "kw_all")
        with tab_noise:
            extra_cols = kw_display_cols + ["matching_authors", "cited_match_authors"]
            if noise_df.empty:
                st.info("Nothing was filtered out.")
                edited_noise = noise_df
            else:
                edited_noise = show_editable_table(noise_df, extra_cols, "kw_noise")

        st.markdown("---")
        st.subheader("Download results")
        st.caption("Only rows with **Include** checked will be in the CSV.")

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            csv_all, n_all = get_included_csv(edited_all, df)
            st.download_button(
                f"⬇ All results ({n_all} rows)", csv_all,
                file_name="openalex_all_results.csv", mime="text/csv",
            )
        with dl2:
            csv_signal, n_signal = get_included_csv(edited_signal, signal_df)
            st.download_button(
                f"⬇ Potential matches ({n_signal} rows)", csv_signal,
                file_name="openalex_potential_matches.csv", mime="text/csv",
            )
        with dl3:
            csv_noise, n_noise = get_included_csv(edited_noise, noise_df)
            st.download_button(
                f"⬇ Filtered out ({n_noise} rows)", csv_noise,
                file_name="openalex_filtered_out.csv", mime="text/csv",
            )
    else:
        st.info("Configure your search above and click **Run keyword search**.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2: TITLE LOOKUP
# ──────────────────────────────────────────────────────────────────────────────

with mode_lookup:
    st.caption(
        "Upload a CSV with paper titles. The app searches OpenAlex for each title, "
        "retrieves full metadata, and lets you download the enriched results."
    )

    with st.expander("⚙️ Lookup configuration", expanded=True):
        lu_col1, lu_col2 = st.columns(2)
        with lu_col1:
            uploaded_file = st.file_uploader(
                "Upload CSV with titles", type=["csv"],
                help="The CSV must have a column containing paper titles.",
            )
        with lu_col2:
            # OpenAlex removed the email "polite pool" on 2026-02-13; the key
            # is now set in the sidebar (or app secrets). Variable kept as ""
            # so existing call sites that pass `lu_email` still work.
            lu_email = ""
            st.markdown("**Authentication**")
            st.caption("🔑 Set your OpenAlex API key in the sidebar.")

        lc1, lc2 = st.columns(2)
        with lc1:
            title_column = st.text_input(
                "Title column name", value="title",
                help="Column in your CSV that contains paper titles. "
                     "Case-sensitive. If blank, the first column is used.",
            )
        with lc2:
            sim_threshold = st.slider(
                "Minimum similarity for confident match (%)",
                min_value=50, max_value=100, value=75, step=5,
                help="Matches below this are flagged as LOW_CONFIDENCE.",
            )

    # ── Preview uploaded CSV ──────────────────────────────────────────────
    titles_to_lookup = []
    col_name_used = ""
    lookup_input_df = None

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            input_df = None

        if input_df is not None:
            col_name = title_column.strip() if title_column.strip() else None

            if col_name and col_name in input_df.columns:
                col_name_used = col_name
            elif col_name and col_name not in input_df.columns:
                st.warning(
                    f'Column "{col_name}" not found. '
                    f"Available columns: {', '.join(input_df.columns)}"
                )
            else:
                col_name_used = input_df.columns[0]

            if col_name_used:
                # Keep only rows where the title column is non-empty
                lookup_input_df = input_df[input_df[col_name_used].notna()].copy()
                lookup_input_df = lookup_input_df[
                    lookup_input_df[col_name_used].astype(str).str.strip() != ""
                ]
                titles_to_lookup = lookup_input_df[col_name_used].tolist()

            if titles_to_lookup:
                other_cols = [c for c in input_df.columns if c != col_name_used]
                extra_info = (
                    f" + {len(other_cols)} extra column(s): "
                    f"**{', '.join(other_cols)}**"
                    if other_cols else ""
                )
                st.write(
                    f"Found **{len(titles_to_lookup)}** titles in column "
                    f"**{col_name_used}**{extra_info}. Preview:"
                )
                st.dataframe(
                    input_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

    run_lookup = st.button(
        "📄 Look up titles", type="primary",
        disabled=len(titles_to_lookup) == 0,
    )

    # ── Execute title lookup ──────────────────────────────────────────────
    if run_lookup and titles_to_lookup and lookup_input_df is not None:
        progress = st.progress(0, text="Looking up titles on OpenAlex…")

        def update_lu_progress(current, total):
            pct = current / max(total, 1)
            progress.progress(pct, text=f"Looked up {current} of {total} titles…")

        lu_df = lookup_titles(
            lookup_input_df, col_name_used, lu_email,
            sim_threshold=sim_threshold,
            progress_callback=update_lu_progress,
        )
        progress.progress(1.0, text=f"Done — {len(lu_df)} titles processed.")

        st.session_state["lu_results_df"] = lu_df

    # ── Display title lookup results ──────────────────────────────────────
    if "lu_results_df" in st.session_state:
        lu_df = st.session_state["lu_results_df"]

        if lu_df.empty:
            st.warning("No results.")
        else:
            matched = lu_df[lu_df["match_status"] == "MATCHED"].copy()
            low_conf = lu_df[lu_df["match_status"] == "LOW_CONFIDENCE"].copy()
            not_found = lu_df[
                (lu_df["match_status"] == "NOT_FOUND")
                | lu_df["match_status"].str.startswith("API_ERROR")
            ].copy()

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total titles", len(lu_df))
            m2.metric("Matched", len(matched))
            m3.metric("Low confidence", len(low_conf))
            m4.metric("Not found", len(not_found))

            lu_core_cols = [
                "query_title", "match_status", "similarity",
                "title", "authors", "publication_date", "journal",
                "field", "doi",
            ]
            # Detect original CSV columns carried through (anything not from
            # OpenAlex metadata or the lookup pipeline itself)
            oa_cols = {
                "query_title", "match_status", "similarity",
                "openalex_id", "doi", "title", "abstract",
                "publication_date", "authors", "institutions", "journal",
                "field", "subfield", "topic", "cited_by_count",
                "cited_by_count", "has_fulltext", "is_oa",
            }
            orig_carried_cols = [
                c for c in lu_df.columns if c not in oa_cols
            ]

            lu_display_cols = [
                c for c in lu_core_cols + orig_carried_cols
                if c in lu_df.columns
            ]

            tab_m, tab_lc, tab_nf, tab_la = st.tabs([
                f"✅ Matched ({len(matched)})",
                f"⚠️ Low confidence ({len(low_conf)})",
                f"❌ Not found ({len(not_found)})",
                f"📋 All ({len(lu_df)})",
            ])

            with tab_m:
                edited_matched = show_editable_table(matched, lu_display_cols, "lu_matched")
            with tab_lc:
                edited_lowconf = show_editable_table(low_conf, lu_display_cols, "lu_lowconf")
            with tab_nf:
                nf_cols = [
                    c for c in ["query_title", "match_status"] + orig_carried_cols
                    if c in lu_df.columns
                ]
                if not_found.empty:
                    st.info("All titles were found.")
                    edited_nf = not_found
                else:
                    edited_nf = show_editable_table(not_found, nf_cols, "lu_nf")
            with tab_la:
                edited_lu_all = show_editable_table(lu_df, lu_display_cols, "lu_all")

            st.markdown("---")
            st.subheader("Download results")
            st.caption("Only rows with **Include** checked will be in the CSV.")

            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                csv_la, n = get_included_csv(edited_lu_all, lu_df)
                st.download_button(
                    f"⬇ All looked-up ({n} rows)", csv_la,
                    file_name="openalex_title_lookup_all.csv", mime="text/csv",
                )
            with dl2:
                csv_m, n = get_included_csv(edited_matched, matched)
                st.download_button(
                    f"⬇ Matched only ({n} rows)", csv_m,
                    file_name="openalex_title_lookup_matched.csv", mime="text/csv",
                )
            with dl3:
                csv_lc, n = get_included_csv(edited_lowconf, low_conf)
                st.download_button(
                    f"⬇ Low confidence ({n} rows)", csv_lc,
                    file_name="openalex_title_lookup_low_confidence.csv", mime="text/csv",
                )

    elif uploaded_file is None:
        st.info("Upload a CSV with paper titles to get started.")
