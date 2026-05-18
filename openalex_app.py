"""
OpenAlex Paper Tracker — Streamlit App

A web interface for searching academic papers on OpenAlex by keyword,
filtering by scientific fields, and classifying results to separate
genuine mentions from false positives (author names, reference lists).

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

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenAlex Paper Tracker",
    page_icon="📚",
    layout="wide",
)

BASE_URL = "https://api.openalex.org"

ALL_FIELDS = [
    "Materials Science",
    "Chemistry",
    "Chemical Engineering",
    "Biochemistry, Genetics and Molecular Biology",
    "Medicine",
    "Engineering",
    "Physics and Astronomy",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Agricultural and Biological Sciences",
    "Immunology and Microbiology",
    "Neuroscience",
    "Health Professions",
    "Multidisciplinary",
    "Computer Science",
    "Mathematics",
    "Environmental Science",
    "Earth and Planetary Sciences",
    "Social Sciences",
    "Arts and Humanities",
    "Psychology",
    "Economics, Econometrics and Finance",
    "Business, Management and Accounting",
    "Decision Sciences",
    "Energy",
    "Nursing",
    "Veterinary",
    "Dentistry",
]

DEFAULT_FIELDS = [
    "Materials Science",
    "Chemistry",
    "Chemical Engineering",
    "Biochemistry, Genetics and Molecular Biology",
    "Medicine",
    "Engineering",
    "Physics and Astronomy",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Agricultural and Biological Sciences",
    "Immunology and Microbiology",
    "Neuroscience",
    "Health Professions",
    "Multidisciplinary",
]


# ── Helpers (from original script) ─────────────────────────────────────────────

def openalex_get(endpoint, params=None, polite_email=""):
    headers = {"User-Agent": f"PaperTracker/1.0 (mailto:{polite_email})"}
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"OpenAlex request failed after 3 attempts: {e}")


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


# ── Core pipeline ──────────────────────────────────────────────────────────────

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
            "select": "id,doi,display_name,publication_date,authorships,"
                      "primary_location,topics,cited_by_count,has_fulltext,"
                      "open_access,abstract_inverted_index,referenced_works",
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
    return df, signal_classes


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("📚 OpenAlex Paper Tracker")
st.caption(
    "Search for academic papers mentioning a keyword (company name, instrument, "
    "technique) and filter out false positives from author names and reference lists."
)

# ── Sidebar: all configuration ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Search parameters")

    search_term = st.text_input(
        "Search term",
        value="Pavone",
        help="Company name, instrument name, or keyword to search for.",
    )

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input(
            "From date",
            value=date(2025, 1, 1),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )
    with col2:
        date_to = st.date_input(
            "To date",
            value=date.today(),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )

    polite_email = st.text_input(
        "Email (for OpenAlex polite pool)",
        value="",
        help="Optional. Providing an email gives you faster rate limits.",
    )

    st.markdown("---")
    st.header("Filters")

    target_fields = st.multiselect(
        "Scientific fields",
        options=ALL_FIELDS,
        default=DEFAULT_FIELDS,
        help="Only papers in these fields will be returned.",
    )

    excluded_terms_text = st.text_area(
        "Excluded terms (one per line)",
        value="",
        help=(
            "Papers containing any of these terms in their title or abstract "
            "will be classified as EXCLUDED_TERM. Case-insensitive."
        ),
    )
    excluded_terms = [
        t.strip() for t in excluded_terms_text.strip().splitlines() if t.strip()
    ]

    excluded_authors_text = st.text_area(
        "Excluded authors (one per line)",
        value="",
        help=(
            'Authors whose surname matches the search term but are not relevant. '
            'Use initials or full names, e.g. "S. Pavone" or "Francesco Pavone".'
        ),
    )
    excluded_authors = [
        a.strip() for a in excluded_authors_text.strip().splitlines() if a.strip()
    ]

    check_refs = st.checkbox(
        "Check reference lists",
        value=True,
        help=(
            "For fulltext-only hits, check if the keyword appears only in cited "
            "references. Slower but reduces false positives."
        ),
    )

    st.markdown("---")
    run_search = st.button("🔍 Run search", type="primary", use_container_width=True)


# ── Main area: results ─────────────────────────────────────────────────────────

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

    # Step 1: Discover fields
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

    # Step 2: Search
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

    # Step 3: Parse & classify
    classify_status = st.status(
        f"Classifying {len(works)} papers…", expanded=False,
    )

    def update_classify_progress(current, total):
        classify_status.update(
            label=f"Classifying… {current}/{total}",
        )

    df, signal_classes = parse_results(
        works, search_term, excluded_authors, excluded_terms, check_refs,
        polite_email,
        progress_callback=update_classify_progress,
    )
    classify_status.update(label="Classification complete ✓", state="complete")

    # Store in session state so it survives re-renders
    st.session_state["results_df"] = df
    st.session_state["signal_classes"] = signal_classes


# ── Display results (persisted in session state) ──────────────────────────────

if "results_df" in st.session_state:
    df = st.session_state["results_df"]
    signal_classes = st.session_state["signal_classes"]

    signal_df = df[df["classification"].isin(signal_classes)].copy()
    noise_df = df[~df["classification"].isin(signal_classes)].copy()

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total papers", len(df))
    m2.metric("Potential matches", len(signal_df))
    m3.metric("Filtered out", len(noise_df))
    snr = len(signal_df) / max(len(noise_df), 1)
    m4.metric("Signal / noise", f"{snr:.2f}")

    # ── Classification breakdown ──────────────────────────────────────────
    with st.expander("Classification breakdown", expanded=True):
        class_counts = df["classification"].value_counts().reset_index()
        class_counts.columns = ["Classification", "Count"]
        st.bar_chart(class_counts, x="Classification", y="Count")

    # ── Helper: editable table with "Include" checkbox ────────────────────
    display_cols = [
        "title", "authors", "publication_date", "journal", "field",
        "found_in", "classification", "doi",
    ]

    def show_editable_table(source_df, key_prefix):
        """Show a data_editor with an Include checkbox. Returns the edited df."""
        if source_df.empty:
            st.info("No papers in this category.")
            return source_df

        edit_df = source_df.reset_index(drop=True).copy()
        edit_df.insert(0, "Include", True)

        edited = st.data_editor(
            edit_df[["Include"] + display_cols],
            use_container_width=True,
            hide_index=True,
            key=f"{key_prefix}_editor",
            column_config={
                "Include": st.column_config.CheckboxColumn(
                    "Include",
                    help="Uncheck to exclude from CSV download",
                    default=True,
                ),
                "doi": st.column_config.LinkColumn("DOI"),
            },
            disabled=display_cols,  # only Include is editable
        )
        return edited

    # ── Tabs: Signal / All / Noise ────────────────────────────────────────
    tab_signal, tab_all, tab_noise = st.tabs([
        f"✅ Potential matches ({len(signal_df)})",
        f"📋 All results ({len(df)})",
        f"🚫 Filtered out ({len(noise_df)})",
    ])

    with tab_signal:
        edited_signal = show_editable_table(signal_df, "signal")

    with tab_all:
        edited_all = show_editable_table(df, "all")

    with tab_noise:
        extra_cols = display_cols + ["matching_authors", "cited_match_authors"]
        if noise_df.empty:
            st.info("Nothing was filtered out.")
            edited_noise = noise_df
        else:
            edit_noise = noise_df.reset_index(drop=True).copy()
            edit_noise.insert(0, "Include", True)
            edited_noise = st.data_editor(
                edit_noise[["Include"] + extra_cols],
                use_container_width=True,
                hide_index=True,
                key="noise_editor",
                column_config={
                    "Include": st.column_config.CheckboxColumn(
                        "Include",
                        help="Uncheck to exclude from CSV download",
                        default=True,
                    ),
                    "doi": st.column_config.LinkColumn("DOI"),
                },
                disabled=extra_cols,
            )

    # ── Downloads (only checked rows) ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Download results")
    st.caption("Only rows with **Include** checked will be in the CSV.")

    def get_included_csv(edited_df, full_df):
        """Return CSV bytes for rows where Include is True, with all columns."""
        if edited_df.empty or "Include" not in edited_df.columns:
            return full_df.to_csv(index=False).encode("utf-8"), len(full_df)
        mask = edited_df["Include"].values
        included = full_df.reset_index(drop=True).loc[mask]
        return included.to_csv(index=False).encode("utf-8"), len(included)

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        csv_all, n_all = get_included_csv(edited_all, df)
        st.download_button(
            f"⬇ All results ({n_all} rows)",
            csv_all,
            file_name="openalex_all_results.csv",
            mime="text/csv",
        )
    with dl2:
        csv_signal, n_signal = get_included_csv(edited_signal, signal_df)
        st.download_button(
            f"⬇ Potential matches ({n_signal} rows)",
            csv_signal,
            file_name="openalex_potential_matches.csv",
            mime="text/csv",
        )
    with dl3:
        csv_noise, n_noise = get_included_csv(edited_noise, noise_df)
        st.download_button(
            f"⬇ Filtered out ({n_noise} rows)",
            csv_noise,
            file_name="openalex_filtered_out.csv",
            mime="text/csv",
        )

else:
    # Landing state
    st.info("👈 Configure your search in the sidebar and click **Run search**.")