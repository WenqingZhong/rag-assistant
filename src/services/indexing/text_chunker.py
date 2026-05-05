import json
import logging
import re
from typing import Dict, List, Optional, Union

from src.schemas.indexing.chunks import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits a paper's text into overlapping chunks ready for embedding.

    WHY chunk at all?
    A 10,000-word paper on "neural networks" might have one truly relevant
    300-word passage. If you embed the whole paper, that signal drowns in
    9,700 irrelevant words. Chunking lets the search return the *specific
    passage* that matched, not the whole document — much more useful to the
    user and to the LLM that will read the retrieved context.

    STRATEGY — section-based first, word-based fallback:

        Section 100–800 words  →  single chunk (right size, keep it whole)
        Section < 100 words    →  combine with adjacent small sections
        Section > 800 words    →  split with sliding-window word chunking
        No sections available  →  sliding-window word chunking on full_text

    Every chunk (regardless of strategy) gets the paper title + abstract
    prepended. WHY? A chunk is retrieved in isolation — without its
    surroundings it loses context. Prepending the abstract gives the
    embedding model enough global context to produce a good vector, and
    gives the LLM enough context to answer questions without hallucinating.

    OVERLAP:
    Consecutive word-based chunks share overlap_size words. This prevents
    a key sentence that falls at a chunk boundary from appearing in neither
    chunk with sufficient context. Rule enforced in __init__: overlap < chunk_size.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap_size: int = 100,
        min_chunk_size: int = 100,
    ):
        """
        Args:
            chunk_size:     Target word count per chunk. 600 words ≈ one dense
                            paragraph — large enough for semantic coherence, small
                            enough to stay within embedding token limits and return
                            focused search results.
            overlap_size:   Words shared between consecutive chunks. 100 words ≈
                            one short paragraph. Ensures sentences at chunk
                            boundaries appear in full context in at least one chunk.
                            Must be strictly less than chunk_size.
            min_chunk_size: Chunks with fewer words than this are either merged
                            with a neighbour or promoted to a single chunk rather
                            than discarded. Below ~100 words an embedding carries
                            too little signal to be useful.
        """
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        logger.info(
            f"TextChunker initialised: chunk_size={chunk_size}, "
            f"overlap_size={overlap_size}, min_chunk_size={min_chunk_size}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: Optional[Union[Dict, str, list]] = None,
    ) -> List[TextChunk]:
        """
        Entry point. Chunk a paper using the hybrid section-based strategy.

        Decision flow:
            1. If sections are provided → try _chunk_by_sections()
            2. If that succeeds and produces chunks → return them
            3. Otherwise (no sections, parse failure, or empty result) →
               fall back to chunk_text() on the raw full_text

        WHY prefer sections?
        PDF parsers (Docling, pypdfium2) extract logical section boundaries —
        "Introduction", "Methods", "Results". These are *meaningful* semantic
        units written by the authors. Splitting at those boundaries produces
        cleaner chunks than blindly cutting every 600 words, which might split
        mid-argument or mid-table.

        Args:
            title:     Prepended to every chunk as context for the embedding.
            abstract:  Prepended to every chunk as context for the embedding.
            full_text: Raw extracted text — used only when sections are absent.
            arxiv_id:  e.g. "2301.07041". Stored on every chunk so search results
                       can cite their source without a DB round-trip.
            paper_id:  PostgreSQL row ID. Stored on every chunk so you can JOIN
                       back to the full paper record if needed.
            sections:  Section data in any of three formats:
                         - dict:  {"Introduction": "text...", "Methods": "text..."}
                         - list:  [{"title": "Intro", "content": "..."}, ...]
                         - str:   JSON-encoded version of either of the above
                       _parse_sections() normalises all three to a plain dict.
        """
        if sections:
            try:
                section_chunks = self._chunk_by_sections(
                    title, abstract, arxiv_id, paper_id, sections
                )
                if section_chunks:
                    logger.info(
                        f"Section-based chunking: {len(section_chunks)} chunks for {arxiv_id}"
                    )
                    return section_chunks
            except Exception as e:
                logger.warning(f"Section-based chunking failed for {arxiv_id}: {e}")

        logger.info(f"Falling back to word-based chunking for {arxiv_id}")
        return self.chunk_text(full_text, arxiv_id, paper_id)

    def chunk_text(self, text: str, arxiv_id: str, paper_id: str) -> List[TextChunk]:
        """
        Split arbitrary text into overlapping sliding-window chunks.

        This is both the public fallback (called by chunk_paper when no sections
        are available) and an internal helper (called by _split_large_section
        when a single section exceeds 800 words).

        Algorithm — sliding window with overlap:

            words = ["w0", "w1", ..., "wN"]

            chunk 0: words[0 : chunk_size]           e.g. words[0:600]
            chunk 1: words[chunk_size-overlap : ...]  e.g. words[500:1100]
            chunk 2: words[2*(chunk_size-overlap):]   e.g. words[1000:1600]
            ...

            Each step advances by (chunk_size - overlap_size) = 500 words.
            The trailing 100 words of chunk k become the leading 100 words
            of chunk k+1, so a sentence split across the boundary appears in
            full in at least one of the two chunks.

        Edge cases handled:
            - Empty text → return []
            - Text shorter than min_chunk_size → return as a single chunk
              rather than discarding (better to have one small chunk than none)
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for paper {arxiv_id} — no chunks created")
            return []

        words = self._split_into_words(text)

        # Text too short even for a single minimum-size chunk.
        # Return it as-is rather than discarding — a short abstract is still
        # searchable even if it doesn't meet the target chunk size.
        if len(words) < self.min_chunk_size:
            logger.warning(
                f"Paper {arxiv_id} has only {len(words)} words "
                f"(min {self.min_chunk_size}) — returning as single chunk"
            )
            if words:
                return [self._make_chunk(words, 0, 0, len(text), arxiv_id, paper_id)]
            return []

        chunks: List[TextChunk] = []
        chunk_index = 0
        pos = 0  # current start position in the words list

        while pos < len(words):
            end = min(pos + self.chunk_size, len(words))
            chunk_words = words[pos:end]

            # Character offsets are approximate (word join loses original spacing)
            # but are good enough for debugging and highlighting purposes.
            start_char = len(" ".join(words[:pos])) if pos > 0 else 0
            end_char = len(" ".join(words[:end]))

            # overlap_with_previous: how many words this chunk shares with the
            # preceding chunk. Zero for the very first chunk.
            overlap_prev = min(self.overlap_size, pos) if pos > 0 else 0
            # overlap_with_next: how many words the next chunk will re-read from
            # this one. Zero for the last chunk.
            overlap_next = self.overlap_size if end < len(words) else 0

            chunks.append(
                TextChunk(
                    text=self._reconstruct_text(chunk_words),
                    metadata=ChunkMetadata(
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        word_count=len(chunk_words),
                        overlap_with_previous=overlap_prev,
                        overlap_with_next=overlap_next,
                        section_title=None,  # word-based chunks have no section title
                    ),
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                )
            )

            # Advance by stride = chunk_size - overlap_size.
            # With chunk_size=600 and overlap_size=100, stride=500:
            # chunk 0 covers words 0–599, chunk 1 covers words 500–1099, etc.
            pos += self.chunk_size - self.overlap_size
            chunk_index += 1

            if end >= len(words):
                break

        logger.info(
            f"Word-based chunking: {len(words)} words → {len(chunks)} chunks for {arxiv_id}"
        )
        return chunks

    # ── Section-based chunking ─────────────────────────────────────────────

    def _chunk_by_sections(
        self,
        title: str,
        abstract: str,
        arxiv_id: str,
        paper_id: str,
        sections: Union[Dict, str, list],
    ) -> List[TextChunk]:
        """
        Apply the hybrid section strategy to produce chunks with semantic boundaries.

        Steps:
            1. _parse_sections()  — normalise input to {title: content} dict
            2. _filter_sections() — remove noise (metadata, abstract duplicates)
            3. Build header = title + abstract (prepended to every chunk)
            4. Walk sections in order, routing each to the right strategy:
                 < 100 words  → buffer in small_buffer
                 100–800      → single chunk
                 > 800        → split via _split_large_section()
            5. Flush small_buffer when we hit a large section or reach the end

        WHY process sections in order and buffer small ones?
        Sections often cluster: "Limitations" (60 words) followed by
        "Future Work" (70 words) followed by "Conclusion" (80 words).
        Buffering groups these into one coherent chunk rather than three
        near-useless embeddings. The flush triggers when the next section
        is large (and therefore self-contained) or when we reach the end.

        Example — a typical paper with mixed section sizes:

            sections = {
                "Introduction":   "...1200 words..."   → > 800, split into 3 sub-chunks
                "Related Work":   "...500 words..."    → 100-800, one chunk
                "Methods":        "...700 words..."    → 100-800, one chunk
                "Results":        "...600 words..."    → 100-800, one chunk
                "Limitations":    "...60 words..."     → < 100, buffered
                "Future Work":    "...70 words..."     → < 100, buffered (is_last=False, next_is_large=False)
                "Conclusion":     "...80 words..."     → < 100, buffered, is_last=True → flush!
            }

            Result: 3 + 1 + 1 + 1 + 1 = 7 chunks
                    (Introduction split into 3, each remaining section is 1 chunk,
                     Limitations+Future Work+Conclusion combined into 1 chunk)
        """
        sections_dict = self._parse_sections(sections)
        if not sections_dict:
            return []

        sections_dict = self._filter_sections(sections_dict, abstract)
        if not sections_dict:
            logger.warning(f"No meaningful sections after filtering for {arxiv_id}")
            return []

        # Prepended to every chunk so the embedding captures the paper's topic,
        # not just the isolated section content.
        header = f"{title}\n\nAbstract: {abstract}\n\n"

        chunks: List[TextChunk] = []
        small_buffer: List[tuple] = []  # (section_title, content, word_count)

        items = list(sections_dict.items())

        for i, (sec_title, sec_content) in enumerate(items):
            content = str(sec_content) if sec_content else ""
            word_count = len(content.split())

            is_last = (i == len(items) - 1)
            # Peek ahead: is the NEXT section large enough to stand alone?
            # If yes, flush the buffer now — the next section won't be joined.
            next_is_large = (
                not is_last
                and len(str(items[i + 1][1]).split()) >= 100
            )

            if word_count < 100:
                # Too small to stand alone — buffer it.
                # Will be combined when we hit a large section or the end.
                small_buffer.append((sec_title, content, word_count))
                if is_last or next_is_large:
                    chunks.extend(
                        self._flush_small_buffer(header, small_buffer, chunks, arxiv_id, paper_id)
                    )
                    small_buffer = []

            elif word_count <= 800:
                # Ideal size — one section becomes exactly one chunk.
                # The header (title + abstract) is prepended for context.
                chunk_text = f"{header}Section: {sec_title}\n\n{content}"
                chunks.append(
                    self._make_section_chunk(chunk_text, sec_title, len(chunks), arxiv_id, paper_id)
                )

            else:
                # Section is too long for a single chunk.
                # Delegate to _split_large_section(), which runs word-based
                # chunking on the section content and re-attaches the header
                # to each resulting sub-chunk.
                section_text = f"Section: {sec_title}\n\n{content}"
                sub_chunks = self._split_large_section(
                    header, section_text, sec_title, len(chunks), arxiv_id, paper_id
                )
                chunks.extend(sub_chunks)

        return chunks

    def _flush_small_buffer(
        self,
        header: str,
        small_sections: List[tuple],
        existing_chunks: List[TextChunk],
        arxiv_id: str,
        paper_id: str,
    ) -> List[TextChunk]:
        """
        Combine all buffered small sections into one chunk, or absorb them
        into the previously created chunk if the combined result is still tiny.

        WHY two levels of merging?
        Level 1 — combine small sections together:
            "Limitations" (60 words) + "Future Work" (70 words) = 130-word chunk.
            130 words is still small but meaningful enough to embed.

        Level 2 — absorb into previous chunk if combined total < 200 words:
            "Acknowledgements" (30 words) alone is too small even combined.
            Appending it to the preceding "Conclusion" chunk is harmless and
            avoids creating a nearly empty embedding.

        The 200-word threshold for level 2 is intentionally conservative —
        we only absorb if both the header AND the combined sections are tiny.

        Mutates existing_chunks[-1] in place for level-2 absorption.
        Returns [] in that case (caller should not append anything extra).

        Example — Level 1 (combine small sections into a new chunk):

            small_buffer = [
                ("Limitations", "This approach requires labelled data ...", 60),
                ("Future Work",  "We plan to extend this to ...",            70),
                ("Conclusion",   "We presented a method that ...",           80),
            ]
            total_words = 60 + 70 + 80 = 210
            header_words = ~50   →   210 + 50 = 260 ≥ 200   →   Level 1

            Result: ONE new TextChunk whose text is:
                "Title\n\nAbstract: ...\n\n
                 Section: Limitations\n\nThis approach...\n\n
                 Section: Future Work\n\nWe plan to...\n\n
                 Section: Conclusion\n\nWe presented..."
                section_title = "Limitations + Future Work + Conclusion"

        Example — Level 2 (absorb into the previous chunk):

            small_buffer = [("Acknowledgements", "We thank the reviewers ...", 20)]
            total_words = 20
            header_words = ~50   →   20 + 50 = 70 < 200   →   Level 2

            existing_chunks[-1] was the "Conclusion" chunk (300 words).
            After absorption: existing_chunks[-1].text gets
                "\n\nSection: Acknowledgements\n\nWe thank..." appended.
            section_title becomes "Conclusion + combined"
            Return [] — no new chunk added to the list.
        """
        if not small_sections:
            return []

        # Format each buffered section the same way as a normal section chunk
        combined_parts = [
            f"Section: {t}\n\n{c}" for t, c, _ in small_sections
        ]
        combined_text = f"{header}{chr(10) + chr(10).join(combined_parts)}"
        total_words = sum(w for _, _, w in small_sections)

        # Level 2: combined result is still tiny — absorb into previous chunk
        if total_words + len(header.split()) < 200 and existing_chunks:
            prev = existing_chunks[-1]
            merged = f"{prev.text}\n\n{chr(10) + chr(10).join(combined_parts)}"
            existing_chunks[-1] = TextChunk(
                text=merged,
                metadata=ChunkMetadata(
                    chunk_index=prev.metadata.chunk_index,
                    start_char=0,
                    end_char=len(merged),
                    word_count=len(merged.split()),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=f"{prev.metadata.section_title} + combined",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            return []  # signal to caller: nothing new to append

        # Level 1: produce a new combined chunk
        # Limit the title to at most 3 section names to keep it readable
        titles = [t for t, _, _ in small_sections]
        combined_title = " + ".join(titles[:3])
        if len(titles) > 3:
            combined_title += f" + {len(titles) - 3} more"

        return [
            self._make_section_chunk(
                combined_text, combined_title, len(existing_chunks), arxiv_id, paper_id
            )
        ]

    def _split_large_section(
        self,
        header: str,
        section_text: str,
        section_title: str,
        base_index: int,
        arxiv_id: str,
        paper_id: str,
    ) -> List[TextChunk]:
        """
        Word-chunk a section that exceeds 800 words, then re-attach the header
        to every resulting sub-chunk.

        Flow:
            1. chunk_text(section_text) → raw sub-chunks with word-based overlap
            2. For each raw sub-chunk, prepend the header (title + abstract)
            3. Renumber chunk_index starting from base_index (so global ordering
               is preserved across all chunks of the paper)
            4. Tag section_title as "Section Name (part N)" so users can see
               which section and which part of it was matched

        WHY re-attach the header to every sub-chunk?
        chunk_text() doesn't know it's splitting a section — it just sees raw
        text. Without the header, sub-chunk 3 of "Related Work" would embed as
        a generic text fragment. With the header each sub-chunk embeds with full
        paper context, producing much better cosine similarity against queries.

        Example — "Introduction" section with 1500 words:

            header      = "Attention Is All You Need\n\nAbstract: We propose...\n\n"
            section_text = "Section: Introduction\n\nThe dominant sequence model..."

            chunk_text(section_text) → 3 raw sub-chunks (words[0:600], [500:1100], [1000:1500])

            After re-attaching header:
                sub-chunk 0: header + "Section: Introduction\n\nThe dominant..."
                             section_title = "Introduction (part 1)"   chunk_index = base_index + 0
                sub-chunk 1: header + "...sequence models generally use..."
                             section_title = "Introduction (part 2)"   chunk_index = base_index + 1
                sub-chunk 2: header + "...In this work we propose..."
                             section_title = "Introduction (part 3)"   chunk_index = base_index + 2

            If base_index=0 (Introduction is the first section), chunks get
            indices 0, 1, 2. The next section's chunks start at base_index=3.
        """
        # chunk_text() runs word-based sliding window on the section content
        raw_chunks = self.chunk_text(section_text, arxiv_id, paper_id)
        result = []
        for i, chunk in enumerate(raw_chunks):
            enhanced_text = f"{header}{chunk.text}"
            result.append(
                TextChunk(
                    text=enhanced_text,
                    metadata=ChunkMetadata(
                        # base_index offsets this section's chunks so they don't
                        # collide with chunks from earlier sections of the same paper
                        chunk_index=base_index + i,
                        start_char=chunk.metadata.start_char,
                        # end_char shifts by len(header) because we prepended it
                        end_char=chunk.metadata.end_char + len(header),
                        word_count=len(enhanced_text.split()),
                        overlap_with_previous=chunk.metadata.overlap_with_previous,
                        overlap_with_next=chunk.metadata.overlap_with_next,
                        section_title=f"{section_title} (part {i + 1})",
                    ),
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                )
            )
        return result

    # ── Parsing & filtering helpers ────────────────────────────────────────

    def _parse_sections(self, sections: Union[Dict, str, list]) -> Dict[str, str]:
        """
        Normalise sections into a plain {title: content} dict.

        The sections field in PostgreSQL can arrive in three formats depending
        on how the PDF parser stored it:

            dict  — already the right format, return as-is
            list  — list of {"title": ..., "content": ...} dicts (Docling output)
                    or list of {"heading": ..., "text": ...} (alternative parser)
                    Fall back to "Section N" as the title if neither key exists.
            str   — JSON-encoded dict or list. Recursively parse then normalise.

        Returns {} (empty dict) if the input cannot be interpreted, which causes
        _chunk_by_sections() to return [] and chunk_paper() to fall back to
        word-based chunking.

        Examples:

            # dict — returned as-is
            _parse_sections({"Introduction": "We study...", "Methods": "We use..."})
            → {"Introduction": "We study...", "Methods": "We use..."}

            # list of Docling-style dicts
            _parse_sections([
                {"title": "Introduction", "content": "We study..."},
                {"title": "Methods",      "content": "We use..."},
            ])
            → {"Introduction": "We study...", "Methods": "We use..."}

            # list with alternative keys
            _parse_sections([{"heading": "Intro", "text": "We study..."}])
            → {"Intro": "We study..."}

            # list item with no recognised title key
            _parse_sections([{"body": "some text"}])
            → {"Section 1": ""}   (title falls back to "Section N", content="")

            # JSON-encoded string — recursively parsed
            _parse_sections('{"Introduction": "We study..."}')
            → {"Introduction": "We study..."}

            # unparseable string → empty dict → fallback to word-based chunking
            _parse_sections("not valid JSON")
            → {}
        """
        if isinstance(sections, dict):
            return sections

        if isinstance(sections, list):
            result = {}
            for i, item in enumerate(sections):
                if isinstance(item, dict):
                    # Try "title" first (Docling), then "heading" (alternative parsers)
                    title = item.get("title", item.get("heading", f"Section {i + 1}"))
                    content = item.get("content", item.get("text", ""))
                else:
                    title, content = f"Section {i + 1}", str(item)
                result[title] = content
            return result

        if isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                # Recurse so we handle JSON-encoded dict or JSON-encoded list
                return self._parse_sections(parsed)
            except json.JSONDecodeError:
                logger.warning("Could not parse sections JSON string")

        return {}

    def _filter_sections(
        self, sections_dict: Dict[str, str], abstract: str
    ) -> Dict[str, str]:
        """
        Remove sections that add noise rather than signal to the search index.

        Three categories are filtered out:

        1. Empty sections — no content, nothing to embed.

        2. Metadata/header sections — sections titled "Authors", "Affiliations",
           "Email", "arXiv", etc. These typically contain author names, university
           addresses, and email addresses. Embedding them wastes index space and
           can cause false matches (e.g. a search for "Stanford" matches the
           affiliation block of every paper from Stanford authors, not the content).

        3. Abstract duplicates — some PDF parsers extract the abstract as both
           a standalone field AND as a section in the body text. Indexing it
           twice wastes space and skews results (the paper's abstract chunk would
           appear in every query that matches any paper). We detect duplicates via:
             a) Direct string containment
             b) >80% word overlap (catches minor formatting differences)

        Example — what gets removed:

            sections_dict = {
                "Authors":      "Jane Doe, John Smith",         # ← removed: metadata title
                "arXiv":        "arXiv:2301.07041 [cs.CL]",     # ← removed: metadata title
                "":             "some stray text",              # ← removed: empty title (len<5)
                "Abstract":     "We propose a new method ...",  # ← removed: exact match with abstract
                "Introduction": "We propose a new method ...",  # ← removed: >80% word overlap with abstract
                "Methods":      "We collected 10k samples ...", # ← KEPT: genuine content
                "Results":      "Our model achieves 94.2% ...", # ← KEPT: genuine content
                "":             "",                             # ← removed: empty content
            }

            After _filter_sections():
            → {"Methods": "We collected...", "Results": "Our model achieves..."}
        """
        abstract_words = set(abstract.lower().split())
        filtered = {}

        for title, content in sections_dict.items():
            content = str(content).strip()
            if not content:
                continue
            if self._is_metadata_section(title):
                continue
            if self._is_duplicate_abstract(content, abstract, abstract_words):
                logger.debug(f"Skipping section '{title}' — duplicate of abstract")
                continue
            filtered[title] = content

        return filtered

    def _is_metadata_section(self, title: str) -> bool:
        """
        Return True if the section title indicates author/affiliation boilerplate.

        Heuristics (deliberately conservative to avoid false positives):
            - Exact match against a known metadata keyword set
            - Very short title (< 5 chars) — likely a stray heading artifact
            - Short title (< 20 chars) that CONTAINS a metadata keyword
              e.g. "Author Info", "arXiv preprint"
        """
        t = title.lower().strip()
        metadata_keywords = {
            "content", "header", "authors", "author", "affiliation",
            "email", "arxiv", "preprint", "submitted", "received", "accepted",
        }
        if t in metadata_keywords or len(t) < 5:
            return True
        return any(kw in t for kw in metadata_keywords if len(t) < 20)

    def _is_duplicate_abstract(
        self, content: str, abstract: str, abstract_words: set
    ) -> bool:
        """
        Return True if the section content is substantially the same as the abstract.

        Two checks in order (fast to slow):
            1. Direct string containment — catches exact or near-exact copies.
               We check both directions: abstract in content (abstract appears
               verbatim inside a longer section) and content in abstract (section
               is a truncated version of the abstract).

            2. Word overlap ratio — catches re-formatted duplicates where
               line breaks, punctuation, or minor edits differ but 80%+ of
               the words are shared. Only applied to abstracts with >10 words
               to avoid false positives on very short abstracts.

        The 80% threshold was chosen empirically — it catches reformatted
        duplicates while allowing a "Summary" section that paraphrases the
        abstract (typically 50–60% word overlap) to pass through.
        """
        content_lower = content.lower().strip()
        abstract_lower = abstract.lower().strip()

        # Fast path: direct containment
        if abstract_lower in content_lower or content_lower in abstract_lower:
            return True

        # Slower path: word overlap ratio
        if len(abstract_words) > 10:
            overlap = len(abstract_words & set(content_lower.split()))
            if overlap / len(abstract_words) > 0.8:
                return True

        return False

    # ── Low-level text helpers ─────────────────────────────────────────────

    def _split_into_words(self, text: str) -> List[str]:
        """
        Split text into a list of non-whitespace tokens.

        Uses regex r"\S+" rather than str.split() for consistent behaviour
        across different Unicode whitespace characters (non-breaking spaces,
        em-dashes used as separators, etc.).
        """
        return re.findall(r"\S+", text)

    def _reconstruct_text(self, words: List[str]) -> str:
        """Join word tokens back into a string with single spaces between them."""
        return " ".join(words)

    def _make_chunk(
        self,
        words: List[str],
        chunk_index: int,
        start_char: int,
        end_char: int,
        arxiv_id: str,
        paper_id: str,
    ) -> TextChunk:
        """
        Build a TextChunk from a word list.
        Used for the edge case where the entire text is smaller than min_chunk_size.
        """
        return TextChunk(
            text=self._reconstruct_text(words),
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                word_count=len(words),
                overlap_with_previous=0,
                overlap_with_next=0,
                section_title=None,
            ),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

    def _make_section_chunk(
        self,
        text: str,
        section_title: str,
        chunk_index: int,
        arxiv_id: str,
        paper_id: str,
    ) -> TextChunk:
        """
        Build a TextChunk from a pre-formatted section string.
        Used for section-based chunks where character offsets map to the
        combined header+section string rather than the original document.
        """
        return TextChunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(text),
                word_count=len(text.split()),
                overlap_with_previous=0,  # section chunks don't overlap each other
                overlap_with_next=0,
                section_title=section_title,
            ),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )
