# pipeline/cleaner.py
import logging
import re
from typing import Dict, List

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

class EnhancedDataCleaner:
    """
    Sophisticated data cleaner for handling code blocks, boilerplate, and text normalization.
    """
    def __init__(self):
        self.code_block_pattern = re.compile(r"```.*?```", re.DOTALL)
        self.inline_code_pattern = re.compile(r"`[^`]*`")
        self.boilerplate_patterns = [
            re.compile(r"on this page", re.IGNORECASE),
            re.compile(r"table of contents", re.IGNORECASE),
            re.compile(r"was this page helpful\?", re.IGNORECASE),
            re.compile(r"edit this page on.*", re.IGNORECASE),
            re.compile(r"next\s*→", re.IGNORECASE),
            re.compile(r"←\s*previous", re.IGNORECASE),
            re.compile(r"© \d{4}.* All rights reserved.", re.IGNORECASE),
        ]
        self.markdown_link_pattern = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
        self.extra_whitespace_pattern = re.compile(r"[ \t]+")
        self.extra_newlines_pattern = re.compile(r"\n{3,}")
        self.md = MarkdownIt()

    def _remove_html_and_markdown(self, text: str) -> str:
        try:
            text = self.markdown_link_pattern.sub(r"\1", text)
            html = self.md.render(text)
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()
        except Exception as e:
            logging.error("Error removing HTML/Markdown: %s", e)
            return text

    def clean(self, markdown_content: str) -> str:
        if not isinstance(markdown_content, str):
            logging.warning("Markdown content was not a string. Returning empty.")
            return ""
        text = self.code_block_pattern.sub("", markdown_content)
        text = self.inline_code_pattern.sub("", text)
        text = self._remove_html_and_markdown(text)
        for pattern in self.boilerplate_patterns:
            text = pattern.sub("", text)
        text = text.lower()
        text = self.extra_whitespace_pattern.sub(" ", text)
        text = self.extra_newlines_pattern.sub("\n\n", text)
        return text.strip()

def chunk_document_by_headers(markdown_content: str) -> List[Dict[str, str]]:
    """Splits a markdown document into chunks based on its headers."""
    header_pattern = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    chunks = []
    matches = list(header_pattern.finditer(markdown_content))

    if matches and matches[0].start() > 0:
        chunks.append({
            "header": "Introduction",
            "content": markdown_content[0:matches[0].start()].strip()
        })

    for i, match in enumerate(matches):
        header_text = match.group(2).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_content)
        content = markdown_content[content_start:content_end].strip()
        chunks.append({"header": header_text, "content": content})

    if not chunks and markdown_content:
        chunks.append({"header": "General", "content": markdown_content})

    return chunks