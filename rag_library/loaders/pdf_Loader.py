from __future__ import annotations

from typing import List, Dict, Any
import os

import fitz  # PyMuPDF

from ..core.Models import Document


def _extract_page_text(page: fitz.Page) -> str:
    """
    Extract text from a single PDF page using PyMuPDF.

    get_text("text") gives a layout-aware plain text, usually much better
    than PyPDF2 for real-world PDFs.
    """
    # You can also try "blocks" or "blocks" + custom ordering if needed.
    text = page.get_text("text")  # "text" is usually the best default
    return (text or "").strip()


def load_pdf_as_documents(
    path: str,
    per_page: bool = True,
    extra_metadata: Dict[str, Any] | None = None,
) -> List[Document]:
    """
    Load a PDF file and return a list of Document objects using PyMuPDF.

    - If per_page=True  -> one Document per page
    - If per_page=False -> one Document for the whole PDF

    extra_metadata is merged into each Document.metadata.

    This uses PyMuPDF, which generally gives better text extraction quality
    than PyPDF2 (better handling of layout, encoding, and real documents).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    extra_metadata = extra_metadata or {}
    documents: List[Document] = []

    # Open with PyMuPDF
    with fitz.open(path) as pdf:
        if per_page:
            for i, page in enumerate(pdf):
                text = _extract_page_text(page)
                if not text:
                    continue

                doc_id = f"{os.path.basename(path)}_page_{i}"
                metadata = {
                    "source": os.path.abspath(path),
                    "page": i,
                    **extra_metadata,
                }
                documents.append(
                    Document(
                        id=doc_id,
                        text=text,
                        metadata=metadata,
                    )
                )
        else:
            all_text_parts: List[str] = []
            for page in pdf:
                t = _extract_page_text(page)
                if t:
                    all_text_parts.append(t)

            full_text = "\n\n".join(all_text_parts).strip()
            if full_text:
                doc_id = os.path.basename(path)
                metadata = {
                    "source": os.path.abspath(path),
                    **extra_metadata,
                }
                documents.append(
                    Document(
                        id=doc_id,
                        text=full_text,
                        metadata=metadata,
                    )
                )

    return documents
