"""
Access to the draw.io architecture diagrams in the specs folder for the Architecture tab.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SPECS_DIR = Path(__file__).resolve().parents[2] / "specs"


@dataclass(frozen=True)
class ArchitectureDiagram:
    title: str
    filename: str
    description: str

    @property
    def path(self) -> Path:
        return SPECS_DIR / self.filename


DIAGRAMS = (
    ArchitectureDiagram(
        title="Full application (high level)",
        filename="application-architecture.drawio",
        description="The Streamlit tabs, the question-answering flow, the DuckDB stores, the models, and offline ingestion.",
    ),
    ArchitectureDiagram(
        title="Ingestion",
        filename="ingestion-architecture.drawio",
        description="How PDFs are extracted, chunked, embedded, and stored in DuckDB.",
    ),
    ArchitectureDiagram(
        title="Retrieval",
        filename="retrieval-architecture.drawio",
        description="How a question is checked against saved answers, retrieved with hybrid search, and sent to synthesis.",
    ),
    ArchitectureDiagram(
        title="Synthesis and fact-check loop",
        filename="synthesis-factcheck-loop.drawio",
        description="How a draft is written, fact-checked, and revised up to two times.",
    ),
)


def read_diagram(diagram: ArchitectureDiagram) -> Optional[str]:
    """Return the diagram's draw.io XML, or None if the file is missing."""
    if not diagram.path.is_file():
        return None
    return diagram.path.read_text(encoding="utf-8")
