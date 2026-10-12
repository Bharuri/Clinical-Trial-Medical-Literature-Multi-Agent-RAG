import xml.etree.ElementTree as ET

from src.app.architecture import DIAGRAMS, SPECS_DIR, read_diagram


def test_all_diagram_files_exist_and_are_valid_xml():
    for diagram in DIAGRAMS:
        assert diagram.path.parent == SPECS_DIR
        xml = read_diagram(diagram)
        assert xml is not None, diagram.filename
        ET.fromstring(xml)


def test_read_diagram_returns_none_for_missing_file(tmp_path, monkeypatch):
    from src.app import architecture

    monkeypatch.setattr(architecture, "SPECS_DIR", tmp_path)
    missing = architecture.ArchitectureDiagram("Missing", "nope.drawio", "not here")
    assert read_diagram(missing) is None
