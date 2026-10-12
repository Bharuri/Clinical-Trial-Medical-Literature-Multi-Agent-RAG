import re
import xml.etree.ElementTree as ET

import pytest

from src.app.architecture import DIAGRAMS, read_diagram
from src.app.diagram_render import (
    CHAR_WIDTH_RATIO,
    FONT_SIZE,
    TEXT_PADDING,
    drawio_to_html,
    drawio_to_svg,
)

MINIMAL = """<mxfile><diagram id="d" name="d"><mxGraphModel><root>
<mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="a" value="Start &amp; go" style="ellipse;html=0;" vertex="1" parent="1">
  <mxGeometry x="0" y="0" width="100" height="50" as="geometry"/></mxCell>
<mxCell id="b" value="Decide?" style="rhombus;html=0;" vertex="1" parent="1">
  <mxGeometry x="0" y="120" width="100" height="60" as="geometry"/></mxCell>
<mxCell id="c" value="Box" style="rounded=1;html=0;" vertex="1" parent="1">
  <mxGeometry x="200" y="120" width="100" height="60" as="geometry"/></mxCell>
<mxCell id="e1" value="yes" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;" edge="1" parent="1" source="b" target="c">
  <mxGeometry relative="1" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>
"""


@pytest.mark.parametrize("diagram", DIAGRAMS, ids=lambda d: d.filename)
def test_each_architecture_diagram_renders_every_edge_with_an_arrow(diagram):
    xml = read_diagram(diagram)
    edge_count = sum(1 for cell in ET.fromstring(xml).iter("mxCell") if cell.get("edge") == "1")

    svg, width, height = drawio_to_svg(xml)

    assert svg.startswith("<svg")
    assert width > 0 and height > 0
    assert svg.count('marker-end="url(#arrowhead)"') == edge_count


def test_diagram_text_is_escaped_in_svg():
    svg, _, _ = drawio_to_svg(MINIMAL)

    assert "Start &amp; go" in svg
    assert "Start & go" not in svg


def test_minimal_diagram_draws_each_shape_kind():
    svg, _, _ = drawio_to_svg(MINIMAL)

    assert "<ellipse" in svg
    assert "<polygon" in svg  # rhombus
    assert '<rect x="200.0"' in svg or '<rect x="200"' in svg
    assert svg.count("<path") == 1
    assert ">yes<" in svg


def test_long_label_wraps_inside_its_shape():
    long_label = "word " * 40
    xml = MINIMAL.replace("Box", long_label.strip())

    svg, _, _ = drawio_to_svg(xml)

    lines = [line for line in re.findall(r"<tspan[^>]*>([^<]*)</tspan>", svg) if "word" in line]
    assert len(lines) > 3
    max_chars = int((100 - 2 * TEXT_PADDING) / (FONT_SIZE * CHAR_WIDTH_RATIO))
    assert all(len(line) <= max_chars for line in lines)


def test_html_wrapper_caps_iframe_height():
    page, height = drawio_to_html(MINIMAL)

    assert page.startswith("<html>")
    assert "<svg" in page
    assert 0 < height <= 800


def test_real_retrieval_diagram_renders_with_container_children():
    retrieval = next(d for d in DIAGRAMS if d.title == "Retrieval")

    svg, _, _ = drawio_to_svg(read_diagram(retrieval))

    assert "Hybrid retrieval" in svg
    assert "Ask &amp; Verify" in svg
