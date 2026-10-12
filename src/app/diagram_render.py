"""
Render draw.io diagrams from specs/ as inline SVG for the Architecture tab.

Supports the shapes used in specs/: ellipses, rhombuses, rounded and plain rectangles,
notes, dashed containers with child shapes, and orthogonal edges with labels. Any other
shape is drawn as a plain rectangle.
"""

import html
import textwrap
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]  # x, y, width, height
Point = Tuple[float, float]

FONT_SIZE = 12
LINE_HEIGHT = 16
CHAR_WIDTH_RATIO = 0.55  # average Arial glyph width as a share of the font size
TEXT_PADDING = 8
EDGE_PADDING = 30
OUTER_PADDING = 24
MAX_IFRAME_HEIGHT = 800
ARROW_MARKER = (
    '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" '
    'orient="auto" markerUnits="userSpaceOnUse">'
    '<polygon points="0 0, 10 3.5, 0 7" fill="#333333"/></marker></defs>'
)


def _parse_style(raw: Optional[str]) -> Dict[str, str]:
    style: Dict[str, str] = {}
    for token in (raw or "").split(";"):
        if token:
            key, _, value = token.partition("=")
            style[key] = value
    return style


def _vertex_parent(cell: ET.Element, cells: Dict[str, ET.Element]) -> Optional[ET.Element]:
    parent = cells.get(cell.get("parent", ""))
    return parent if parent is not None and parent.get("vertex") == "1" else None


def _absolute_origin(cell: ET.Element, cells: Dict[str, ET.Element]) -> Point:
    geometry = cell.find("mxGeometry")
    x = float(geometry.get("x", 0))
    y = float(geometry.get("y", 0))
    parent = _vertex_parent(cell, cells)
    if parent is not None:
        px, py = _absolute_origin(parent, cells)
        x += px
        y += py
    return x, y


def _box(cell: ET.Element, cells: Dict[str, ET.Element]) -> Box:
    geometry = cell.find("mxGeometry")
    x, y = _absolute_origin(cell, cells)
    return x, y, float(geometry.get("width", 0)), float(geometry.get("height", 0))


def _depth(cell: ET.Element, cells: Dict[str, ET.Element]) -> int:
    depth = 0
    parent = _vertex_parent(cell, cells)
    while parent is not None:
        depth += 1
        parent = _vertex_parent(parent, cells)
    return depth


def _center(box: Box) -> Point:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _points_attr(points: List[Point]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _shape_svg(style: Dict[str, str], box: Box) -> str:
    x, y, w, h = box
    fill = style.get("fillColor", "#ffffff")
    stroke = style.get("strokeColor", "#000000")
    dash = ' stroke-dasharray="6,4"' if style.get("dashed") == "1" else ""
    common = f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}'
    if "ellipse" in style:
        return f'<ellipse cx="{x + w / 2}" cy="{y + h / 2}" rx="{w / 2}" ry="{h / 2}" {common}/>'
    if "rhombus" in style:
        points = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]
        return f'<polygon points="{_points_attr(points)}" {common}/>'
    if style.get("shape") == "note":
        fold = 14
        points = [(x, y), (x + w - fold, y), (x + w, y + fold), (x + w, y + h), (x, y + h)]
        return f'<polygon points="{_points_attr(points)}" {common}/>'
    radius = 8 if style.get("rounded") == "1" else 0
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" {common}/>'


def _wrap_lines(value: str, width: float) -> List[str]:
    max_chars = max(1, int((width - 2 * TEXT_PADDING) / (FONT_SIZE * CHAR_WIDTH_RATIO)))
    return [
        wrapped
        for line in value.split("\n")
        for wrapped in (textwrap.wrap(line, max_chars) or [""])
    ]


def _text_svg(value: str, style: Dict[str, str], box: Box) -> str:
    x, y, w, h = box
    lines = _wrap_lines(value, w)
    if not any(line.strip() for line in lines):
        return ""
    bold = ' font-weight="bold"' if style.get("fontStyle") == "1" else ""
    if style.get("verticalAlign") == "top" and style.get("align") == "left":
        # Container label, pinned to the top-left corner.
        anchor, text_x, first_y = "start", x + 10, y + 6 + FONT_SIZE
    else:
        anchor = "middle"
        text_x = x + w / 2
        first_y = y + h / 2 - len(lines) * LINE_HEIGHT / 2 + LINE_HEIGHT * 0.75
    tspans = "".join(
        f'<tspan x="{text_x}" y="{first_y + index * LINE_HEIGHT}">{html.escape(line, quote=False)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text font-family="Arial, Helvetica, sans-serif" font-size="{FONT_SIZE}" '
        f'text-anchor="{anchor}" fill="#000000"{bold}>{tspans}</text>'
    )


def _edge_points(src: Box, tgt: Box, style: Dict[str, str]) -> List[Point]:
    """Orthogonal route between two boxes. Side exits (exitX=0 or 1 on both ends) route outside the boxes."""
    exit_x, entry_x = style.get("exitX"), style.get("entryX")
    if exit_x is not None and exit_x == entry_x and float(exit_x) in (0.0, 1.0):
        start = (src[0] + float(exit_x) * src[2], src[1] + float(style.get("exitY", 0.5)) * src[3])
        end = (tgt[0] + float(entry_x) * tgt[2], tgt[1] + float(style.get("entryY", 0.5)) * tgt[3])
        if float(exit_x) == 0:
            side_x = min(src[0], tgt[0]) - EDGE_PADDING
        else:
            side_x = max(src[0] + src[2], tgt[0] + tgt[2]) + EDGE_PADDING
        return [start, (side_x, start[1]), (side_x, end[1]), end]

    sx, sy = _center(src)
    tx, ty = _center(tgt)
    if abs(ty - sy) >= abs(tx - sx):
        start = (sx, src[1] + src[3]) if ty >= sy else (sx, src[1])
        end = (tx, tgt[1]) if ty >= sy else (tx, tgt[1] + tgt[3])
        if abs(tx - sx) < 1:
            return [start, end]
        mid_y = (start[1] + end[1]) / 2
        return [start, (sx, mid_y), (tx, mid_y), end]

    start = (src[0] + src[2], sy) if tx >= sx else (src[0], sy)
    end = (tgt[0], ty) if tx >= sx else (tgt[0] + tgt[2], ty)
    if abs(ty - sy) < 1:
        return [start, end]
    mid_x = (start[0] + end[0]) / 2
    return [start, (mid_x, sy), (mid_x, ty), end]


def _edge_svg(points: List[Point], label: str) -> str:
    path = "M " + " L ".join(f"{x} {y}" for x, y in points)
    svg = (
        f'<path d="{path}" fill="none" stroke="#333333" stroke-width="1.5" '
        'marker-end="url(#arrowhead)"/>'
    )
    if label.strip():
        (x0, y0), (x1, y1) = points[0], points[1]
        if abs(x1 - x0) < abs(y1 - y0):
            label_x, label_y, anchor = (x0 + x1) / 2 + 6, (y0 + y1) / 2, "start"
        else:
            label_x, label_y, anchor = (x0 + x1) / 2, (y0 + y1) / 2 - 6, "middle"
        svg += (
            f'<text x="{label_x}" y="{label_y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="11" fill="#333333" text-anchor="{anchor}">{html.escape(label, quote=False)}</text>'
        )
    return svg


def drawio_to_svg(xml_text: str) -> Tuple[str, int, int]:
    """Convert draw.io XML to an SVG string. Returns the SVG and its width and height in pixels."""
    root = ET.fromstring(xml_text)
    cells = {cell.get("id"): cell for cell in root.iter("mxCell")}
    vertices = [cell for cell in cells.values() if cell.get("vertex") == "1"]
    edges = [cell for cell in cells.values() if cell.get("edge") == "1"]
    boxes = {cell.get("id"): _box(cell, cells) for cell in vertices}

    body: List[str] = []
    # Containers have lower depth, so they are drawn first and their children appear on top.
    for cell in sorted(vertices, key=lambda c: _depth(c, cells)):
        style = _parse_style(cell.get("style"))
        box = boxes[cell.get("id")]
        body.append(_shape_svg(style, box))
        body.append(_text_svg(cell.get("value") or "", style, box))

    all_points: List[Point] = []
    for box in boxes.values():
        x, y, w, h = box
        all_points += [(x, y), (x + w, y + h)]
    for edge in edges:
        source, target = boxes.get(edge.get("source")), boxes.get(edge.get("target"))
        if source is None or target is None:
            continue
        points = _edge_points(source, target, _parse_style(edge.get("style")))
        all_points += points
        body.append(_edge_svg(points, edge.get("value") or ""))

    if not all_points:
        raise ValueError("The diagram has no shapes to draw.")
    min_x = min(x for x, _ in all_points) - OUTER_PADDING
    min_y = min(y for _, y in all_points) - OUTER_PADDING
    width = max(x for x, _ in all_points) - min_x + OUTER_PADDING
    height = max(y for _, y in all_points) - min_y + OUTER_PADDING
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{min_x} {min_y} {width} {height}">'
        f'{ARROW_MARKER}<rect x="{min_x}" y="{min_y}" width="{width}" height="{height}" fill="#ffffff"/>'
        + "".join(body)
        + "</svg>"
    )
    return svg, int(width), int(height)


def drawio_to_html(xml_text: str) -> Tuple[str, int]:
    """Return an HTML page showing the diagram, and the iframe height to display it at."""
    svg, _, height = drawio_to_svg(xml_text)
    page = f'<html><body style="margin:0;background:#ffffff">{svg}</body></html>'
    return page, min(height + 16, MAX_IFRAME_HEIGHT)
