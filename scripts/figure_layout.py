"""Print-size layout and a text-collision check for the v1.1 figures.

WER asks that figures stay legible when reduced to one column (75 mm), and
Wiley takes artwork 80-180 mm wide. Every v1.1 panel is therefore drawn at its
final size, one column (85 mm) or the full page width (178 mm), with no text
below 7 pt, and ``check_figure`` lists every place where text

- is smaller than 7 pt,
- leaves the canvas (it would be cut off in the PDF),
- overlaps other text,
- sits on a data marker, a plotted line or a bar (free text in any axes and figure-level text,
  tested against the drawn data of every axes it covers; lines are clipped to their axes and
  have their width, markers their real outline),
- or where a legend covers plotted data (markers, lines, bars, filled bands, images).

``save_checked`` refuses to write a figure with any such issue, so a layout
regression fails the scripts and their tests instead of reaching the paper.
Geometry is taken from a real draw: only the text the renderer draws is
checked, and rotated labels are tested as matplotlib's own layout box rotated
with the text, not as their axis-aligned boxes. Mathtext sub- and superscripts
(about 70 % of the base size) are not checked against the 7 pt floor.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

MM = 1 / 25.4
COLUMN_IN = 85 * MM
DOUBLE_IN = 178 * MM
MIN_FONT_PT = 7.0
TOLERANCE_PT = 0.5

RC = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "figure.titlesize": 8,
    "figure.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def print_style() -> Iterator[None]:
    """rcParams for figures drawn at their printed size."""
    import matplotlib.pyplot as plt

    with plt.rc_context(RC):
        yield


def _rect(bb) -> np.ndarray:
    return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]], dtype=float)


def _text_polygon(t: Text, renderer) -> np.ndarray:
    """Corners of the drawn text: its layout box, rotated with the text (axis-aligned text: the box itself)."""
    bb = t.get_window_extent(renderer)
    rot = t.get_rotation() % 180.0
    if min(rot, 180.0 - rot) < 1.0 or abs(rot - 90.0) < 1.0:
        return _rect(bb)
    original = t.get_rotation()
    t.set_rotation(0)
    try:
        flat = t.get_window_extent(renderer)
    finally:
        t.set_rotation(original)
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    th = math.radians(rot)
    u = np.array([math.cos(th), math.sin(th)]) * flat.width / 2
    v = np.array([-math.sin(th), math.cos(th)]) * flat.height / 2
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def _overlap_depth(p: np.ndarray, q: np.ndarray) -> float:
    """Separating-axis overlap of two convex polygons (0 when they do not intersect)."""
    depth = math.inf
    for poly in (p, q):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            norm = math.hypot(e[0], e[1])
            if norm == 0:
                continue
            axis = np.array([-e[1], e[0]]) / norm
            pa, qa = p @ axis, q @ axis
            o = min(pa.max(), qa.max()) - max(pa.min(), qa.min())
            if o <= 0:
                return 0.0
            depth = min(depth, o)
    return 0.0 if depth is math.inf else depth


def _shrink(rect: np.ndarray, by: float) -> np.ndarray:
    """A (possibly rotated) rectangle with every edge moved inwards by ``by`` pixels (the touching tolerance)."""
    c = rect.mean(axis=0)
    u, v = rect[1] - rect[0], rect[3] - rect[0]
    lu, lv = math.hypot(*u), math.hypot(*v)
    hu = u / 2 * (max(0.0, 1 - 2 * by / lu) if lu else 0.0)
    hv = v / 2 * (max(0.0, 1 - 2 * by / lv) if lv else 0.0)
    return np.array([c - hu - hv, c + hu - hv, c + hu + hv, c - hu + hv])


def _point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    sign = None
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross == 0:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif sign != s:
            return False
    return True


def _point_segment_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else max(0.0, min(1.0, float((pt - a) @ ab) / denom))
    return float(np.hypot(*(a + t * ab - pt)))


def _circle_hits_polygon(centre: np.ndarray, radius: float, poly: np.ndarray) -> bool:
    if _point_in_polygon(centre, poly):
        return True
    n = len(poly)
    return any(_point_segment_distance(centre, poly[i], poly[(i + 1) % n]) < radius for i in range(n))


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _segment_polygon_distance(a: np.ndarray, b: np.ndarray, poly: np.ndarray) -> float:
    """0 when the segment touches the convex polygon, else the smallest distance between them."""
    if _point_in_polygon(a, poly) or _point_in_polygon(b, poly):
        return 0.0
    n = len(poly)
    if any(_segments_cross(a, b, poly[i], poly[(i + 1) % n]) for i in range(n)):
        return 0.0
    d = min(_point_segment_distance(a, poly[i], poly[(i + 1) % n]) for i in range(n))
    return min(d, min(_point_segment_distance(corner, a, b) for corner in poly))


def _clip_segment(a: np.ndarray, b: np.ndarray, box) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky: the part of segment a-b inside the box (what the renderer draws), or None."""
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - box.x0), (d[0], box.x1 - a[0]), (-d[1], a[1] - box.y0), (d[1], box.y1 - a[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return a + t0 * d, a + t1 * d


def _visible_segments(xy: np.ndarray, box) -> list[tuple[np.ndarray, np.ndarray]]:
    """Segments of a polyline clipped to its axes, as drawn."""
    segs = []
    finite = np.all(np.isfinite(xy), axis=1)
    for i in range(len(xy) - 1):
        if finite[i] and finite[i + 1]:
            seg = _clip_segment(xy[i], xy[i + 1], box)
            if seg is not None:
                segs.append(seg)
    return segs


def _drawn_texts(fig) -> list[Text]:
    """Texts the renderer actually draws (tick pools hold undrawn, stale labels)."""
    drawn: list[Text] = []
    original = Text.draw

    def recording(self, renderer):
        if self.get_visible() and self.get_text().strip():
            drawn.append(self)
        return original(self, renderer)

    Text.draw = recording
    try:
        fig.canvas.draw()
    finally:
        Text.draw = original
    seen, out = set(), []
    for t in drawn:
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def _label(t: Text) -> str:
    s = t.get_text().replace("\n", " ")
    return s if len(s) <= 40 else s[:37] + "..."


def _marker_reach(marker, fillstyle="full") -> float:
    """Largest distance of the marker outline from its centre, per point of marker size."""
    style = MarkerStyle(marker, fillstyle)
    verts = style.get_path().transformed(style.get_transform()).vertices
    return float(np.max(np.hypot(verts[:, 0], verts[:, 1]))) if len(verts) else 0.5


def _markers(ax) -> list[tuple[np.ndarray, float, str]]:
    """(centre, reach in px, owner) of every drawn data marker in the axes, edge width included."""
    px = ax.figure.dpi / 72.0
    out = []
    for line in ax.lines:
        if not line.get_visible() or line.get_marker() in (None, "None", "", " ", ","):
            continue
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))
        r = (_marker_reach(line.get_marker(), line.get_fillstyle()) * line.get_markersize()
             + line.get_markeredgewidth() / 2) * px
        out += [(p, r, line.get_label()) for p in xy if np.all(np.isfinite(p))]
    for coll in ax.collections:
        if not isinstance(coll, PathCollection) or not coll.get_visible():
            continue
        offsets = np.asarray(coll.get_offsets(), dtype=float)
        if offsets.size == 0:
            continue
        xy = coll.get_offset_transform().transform(offsets)
        sizes = coll.get_sizes()
        paths = coll.get_paths()
        widths = coll.get_linewidths()
        for i, p in enumerate(xy):
            if not np.all(np.isfinite(p)):
                continue
            s = sizes[i % len(sizes)] if len(sizes) else 36.0
            verts = paths[i % len(paths)].vertices if paths else np.array([[0.5, 0.0]])
            reach = float(np.max(np.hypot(verts[:, 0], verts[:, 1])))
            lw = widths[i % len(widths)] if len(widths) else 0.0
            out.append((p, (reach * math.sqrt(s) + lw / 2) * px, coll.get_label()))
    return out


def _polylines(ax) -> list[tuple[np.ndarray, float, str]]:
    """(display vertices, half line width in px, owner) of every drawn line, error bars included."""
    px = ax.figure.dpi / 72.0
    out = []
    for line in ax.lines:
        if not line.get_visible() or line.get_linestyle() in ("None", "none", " ", ""):
            continue
        out.append((line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float)),
                    line.get_linewidth() / 2 * px, line.get_label()))
    for coll in ax.collections:
        if isinstance(coll, LineCollection) and coll.get_visible():
            trans = coll.get_transform()
            widths = coll.get_linewidths()
            for k, seg in enumerate(coll.get_segments()):
                lw = widths[k % len(widths)] if len(widths) else 1.0
                out.append((trans.transform(np.asarray(seg, dtype=float)), lw / 2 * px, coll.get_label()))
    return out


def _filled_patches(ax, renderer) -> list[tuple[np.ndarray, str]]:
    out = []
    for patch in ax.patches:
        if isinstance(patch, Rectangle) and patch.get_visible() and patch.get_fill() \
                and patch.get_width() != 0 and patch.get_height() != 0:
            out.append((_rect(patch.get_window_extent(renderer)), patch.get_label()))
    return out


def _areas(ax) -> list[tuple[object, str]]:
    """Filled areas (fill_between bands and similar) as display-space paths."""
    out = []
    for coll in ax.collections:
        if isinstance(coll, PolyCollection) and coll.get_visible():
            trans = coll.get_transform()
            out += [(path.transformed(trans), coll.get_label()) for path in coll.get_paths()]
    return out


def _data_hit(ax, region: np.ndarray, renderer, with_areas: bool) -> tuple[str, str] | None:
    """(kind, owner) of the first drawn data element of ``ax`` that meets the convex ``region``, else None."""
    box = ax.bbox
    lo, hi = region.min(axis=0), region.max(axis=0)
    if hi[0] < box.x0 or lo[0] > box.x1 or hi[1] < box.y0 or lo[1] > box.y1:
        return None
    for centre, r, owner in _markers(ax):
        if box.x0 <= centre[0] <= box.x1 and box.y0 <= centre[1] <= box.y1 and _circle_hits_polygon(centre, r, region):
            return "marker", owner
    for xy, half_width, owner in _polylines(ax):
        if any(_segment_polygon_distance(a, b, region) <= half_width for a, b in _visible_segments(xy, box)):
            return "line", owner
    for rect, owner in _filled_patches(ax, renderer):
        if _overlap_depth(region, rect) > 0:
            return "bar", owner
    if with_areas:
        clip = Bbox.intersection(Bbox.from_extents(lo[0], lo[1], hi[0], hi[1]), box)
        if clip is not None:
            for path, owner in _areas(ax):
                if path.intersects_bbox(clip, filled=True):
                    return "area", owner
            for image in ax.images:
                if image.get_visible() and Bbox.intersection(image.get_window_extent(renderer), clip) is not None:
                    return "image", "image"
    return None


def check_figure(fig, min_font: float = MIN_FONT_PT, tolerance_pt: float = TOLERANCE_PT) -> list[str]:
    """Every text-layout problem of ``fig`` at its saved size, as readable strings (empty = clean).

    Sub- and superscripts in mathtext are drawn at about 70 % of the base size; only the base size is
    checked against ``min_font``.
    """
    texts = _drawn_texts(fig)
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    tol = tolerance_pt * px
    width, height = fig.bbox.width, fig.bbox.height
    issues: list[str] = []
    polys = {id(t): _text_polygon(t, renderer) for t in texts}

    for t in texts:
        size = t.get_fontsize()
        if size < min_font - 1e-6:
            issues.append("font %.1f pt < %.1f pt: '%s'" % (size, min_font, _label(t)))
        p = polys[id(t)]
        if p[:, 0].min() < -tol or p[:, 1].min() < -tol or p[:, 0].max() > width + tol \
                or p[:, 1].max() > height + tol:
            issues.append("outside the canvas (cut off): '%s'" % _label(t))

    boxes = [(t, polys[id(t)]) for t in texts]
    for i in range(len(boxes)):
        ti, pi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, pj = boxes[j]
            if pi[:, 0].max() < pj[:, 0].min() or pj[:, 0].max() < pi[:, 0].min() \
                    or pi[:, 1].max() < pj[:, 1].min() or pj[:, 1].max() < pi[:, 1].min():
                continue
            if _overlap_depth(pi, pj) > tol:
                issues.append("text overlaps text: '%s' / '%s'" % (_label(ti), _label(tj)))

    figure_texts = [t for t in list(fig.texts) + [getattr(fig, name, None) for name in
                                                   ("_suptitle", "_supxlabel", "_supylabel")]
                    if t is not None and id(t) in polys]
    own_axes = {id(t): ax for ax in fig.axes for t in ax.texts if id(t) in polys}
    for t in [t for ax in fig.axes for t in ax.texts if id(t) in polys] + figure_texts:
        p = _shrink(polys[id(t)], tol)
        ax = own_axes.get(id(t))
        if ax is not None:
            box = ax.bbox
            inside = (p[:, 0] >= box.x0) & (p[:, 0] <= box.x1) & (p[:, 1] >= box.y0) & (p[:, 1] <= box.y1)
            if inside.any() and not inside.all():
                issues.append("text crosses the axes frame: '%s'" % _label(t))
        for other in fig.axes:
            hit = _data_hit(other, p, renderer, with_areas=False)
            if hit is not None:
                issues.append("text on a %s: '%s' / %s" % (hit[0], _label(t), hit[1]))
                break

    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                   if ax.get_legend() is not None and ax.get_legend().get_visible()]
    for leg in legends:
        region = _shrink(_rect(leg.get_window_extent(renderer)), tol)
        for ax in fig.axes:
            hit = _data_hit(ax, region, renderer, with_areas=True)
            if hit is not None:
                issues.append("legend covers data: %s (%s)" % (hit[1], hit[0]))
                break
    return issues


def save_checked(fig, png_path: Path, save) -> list[Path]:
    """Check ``fig`` and write it with ``save(fig, png_path)``; refuse a figure with layout issues."""
    issues = check_figure(fig)
    if issues:
        raise ValueError("%s: %d layout issue(s):\n  %s" % (Path(png_path).name, len(issues), "\n  ".join(issues)))
    return save(fig, png_path)


def wrap(text: str, width_in: float, size_pt: float = MIN_FONT_PT) -> str:
    """Wrap a note so that it fits ``width_in`` at ``size_pt`` (about 0.5 em per character)."""
    import textwrap

    chars = max(20, int(width_in * 72.0 / (0.52 * size_pt)))
    return "\n".join(textwrap.wrap(text, chars))


__all__ = ["COLUMN_IN", "DOUBLE_IN", "MIN_FONT_PT", "MM", "RC", "check_figure", "print_style", "save_checked",
           "wrap"]
