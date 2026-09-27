"""The layout check finds small, cut-off and colliding text, and passes a clean figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import figure_layout as fl


def _fig(width=fl.COLUMN_IN, height=2.4):
    return plt.subplots(figsize=(width, height), layout="constrained")


def _kinds(issues):
    return {i.split(":")[0] for i in issues}


def test_a_clean_figure_has_no_issues():
    with fl.print_style():
        fig, ax = _fig()
        ax.plot([0, 1, 2], [0, 1, 0], marker="o", label="series")
        ax.set_xlabel("time (d)")
        ax.set_ylabel("NRMSE")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=1)
        assert fl.check_figure(fig) == []
        plt.close(fig)


def test_small_font_is_reported():
    with fl.print_style():
        fig, ax = _fig()
        ax.set_title("tiny", fontsize=5)
        assert "font 5.0 pt < 7.0 pt" in _kinds(fl.check_figure(fig))
        plt.close(fig)


def test_text_outside_the_canvas_is_reported():
    with fl.print_style():
        fig, ax = _fig()
        ax.set_title("a title far too long for one column of the journal page, so it is cut off at both ends")
        assert "outside the canvas (cut off)" in _kinds(fl.check_figure(fig))
        plt.close(fig)


def test_overlapping_texts_are_reported():
    with fl.print_style():
        fig, ax = _fig()
        ax.text(0.5, 0.5, "first label", transform=ax.transAxes)
        ax.text(0.52, 0.5, "second label", transform=ax.transAxes)
        assert "text overlaps text" in _kinds(fl.check_figure(fig))
        plt.close(fig)


def test_crowded_rotated_tick_labels_are_reported_and_spaced_ones_are_not():
    with fl.print_style():
        fig, ax = _fig(height=2.8)
        ax.set_xticks(range(30))
        ax.set_xticklabels(["label number %d" % i for i in range(30)], rotation=45, ha="right")
        assert "text overlaps text" in _kinds(fl.check_figure(fig))
        plt.close(fig)
        fig, ax = _fig(width=fl.DOUBLE_IN, height=2.8)
        ax.set_xticks(range(8))
        ax.set_xticklabels(["label %d" % i for i in range(8)], rotation=45, ha="right")
        assert fl.check_figure(fig) == []
        plt.close(fig)


def test_text_on_a_marker_and_on_a_line_is_reported():
    with fl.print_style():
        fig, ax = _fig()
        ax.scatter([0.5], [0.5], s=40)
        ax.text(0.5, 0.5, "n=1")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        assert "text on a marker" in _kinds(fl.check_figure(fig))
        plt.close(fig)
        fig, ax = _fig()
        ax.plot([0, 1], [0.5, 0.5])
        ax.text(0.4, 0.49, "threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        assert "text on a line" in _kinds(fl.check_figure(fig))
        plt.close(fig)


def test_a_legend_over_data_is_reported():
    with fl.print_style():
        fig, ax = _fig()
        t = np.linspace(0, 1, 50)
        ax.plot(t, 0.9 + 0.0 * t, label="flat line near the top")
        ax.plot(t, t, label="rising")
        ax.legend(loc="upper left")
        assert "legend covers data" in _kinds(fl.check_figure(fig))
        plt.close(fig)


def test_save_checked_refuses_and_writes(tmp_path):
    written = []

    def save(fig, path):
        written.append(path)
        return [path]

    with fl.print_style():
        fig, ax = _fig()
        ax.set_title("tiny", fontsize=5)
        with pytest.raises(ValueError, match="layout issue"):
            fl.save_checked(fig, tmp_path / "bad.png", save)
        plt.close(fig)
        fig, ax = _fig()
        ax.plot([0, 1], [0, 1])
        assert fl.save_checked(fig, tmp_path / "good.png", save) == [tmp_path / "good.png"]
        plt.close(fig)
    assert written == [tmp_path / "good.png"]


def test_wrap_keeps_lines_within_the_width():
    note = "Outlined cell: best in its row. * more information, never a winner. " * 3
    with fl.print_style():
        fig = plt.figure(figsize=(fl.COLUMN_IN, 1.2))
        fig.text(0.5, 0.5, fl.wrap(note, fl.COLUMN_IN * 0.95), ha="center", va="center", fontsize=7)
        assert fl.check_figure(fig) == []
        plt.close(fig)


def test_dense_rotated_labels_are_tested_as_rotated_rectangles():
    """Axis-aligned boxes of these 45-degree labels overlap; the rotated labels themselves do not."""
    with fl.print_style():
        fig, ax = _fig(height=2.8)
        ax.set_xticks(range(8))
        ax.set_xticklabels(["a longer label %d" % i for i in range(8)], rotation=45, ha="right",
                           rotation_mode="anchor")
        assert fl.check_figure(fig) == []
        renderer = fig.canvas.get_renderer()
        boxes = [fl._rect(t.get_window_extent(renderer)) for t in ax.get_xticklabels()]
        assert any(fl._overlap_depth(boxes[i], boxes[i + 1]) > 0 for i in range(7))
        plt.close(fig)


def test_lines_are_clipped_to_their_axes():
    """A line running past the axis limits is not drawn outside the axes, so a label there is clear of it."""
    with fl.print_style():
        fig, ax = _fig()
        ax.plot([0, 2], [0.5, 0.5])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(1.03, 0.5, "0.500", transform=ax.transAxes, va="center")
        assert fl.check_figure(fig) == []
        plt.close(fig)


def test_a_legend_over_a_band_is_reported():
    with fl.print_style():
        fig, ax = _fig()
        t = np.linspace(0, 1, 20)
        ax.fill_between(t, 0.5, 1.0, label="band")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        assert any(i.startswith("legend covers data") for i in fl.check_figure(fig))
        plt.close(fig)


def test_figure_text_and_text_of_another_axes_are_checked_against_data():
    with fl.print_style():
        fig, ax = _fig()
        ax.plot([0, 1], [0.5, 0.5])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.text(0.55, 0.52, "figure note", ha="center", va="center")
        assert "text on a line" in _kinds(fl.check_figure(fig))
        plt.close(fig)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(fl.DOUBLE_IN, 2.4))
        a2.plot([0, 1], [0.5, 0.5])
        a2.set_xlim(0, 1)
        a2.set_ylim(0, 1)
        box1, box2 = a1.get_position(), a2.get_position()
        x = (box2.x0 + box2.x1) / 2
        a1.text((x - box1.x0) / box1.width, (box2.y0 + 0.5 * box2.height - box1.y0) / box1.height, "stray",
                transform=a1.transAxes, ha="center", va="center")
        assert "text on a line" in _kinds(fl.check_figure(fig))
        plt.close(fig)


def test_line_width_and_square_marker_outline_count():
    with fl.print_style():
        fig, ax = _fig()
        ax.plot([0, 1], [0.5, 0.5], lw=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.annotate("near a thick line", (0.3, 0.5), xytext=(0, 3), textcoords="offset points", va="bottom")
        assert "text on a line" in _kinds(fl.check_figure(fig))
        plt.close(fig)
        fig, ax = _fig()
        ax.plot([0.5], [0.5], marker="s", ms=20, ls="none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.annotate("corner", (0.5, 0.5), xytext=(8.5, 8.5), textcoords="offset points", ha="left", va="bottom")
        assert "text on a marker" in _kinds(fl.check_figure(fig))
        plt.close(fig)
