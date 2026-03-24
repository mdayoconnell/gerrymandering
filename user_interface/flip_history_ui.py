from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from maps.init_squareland import (
    compute_district_stats,
    overlay_district_boundaries,
    visualize_electoral_college_bar,
)


_ACTIVE_HISTORY_VIEWERS: dict[int, dict[str, object]] = {}


@dataclass(frozen=True)
class FlipHistoryFrame:
    labels: np.ndarray
    accepted_flips: int
    attempts: int
    total_loss: float
    loss_delta: float
    candidate: tuple[int, int] | None = None
    current_district: int | None = None
    target_district: int | None = None


def launch_flip_history_viewer(
    pops: np.ndarray,
    swing: np.ndarray,
    frames: Sequence[FlipHistoryFrame],
    electoral_total_ev: int = 100,
):
    if not frames:
        raise ValueError("frames must include at least one history snapshot.")

    max_district_id = max(int(np.max(frame.labels)) for frame in frames)
    district_vmax = max(max_district_id, 0)
    total_population = float(np.sum(pops))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plt.subplots_adjust(bottom=0.16, hspace=0.35, wspace=0.25)
    population_ax, swing_ax = axes[0]
    district_ax, election_ax = axes[1]

    population_im = population_ax.imshow(pops, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(population_im, ax=population_ax, fraction=0.046, pad=0.04)
    population_ax.set_title("Population")
    population_ax.set_xticks([])
    population_ax.set_yticks([])

    swing_im = swing_ax.imshow(swing, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar(swing_im, ax=swing_ax, fraction=0.046, pad=0.04)
    swing_ax.set_title("Swing")
    swing_ax.set_xticks([])
    swing_ax.set_yticks([])

    district_im = district_ax.imshow(frames[0].labels, cmap="tab20", vmin=0, vmax=district_vmax)
    district_ax.set_title("District Labels")
    district_ax.set_xticks([])
    district_ax.set_yticks([])

    slider_ax = fig.add_axes([0.18, 0.06, 0.64, 0.035])
    slider = Slider(
        slider_ax,
        "Flip",
        0,
        len(frames) - 1,
        valinit=0,
        valstep=1,
    )

    instructions = fig.text(
        0.5,
        0.02,
        "Use the slider, mouse wheel, or left/right arrows to navigate.",
        ha="center",
        va="center",
    )
    status_text = fig.text(0.5, 0.965, "", ha="center", va="top")

    state = {"index": 0, "syncing": False}

    def _remove_boundaries(ax) -> None:
        for collection in list(ax.collections):
            collection.remove()

    def _format_status(frame: FlipHistoryFrame, index: int) -> str:
        if frame.accepted_flips == 0:
            return (
                f"Initial map | frame {index + 1}/{len(frames)} | "
                f"total loss={frame.total_loss:.6f}"
            )
        return (
            f"Flip {frame.accepted_flips} | frame {index + 1}/{len(frames)} | "
            f"attempt {frame.attempts} | pixel {frame.candidate} "
            f"{frame.current_district}->{frame.target_district} | "
            f"delta={frame.loss_delta:.6f} | total={frame.total_loss:.6f}"
        )

    def _render(index: int, *, update_slider: bool) -> None:
        index = max(0, min(index, len(frames) - 1))
        frame = frames[index]
        state["index"] = index

        _remove_boundaries(population_ax)
        _remove_boundaries(swing_ax)
        overlay_district_boundaries(population_ax, frame.labels)
        overlay_district_boundaries(swing_ax, frame.labels)

        district_im.set_data(frame.labels)
        election_ax.clear()
        num_districts = int(np.max(frame.labels)) + 1
        district_pop, _, district_mean_swing = compute_district_stats(
            pops=pops,
            swing=swing,
            labels=frame.labels,
            num_districts=num_districts,
        )
        visualize_electoral_college_bar(
            district_pop=district_pop,
            district_mean_swing=district_mean_swing,
            ax=election_ax,
            total_ev=electoral_total_ev,
            show_numbers=True,
            total_population=total_population,
        )
        status_text.set_text(_format_status(frame, index))

        if update_slider and int(slider.val) != index:
            state["syncing"] = True
            slider.set_val(index)
            state["syncing"] = False

        fig.canvas.draw_idle()

    def _move(delta: int) -> None:
        new_index = max(0, min(state["index"] + delta, len(frames) - 1))
        if new_index == state["index"]:
            return
        _render(new_index, update_slider=True)

    def _on_slider_change(value: float) -> None:
        if state["syncing"]:
            return
        _render(int(value), update_slider=False)

    def _on_key_press(event) -> None:
        if event.key in {"right", "down", "d", "n", "pagedown"}:
            _move(1)
        elif event.key in {"left", "up", "a", "p", "pageup"}:
            _move(-1)
        elif event.key == "home":
            _render(0, update_slider=True)
        elif event.key == "end":
            _render(len(frames) - 1, update_slider=True)

    def _on_scroll(event) -> None:
        step = 0
        if getattr(event, "step", 0) > 0 or event.button == "up":
            step = 1
        elif getattr(event, "step", 0) < 0 or event.button == "down":
            step = -1
        if step:
            _move(step)

    slider.on_changed(_on_slider_change)
    fig.canvas.mpl_connect("key_press_event", _on_key_press)
    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    ui_state = {
        "frames": frames,
        "instructions": instructions,
        "population_image": population_im,
        "slider": slider,
        "status_text": status_text,
        "swing_image": swing_im,
    }
    _ACTIVE_HISTORY_VIEWERS[fig.number] = ui_state
    fig._flip_history_ui = ui_state  # type: ignore[attr-defined]

    def _cleanup(_event) -> None:
        _ACTIVE_HISTORY_VIEWERS.pop(fig.number, None)

    fig.canvas.mpl_connect("close_event", _cleanup)
    _render(0, update_slider=False)
    return fig, slider
