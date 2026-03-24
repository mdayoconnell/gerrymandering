from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from model.loss import LossWeights


_WEIGHTS_LOCK = threading.Lock()
_CURRENT_WEIGHTS = LossWeights()
PUBLISHED_WEIGHTS = asdict(_CURRENT_WEIGHTS)
_ACTIVE_SLIDER_UIS: dict[int, dict[str, object]] = {}


def get_loss_weights() -> LossWeights:
    with _WEIGHTS_LOCK:
        return LossWeights(**asdict(_CURRENT_WEIGHTS))


def set_loss_weights(weights: LossWeights) -> None:
    global _CURRENT_WEIGHTS
    with _WEIGHTS_LOCK:
        _CURRENT_WEIGHTS = LossWeights(
            election_outcome=float(weights.election_outcome),
            fault_tolerance=float(weights.fault_tolerance),
            district_cleanliness=float(weights.district_cleanliness),
            district_balance=float(weights.district_balance),
        )


def publish_loss_weights(weights: LossWeights | None = None) -> dict[str, float]:
    if weights is None:
        weights = get_loss_weights()
    payload = asdict(weights)
    with _WEIGHTS_LOCK:
        PUBLISHED_WEIGHTS.clear()
        PUBLISHED_WEIGHTS.update(payload)
    return payload


def launch_loss_weight_sliders(
    initial: LossWeights | None = None,
    on_change: Callable[[LossWeights], None] | None = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
    step: float = 0.01,
):
    if initial is None:
        initial = get_loss_weights()
    else:
        set_loss_weights(initial)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_axis_off()
    plt.subplots_adjust(left=0.25, right=0.9, top=0.95, bottom=0.15, hspace=0.4)

    slider_axes = [
        plt.axes([0.25, 0.78, 0.6, 0.08]),
        plt.axes([0.25, 0.60, 0.6, 0.08]),
        plt.axes([0.25, 0.42, 0.6, 0.08]),
        plt.axes([0.25, 0.24, 0.6, 0.08]),
    ]

    sliders = {
        "election_outcome": Slider(
            slider_axes[0],
            "Election",
            -1.0,
            1.0,
            valinit=initial.election_outcome,
            valstep=2.0,
        ),
        "fault_tolerance": Slider(
            slider_axes[1],
            "Fault tol",
            min_value,
            max_value,
            valinit=initial.fault_tolerance,
            valstep=step,
        ),
        "district_cleanliness": Slider(
            slider_axes[2],
            "Cleanliness",
            min_value,
            max_value,
            valinit=initial.district_cleanliness,
            valstep=step,
        ),
        "district_balance": Slider(
            slider_axes[3],
            "Balance",
            min_value,
            max_value,
            valinit=initial.district_balance,
            valstep=step,
        ),
    }

    def _update(_val):
        new_weights = LossWeights(
            election_outcome=sliders["election_outcome"].val,
            fault_tolerance=sliders["fault_tolerance"].val,
            district_cleanliness=sliders["district_cleanliness"].val,
            district_balance=sliders["district_balance"].val,
        )
        set_loss_weights(new_weights)
        publish_loss_weights(new_weights)
        if on_change is not None:
            on_change(new_weights)

    for slider in sliders.values():
        slider.on_changed(_update)

    # Matplotlib widgets must be kept alive while the figure is open.
    ui_state = {
        "axes": slider_axes,
        "sliders": sliders,
    }
    _ACTIVE_SLIDER_UIS[fig.number] = ui_state
    fig._loss_weight_ui = ui_state  # type: ignore[attr-defined]

    def _cleanup(_event):
        _ACTIVE_SLIDER_UIS.pop(fig.number, None)

    fig.canvas.mpl_connect("close_event", _cleanup)

    publish_loss_weights(initial)
    return fig, sliders


if __name__ == "__main__":
    _fig, _sliders = launch_loss_weight_sliders()
    plt.show()
