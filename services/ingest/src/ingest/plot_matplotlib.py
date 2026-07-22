"""Live matplotlib waterfall/heatmap of amplitude across subcarriers.

Optional debug tool (`pip install -e ".[plot]"`); matplotlib/numpy are only
imported when this function actually runs, so the base install never needs
them. Runs a blocking GUI event loop — call it from the main thread.
"""

from __future__ import annotations

from ingest.buffer import RingBuffer


def run_matplotlib_waterfall(buffer: RingBuffer, *, refresh_hz: float = 10.0) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    im = ax.imshow(
        np.zeros((buffer.capacity, 1)),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel("subcarrier index")
    ax.set_ylabel("frame (most recent at top)")
    ax.set_title("CSI amplitude waterfall")
    fig.colorbar(im, ax=ax, label="amplitude")

    def update(_frame_num: int):
        frames = buffer.snapshot()
        if not frames:
            return (im,)

        matrix = np.array([f["amplitude"] for f in frames])
        pad_rows = buffer.capacity - matrix.shape[0]
        if pad_rows > 0:
            matrix = np.vstack([np.full((pad_rows, matrix.shape[1]), matrix.min()), matrix])

        im.set_data(matrix)
        im.set_extent((0, matrix.shape[1], 0, matrix.shape[0]))
        im.set_clim(vmin=matrix.min(), vmax=matrix.max())
        return (im,)

    # Keep a reference so the animation isn't garbage-collected mid-run.
    _animation = animation.FuncAnimation(
        fig, update, interval=1000 / refresh_hz, blit=False, cache_frame_data=False
    )
    plt.show()
