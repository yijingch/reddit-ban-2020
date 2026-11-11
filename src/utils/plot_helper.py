from matplotlib.ticker import FuncFormatter
import numpy as np 

def _compact_count_label(val):
    """Format original count-scale values compactly."""
    if val >= 1_000_000_000:
        return f"{val/1_000_000_000:.0f}B"
    if val >= 1_000_000:
        return f"{val/1_000_000:.0f}M"
    if val >= 1_000:
        return f"{val/1_000:.0f}K"
    return f"{val:.0f}"


def log1p_count_formatter(x, pos=None):
    """Formatter for axes plotted in np.log1p(count), labeled as original counts."""
    val = np.expm1(x)
    if val < 0:
        return ""
    return _compact_count_label(val)


def set_log1p_count_axis(ax):
    """Set clean count-scale ticks for an axis whose data are np.log1p(count)."""
    xmin, xmax = ax.get_xlim()
    max_count = max(1, np.expm1(xmax))
    nice_counts = np.array([1,10, 1_000, 100_000,10_000_000, 1_000_000_000], dtype=float)
    ticks = np.log1p(nice_counts[nice_counts <= max_count * 1.05])
    ticks = ticks[ticks >= max(0, xmin) - 1e-9]
    if len(ticks) > 0:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(log1p_count_formatter))