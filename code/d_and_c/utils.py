import time
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import warnings
from typing import Any, Callable, Tuple, Optional

class StreamToLogger:
    """
    File-like object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""
        
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
            
    def flush(self):
        pass

def log_stdout_and_warnings(log_path):
        """
        Logs console messages and warnings.

        Parameters:
            log_path (str): path where log will be written.
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_path,
            filemode='w'
        )

        # Redirect stdout and stderr to the logger so all prints/warnings are captured in the log file.
        sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

        # Capture warnings in the log file as well.
        warnings.simplefilter("always")
        logging.captureWarnings(True)

        # Override warnings.showwarning to log warnings via the logging module
        def custom_showwarning(message, category, filename, lineno, file=None, line=None):
            logging.getLogger("py.warnings").error(f"{filename}:{lineno}: {category.__name__}: {message}")
        warnings.showwarning = custom_showwarning


def benchmark(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure runtime in seconds and return the function result along with the elapsed time.

    Parameters:
        func (Callable): The function to benchmark.
        *args (Any): Positional arguments to pass to func.
        **kwargs (Any): Keyword arguments to pass to func.

    Returns:
        output (Tuple[Any, float]): A tuple where the first element is the result of func and the second element is the elapsed time in seconds.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def plot_3D_to_2D(x: np.ndarray,
                  projection: np.ndarray,
                  method: Any,
                  title: str,
                  color: np.ndarray,
                  path: Optional[str] = None,
                  empty: bool = False,
                  format: str = "png") -> Optional[plt.Figure]:
    """
    Plot a 3D dataset along with its 2D projection.

    Parameters:
        x (np.ndarray): The original 3D data points arranged as a matrix.
        projection (np.ndarray): The computed 2D projection of the data.
        method (Any): The dimensionality reduction method used (or its name).
        title (str): Title of the plot.
        color (np.ndarray): Array of colors corresponding to each data point.
        path (Optional[str], optional): File path to save the figure. If None, the figure is returned (default is None).
        empty (bool, optional): If True, the parent directory specified in path is removed (default False).
        format (str, optional): File format for saving the figure (default "png").

    Returns:
        Optional[plt.Figure]: The matplotlib Figure object if no path is provided; otherwise, None.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title, fontsize=16)

    # Original data 3D plot
    ax1 = fig.add_subplot(234, projection='3d')
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)
    ax1.set_title("Original Data")

    # Original data 2D projections
    ax2 = fig.add_subplot(231)
    ax2.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.set_title("Original Data (dims 1,2)")

    ax3 = fig.add_subplot(232)
    ax3.scatter(x[:, 0], x[:, 2], c=color, cmap=plt.cm.Spectral)
    ax3.set_title("Original Data (dims 1,3)")

    ax4 = fig.add_subplot(233)
    ax4.scatter(x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)
    ax4.set_title("Original Data (dims 2,3)")

    # DR method projection
    ax5 = fig.add_subplot(235)
    ax5.scatter(projection[:, 0], projection[:, 1],
                c=color, cmap=plt.cm.Spectral)
    ax5.set_title(f"D&C {method} Embedding")

    plt.tight_layout()

    if path:
        if empty:
            parent_dir = os.path.dirname(path)
            if os.path.exists(parent_dir):
                shutil.rmtree(parent_dir)
            os.makedirs(parent_dir)
        plt.savefig(path + '.' + format, format=format)
        plt.close()
    else:
        return fig
