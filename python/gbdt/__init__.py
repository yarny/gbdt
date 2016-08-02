def _setup_lib_path():
    import os
    import sys
    abspath = os.path.dirname(os.path.abspath(__file__))
    so_lib_path = 'darwin_x86_64' if sys.platform == 'darwin' else 'linux_x86_64'
    sys.path.append('{0}/lib/{1}'.format(abspath, so_lib_path))

_setup_lib_path()

from ._version import version_info, __version__
from libgbdt import DataStore
from libgbdt import Forest
from libgbdt import init_logging
from ._gbdt import train
from ._forest_visualizer import ForestVisualizer
