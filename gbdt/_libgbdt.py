import sys

if sys.platform == 'darwin':
    if sys.version_info < (3,):
        from .lib.darwin_x86_64.py27 import libgbdt
        from .lib.darwin_x86_64.py27.libgbdt import init_logging
        from .lib.darwin_x86_64.py27.libgbdt import BucketizedFloatColumn
        from .lib.darwin_x86_64.py27.libgbdt import StringColumn
        from .lib.darwin_x86_64.py27.libgbdt import RawFloatColumn
    else:
        from .lib.darwin_x86_64.py3 import libgbdt
        from .lib.darwin_x86_64.py3.libgbdt import init_logging
        from .lib.darwin_x86_64.py3.libgbdt import BucketizedFloatColumn
        from .lib.darwin_x86_64.py3.libgbdt import StringColumn
        from .lib.darwin_x86_64.py3.libgbdt import RawFloatColumn
elif 'linux' in sys.platform:
    if sys.version_info < (3,):
        from .lib.linux_x86_64.py27 import libgbdt
        from .lib.linux_x86_64.py27.libgbdt import init_logging
        from .lib.linux_x86_64.py27.libgbdt import BucketizedFloatColumn
        from .lib.linux_x86_64.py27.libgbdt import StringColumn
        from .lib.linux_x86_64.py27.libgbdt import RawFloatColumn
    else:
        raise NotImplementedError
else:
    raise NotImplementedError
