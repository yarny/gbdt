import sys

if sys.platform == 'darwin':
    from .lib.darwin_x86_64 import libgbdt
    from .lib.darwin_x86_64.libgbdt import init_logging
    from .lib.darwin_x86_64.libgbdt import BucketizedFloatColumn
    from .lib.darwin_x86_64.libgbdt import StringColumn
    from .lib.darwin_x86_64.libgbdt import RawFloatColumn
else:
    from .lib.linux_x86_64 import libgbdt
    from .lib.linux_x86_64.libgbdt import init_logging
    from .lib.linux_x86_64.libgbdt import BucketizedFloatColumn
    from .lib.linux_x86_64.libgbdt import StringColumn
    from .lib.linux_x86_64.libgbdt import RawFloatColumn