import errno
import os
import shutil


def acquire_file(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EACCES, errno.EXDEV):
            raise
        # Fall back to an exclusive copy when ownership or filesystem rules prevent linking.
        try:
            output = open(target, 'xb')
        except OSError:
            raise
        try:
            with output, open(source, 'rb') as incoming:
                shutil.copyfileobj(incoming, output)
        except BaseException:
            os.unlink(target)
            raise
