from contextlib import contextmanager
import time
import os

# Create a generator for lock_open
@contextmanager
def lock_open(file, mode, timeout = 20):
    """
    Context manager for opening a file, respecting the lock from other processes.
    """
    # Construct lock file name
    lock_file = file + '.lock'

    # Try opening lock file until timeout
    timeout  = 20/0.1
    retries = 0
    lock = None
    while retries < timeout:
        try:
            # Open lock file
            lock = open(lock_file, 'x')
            break
        except FileExistsError:
            retries += 1
            time.sleep(0.1)

    # If lock file is not opened, raise an exception
    if lock is None:
        raise Exception("LOGGER: Unable to open lock file for {}".format(file))


    f = open(file, mode)

    try:
        yield f
    finally:
        # Delete the lock file
        lock.close()
        os.remove(lock_file)

        f.close()


def exists(path):
    if os.path.exists(path):
        return path

    return exists(os.path.split(path)[0])

# Create a generator for lock_open
@contextmanager
def lock_open_dir(dir, timeout = 20):
    """
    Context manager for opening a file, respecting the lock from other processes.
    """
    # Construct lock file name
    lock_file = exists(os.path.normpath(dir)) + '.lock'

    # Try opening lock file until timeout
    timeout  = 20/0.1
    retries = 0
    lock = None
    while retries < timeout:
        try:
            # Open lock file
            lock = open(lock_file, 'x')
            break
        except FileExistsError:
            retries += 1
            time.sleep(0.1)

    # If lock file is not opened, raise an exception
    if lock is None:
        raise Exception("LOGGER: Unable to open lock file for directory {}".format(dir))


    try:
        yield None
    finally:
        # Delete the lock file
        lock.close()
        os.remove(lock_file)
