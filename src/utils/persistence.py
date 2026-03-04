import sys
import os
import io

class PersistentTee:
    """
    A file-like object that writes to both stdout and a file.
    Critically, it calls flush() and os.fsync() after every write,
    ensuring that even if the computer loses power, the logs are
    physically on the disk.
    """
    def __init__(self, filename, mode='a'):
        self.terminal = sys.stdout
        self.filename = filename
        self.file = open(filename, mode, encoding='utf-8')
        
    def write(self, message):
        # Write to stdout
        self.terminal.write(message)
        self.terminal.flush()
        
        # Write to file
        self.file.write(message)
        self.file.flush()
        
        # Ensure it's on the disk (handle power failure)
        try:
            os.fsync(self.file.fileno())
        except (AttributeError, io.UnsupportedOperation):
            pass

    def flush(self):
        self.terminal.flush()
        self.file.flush()
        try:
            os.fsync(self.file.fileno())
        except (AttributeError, io.UnsupportedOperation):
            pass

    def close(self):
        self.file.close()

def setup_persistent_logging(log_path="pipeline_exec.log"):
    """Hooks into sys.stdout and sys.stderr with PersistentTee."""
    tee = PersistentTee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"\n[LOGGER] Persistent logging started: {log_path}")
    print(f"[LOGGER] Standard output and error redirected. os.fsync() enabled.")
    return tee
