import tempfile
import atexit
import os
import subprocess

temps = []
def remove_all_temps():
    for temp in temps:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
atexit.register(remove_all_temps)

def temp_name():
    (fd, name) = tempfile.mkstemp()
    os.close(fd)
    temps.append(name)
    return name

def run_cmd(cmd):
    n = temp_name()
    with open(n, "w") as f:
        subprocess.call(cmd, stdout=f)
    return n
