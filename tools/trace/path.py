import argparse
import signal
import subprocess
import sys
import os

home_dir = os.environ.get("HOME")
print(home_dir)

sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.10/dist-packages")
sys.path.append(f"{home_dir}/.local/lib/python3-dist-packages")

print(sys.path)

def get_pid(process_name):
    try:
        pid = subprocess.check_output(['pgrep', '-f', process_name]).decode().strip()
        return int(pid)
    except subprocess.CalledProcessError:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    milvus_pid = get_pid("milvus")
    print("milvus pid:", milvus_pid)
    if milvus_pid:
        os.kill(milvus_pid, signal.SIGKILL)
