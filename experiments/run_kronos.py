"""Drive Kronos zero-shot inference on the remote GPU host and pull the result.

Thin wrapper so `run_all.py` has a single entry point per arm. Exact remote
commands (and the environment they assume) are recorded in docs/reproduction.md.
"""
import os
import subprocess
import sys

ROOT = "/opt/data/kvab"
sys.path.insert(0, f"{ROOT}/code/data")


def main():
    from gpu import connect
    import paramiko
    # upload the current script, then run the job in the background on the host
    t = connect()
    sf = paramiko.SFTPClient.from_transport(t)
    sf.put(f"{ROOT}/experiments/kronos_infer.py", "C:/Users/vikk/kvab/kronos_code/kronos_infer.py")
    sf.close()
    t.close()
    cmd = ("cd /d C:\\Users\\vikk\\kvab\\kronos_code && "
           'set "PYTHONPATH=C:\\Users\\vikk\\kvab\\libs;C:\\Users\\vikk\\kvab\\kronos_code" && '
           'set "HF_HOME=C:\\Users\\vikk\\kvab\\hf" && set PYTHONUNBUFFERED=1 && '
           "C:\\Users\\vikk\\AppData\\Local\\Programs\\Python\\Python313\\python.exe -u "
           "kronos_infer.py --start 2014-01-02 --end 2024-12-31 --topn 100 "
           "> C:\\Users\\vikk\\kvab\\kronos.log 2>&1")
    subprocess.run([sys.executable, f"{ROOT}/code/data/gpu.py", "run", cmd], check=True)
    # also measure latency on the same hardware
    subprocess.run([sys.executable, f"{ROOT}/code/data/gpu.py", "put",
                    f"{ROOT}/experiments/kronos_latency.py", "C:/Users/vikk/kvab/kronos_code/kronos_latency.py"],
                   check=True)
    subprocess.run([sys.executable, f"{ROOT}/code/data/gpu.py", "run",
                    'cd /d C:\\Users\\vikk\\kvab\\kronos_code && set "PYTHONPATH=C:\\Users\\vikk\\kvab\\libs;C:\\Users\\vikk\\kvab\\kronos_code" && '
                    'set "HF_HOME=C:\\Users\\vikk\\kvab\\hf" && '
                    "C:\\Users\\vikk\\AppData\\Local\\Programs\\Python\\Python313\\python.exe kronos_latency.py"],
                   check=True)
    # pull artefacts
    t = connect()
    sf = paramiko.SFTPClient.from_transport(t)
    os.makedirs(f"{ROOT}/results/raw", exist_ok=True)
    for remote, local in [("C:/Users/vikk/kvab/kronos_pred.parquet", f"{ROOT}/results/raw/kronos_pred.parquet"),
                          ("C:/Users/vikk/kvab/latency.json", f"{ROOT}/results/raw/latency.json")]:
        try:
            sf.get(remote, local)
            print("pulled", local)
        except Exception as e:
            print("pull failed", remote, e)
    sf.close()
    t.close()


if __name__ == "__main__":
    main()
