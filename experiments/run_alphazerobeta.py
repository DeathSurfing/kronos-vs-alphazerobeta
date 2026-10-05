"""Drive AlphaZeroBeta training on the remote GPU host and pull the weight windows."""
import os
import subprocess
import sys

ROOT = "/opt/data/kvab"
STEPS = os.environ.get("AZB_STEPS", "200")
TOP_N = os.environ.get("AZB_TOPN", "50")


def main():
    sys.path.insert(0, f"{ROOT}/code/data")
    from gpu import connect
    import paramiko
    t = connect()
    sf = paramiko.SFTPClient.from_transport(t)
    sf.put(f"{ROOT}/code/alphazerobeta/azb_train.py", "C:/Users/vikk/kvab/azb/azb_train.py")
    sf.close()
    t.close()

    common = ('cd /d C:\\Users\\vikk\\kvab\\azb && set "PYTHONPATH=C:\\Users\\vikk\\kvab\\libs" && '
              'set "KVAB_ROOT=C:\\Users\\vikk\\kvab" && set PYTHONUNBUFFERED=1 && '
              "C:\\Users\\vikk\\AppData\\Local\\Programs\\Python\\Python313\\python.exe -u azb_train.py ")

    # main arm: seed 42 on every fold
    subprocess.run([sys.executable, f"{ROOT}/code/data/gpu.py", "run",
                    common + f'--tag azb --folds all --seeds 42 --steps {STEPS} --topn {TOP_N} '
                             "> C:\\Users\\vikk\\kvab\\azb.log 2>&1"], check=True)
    # seed-sensitivity arm: three seeds on a fixed fold subset
    subprocess.run([sys.executable, f"{ROOT}/code/data/gpu.py", "run",
                    common + f'--tag azbseed --folds 0,7,14 --seeds 42,123,456 --steps {STEPS} --topn {TOP_N} '
                             "> C:\\Users\\vikk\\kvab\\azbseed.log 2>&1"], check=True)

    t = connect()
    sf = paramiko.SFTPClient.from_transport(t)
    os.makedirs(f"{ROOT}/windows", exist_ok=True)
    os.makedirs(f"{ROOT}/windows_seed", exist_ok=True)
    for remote_dir, local_dir in [("C:/Users/vikk/kvab/windows_azb", f"{ROOT}/windows"),
                                  ("C:/Users/vikk/kvab/windows_azbseed", f"{ROOT}/windows_seed")]:
        try:
            for f in sf.listdir(remote_dir):
                if f.endswith(".parquet"):
                    sf.get(f"{remote_dir}/{f}", f"{local_dir}/{f}")
            print("pulled", remote_dir, "->", local_dir)
        except Exception as e:
            print("pull failed", remote_dir, e)
    for remote, local in [("C:/Users/vikk/kvab/experiment_log.jsonl", f"{ROOT}/experiment_log.jsonl")]:
        try:
            sf.get(remote, local)
            print("pulled", local)
        except Exception as e:
            print("pull failed", remote, e)
    sf.close()
    t.close()


if __name__ == "__main__":
    main()
