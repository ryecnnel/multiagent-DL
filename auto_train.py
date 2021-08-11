import subprocess

base = "python c:/Users/yamashita/Documents/python_codes/multiagent-deep_learning/train_multiage.py"
option_a = "-O"

for option_b in ("-c -g c", "-c -g b", "-c -g ub", "-g ub"):
    cmd = base + " " + option_a + " " + option_b
    print(cmd)
    returncode = subprocess.call(cmd, shell=True)
