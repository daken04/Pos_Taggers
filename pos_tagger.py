import sys
import subprocess

if len(sys.argv) == 2:
    arg = sys.argv[1]

    if arg == "-f":
        subprocess.run(["python", "FFNN.py"])
    elif arg == "-r":
        subprocess.run(["python", "RNN.py"])
    else:
        print(f"Unknown argument: {arg}")
else:
    print("Usage: python3 pos_tagger.py -f|-r")
