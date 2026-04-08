# app.py

import subprocess
import sys

def main():
    subprocess.run([sys.executable, "inference.py"])