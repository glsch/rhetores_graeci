import os
import sys

# set the current working directory to the root of the repository
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
