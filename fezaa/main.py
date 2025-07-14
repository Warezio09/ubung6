# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from ev import power_method

if __name__ == '__main__':
    print("Hello World!)
