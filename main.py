"""
The main function, which will invoke the run function in the eidolon.deamon package.
The run fucntion receives an argv param, which is necessary.
In version 1.7, all the eidolon applications should specify an unique main function.
@since 2019.12.23
@author yuwei
"""

#inner lib
from eidolon import daemon

#system lib
import sys

if __name__ == "__main__":
    # start running
    daemon.run(sys.argv)