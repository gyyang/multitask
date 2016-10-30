import argparse
import os

def main(HDIM=200, N_RING=16, new_compiledir=True, gpus=0):
    import train # have to do this after the theano configuration
    # Debug now
    train.train(HDIM=HDIM, N_RING=N_RING)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nunit', help="number of recurrent units", required=True, type=int)
a = parser.parse_args()

import train
main(HDIM=a.nunit)