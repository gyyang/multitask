import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nunit', help="number of recurrent units", required=True, type=int)
a = parser.parse_args()

import train # have to do this after the theano configuration
# Debug now
train.train(a.nunit)
