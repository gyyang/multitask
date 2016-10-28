import argparse
import os

def main(HDIM=200, N_RING=16, new_compiledir=True, gpus=0):
    # Set Theano configuration
    os.environ.setdefault('THEANO_FLAGS', '')
    if new_compiledir:
        compiledir = '/scratch/gy441/MultiTask/theano/compileddir'+str(HDIM)+'_'+str(N_RING)
        os.environ['THEANO_FLAGS'] += ',base_compiledir=' + compiledir
    os.environ['THEANO_FLAGS'] += ',floatX=float32,allow_gc=False'
    if gpus > 0:
        os.environ['THEANO_FLAGS'] += ',device=gpu,nvcc.fastmath=True'

    print(os.environ['THEANO_FLAGS'])

    import train # have to do this after the theano configuration
    # Debug now
    train.train(HDIM=HDIM, N_RING=N_RING)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nunit', help="number of recurrent units", required=True, type=int)
a = parser.parse_args()

main(HDIM=a.nunit)