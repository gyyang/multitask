import argparse
import train

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nunit', help="number of recurrent units", required=True, type=int)
parser.add_argument('-s', '--saveaddontype', help="save addon type", required=True, type=int)

a = parser.parse_args()

s = a.saveaddontype
if s == 0:
    save_addon_type = 'allrule_nonoise'
elif s == 1:
    save_addon_type = 'allrule_weaknoise'
elif s == 2:
    save_addon_type = 'allrule_strongnoise'
elif s == 3:
    save_addon_type = 'attendonly_nonoise'
elif s == 4:
    save_addon_type = 'attendonly_weaknoise'
elif s == 5:
    save_addon_type = 'attendonly_strongnoise'

train.train(a.nunit, save_addon_type)
