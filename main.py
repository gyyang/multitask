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
elif s == 6:
    save_addon_type = 'choiceonly_nonoise'
elif s == 7:
    save_addon_type = 'choiceonly_weaknoise'
elif s == 8:
    save_addon_type = 'choiceonly_strongnoise'
elif s == 9:
    save_addon_type = 'delaychoiceonly_nonoise'
elif s == 10:
    save_addon_type = 'delaychoiceonly_weaknoise'
elif s == 11:
    save_addon_type = 'delaychoiceonly_strongnoise'

train.train(a.nunit, save_addon_type)
