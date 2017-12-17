"""
Main file for generating results in the paper:
Clustering and compositionality of task representations
in a neural network trained to perform many cognitive tasks
Yang GR et al. 2017 BioRxiv
"""
from __future__ import absolute_import
import tools


def main_train(nunit, seed, incomplete_set=False):
    '''Training setup for paper.

    Args:
        nunit: int, number of units.
        seed: int, seed.
        incomplete_set : if True,
            do not train 'delayanti' and 'contextdelaydm1'
    '''
    from train import train
    save_name = '{:d}_{:d}paper'.format(seed, nunit)

    if incomplete_set:
        save_name = '{:d}_{:d}incom'.format(seed, nunit)

    ruleset = 'all'

    rule_prob_map = dict()
    # increase probability for contextdm1 & 2
    rule_prob_map['contextdm1'] = 5.
    rule_prob_map['contextdm2'] = 5.

    from task import rules_dict
    rule_trains = [rules_dict[ruleset]]

    if incomplete_set:
        # overwrite rule_trains
        rule_excludes = ['delayanti', 'contextdelaydm1']
        rule_trains = [
                [r for r in rules_dict[ruleset] if r not in rule_excludes]]

    train(save_name,
          ruleset=ruleset,
          n_hidden=nunit,
          learning_rate=0.001,
          target_perf=None,
          seed=seed,
          activation='softplus',
          rnn_type='LeakyRNN',
          l1_h=0.0001,
          training_iters=150000,
          rule_trains=rule_trains,
          rule_prob_map=rule_prob_map,
          run_analysis=['var', 'taskset', 'psy'])


def cont_train(c, ksi, seed, save_name, seq=True):
    '''Sequantial training setup for paper

    Args:
        c, ksi : parameters for continual learning
        seed: int, random seed.
        save_name: string, name of file to be saved.
        seq: bool, sequential or not.
    '''
    from train import train
    param_intsyn = (c, ksi)

    nunit = 256
    ruleset = 'all'

    if seq:
        # Sequential training
        rule_trains = [
                'fdgo', 'delaygo', 'dm1', 'dm2', ('contextdm1', 'contextdm2'),
                'multidm', 'delaydm1', 'delaydm2',
                ('contextdelaydm1', 'contextdelaydm2'), 'multidelaydm']
    else:
        rule_trains = [(
                'fdgo', 'delaygo', 'dm1', 'dm2', 'contextdm1', 'contextdm2',
                'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1',
                'contextdelaydm2', 'multidelaydm')]

    rule_tests = [
            'fdgo', 'delaygo', 'dm1', 'dm2', 'contextdm1', 'contextdm2',
            'multidm', 'delaydm1', 'delaydm2',
            'contextdelaydm1', 'contextdelaydm2', 'multidelaydm']

    learning_rate = 0.001  # learning is much better with smaller learning rate
    activation = 'relu'  # softplus has lots of difficulty learning context-dms
    easy_task = True   # Network has difficulty learning harder version

    train(save_name,
          ruleset=ruleset,
          n_hidden=nunit,
          learning_rate=learning_rate,
          target_perf=None,
          seed=seed,
          activation=activation,
          rnn_type='LeakyRNN',
          l1_h=0.0001,
          training_iters=100000,
          display_step=500,
          param_intsyn=param_intsyn,
          rule_trains=rule_trains,
          rule_tests=rule_tests,
          run_analysis=['var'],
          easy_task=easy_task)

if __name__ == '__main__':

    # Performance Analysis-----------------------------------------------------
    # Name of the model
    save_name = '0_256paper'

    import performance
    import standard_analysis

    # standard_analysis.schematic_plot(save_name) # Generate schematic
    # performance.plot_performanceprogress(save_name) # Plot performance during training
    # performance.psychometric_choice(save_name) # Psychometric for dm
    # performance.psychometric_choiceattend(save_name, no_ylabel=True)
    # performance.psychometric_choiceint(save_name, no_ylabel=True)

    save_names = tools.valid_save_names('*256paper')
    # for save_name in save_names[3:4]:
    #     for rule in ['dm1', 'contextdm1', 'multidm']:
    #         # performance.compute_choicefamily_varytime(save_name, rule)
    #         performance.plot_choicefamily_varytime(save_name, rule)
    # performance.psychometric_delaychoice_varytime(save_name, 'delaydm1')

    # Analysis of Anti tasks---------------------------------------------------
    import antitask
    # ATA = antitask.Analysis(save_name)
    # ATA.plot_example_unit()
    # ATA.plot_lesions()
    # ATA.plot_inout_connections()
    # ATA.plot_rec_connections()
    # ATA.plot_rule_connections()

    # Clustering Analysis------------------------------------------------------
    import clustering
    CA = clustering.Analysis(save_name, data_type='rule')
    # CA.plot_example_unit()
    # CA.plot_variance()
    CA.plot_2Dvisualization()
    # CA.plot_lesions()

    # import variance
    # variance.compute_variance(save_name)
    # CA = clustering.Analysis(save_name, data_type='epoch')
    # CA.plot_variance()
    # CA.plot_2Dvisualization()

    # FTV Analysis-------------------------------------------------------------
    save_pattern='*256paper'
    import variance
    # variance.plot_hist_varprop_selection(save_pattern)
    # variance.plot_hist_varprop_all(save_pattern)

    # Task Representation------------------------------------------------------
    import taskset
    # tsa = taskset.TaskSetAnalysis(save_name)
    # tsa.compute_and_plot_taskspace(
    #         epochs=['stim1'], dim_reduction_type='PCA')

    # Compositional Representation---------------------------------------------
    # setups = [1] # Go, Anti family
    # setups = [2] # Ctx DM family
    setups = [1, 2]
    for setup in setups:
        pass 
        # taskset.plot_taskspace_group(save_pattern, setup=setup,
        #                              restore=True, representation='rate')
        # taskset.plot_taskspace_group(save_pattern, setup=setup,
        #                              restore=True, representation='weight')
        # taskset.plot_replacerule_performance_group(
        #         save_pattern, setup=setup, restore=True)

    # Continual Learning Analysis----------------------------------------------
    # performance.get_allperformance('*cont')
    # performance.plot_performanceprogress_cont(('0_2cont', '0_0cont'))
    # performance.plot_finalperformance_cont('*2cont', '*0cont')
    # for save_pattern in ['*3cont']:
    #     variance.plot_hist_varprop_selection(save_pattern)
    # performance.plot_performanceprogress(save_name='0_3seqlowlr') # Plot performance during training
