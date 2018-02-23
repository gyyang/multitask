# MultiTask Network

## Reproducing results from the paper
All analysis results from the paper can be reproduced from paper.py

Simply go to paper.py, set the model_dir to be the directory of your 
model files, uncomment the analyses you want to run, and run the file.

## Pretrained models
We provide 20 pretrained models and their auxillary data files for
analyses.

## Get started with training
Train a default network with:

    import train
    train.train(train_dir='debug', hparams={'learning_rate': 0.001}, ruleset='mante')

These lines will train a default network for the Mante task, and store the
results in your_working_directory/debug/

## Get started with some simple analyses
After training (you can interrupt at any time), you can visualize the neural
activity using

    import standard_analysis
    model_dir = 'debug'
    rule = 'contextdm1'
    standard_analysis.easy_activity_plot(model_dir, rule)

This will plot some neural activity. See the source code to know how to load
hyperparameters, restore model, and run it for analysis.