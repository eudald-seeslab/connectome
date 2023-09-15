import wandb
from main import main


sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'Test accuracy'
        },
    'parameters': {
        'connectome_layer_number': {'values': [1, 2, 3, 4, 5]},
     }
}
sweep_id = wandb.sweep(sweep_configuration, project="vgg_layer_number_20_runs_debug")


def train(config=None):
    # Note: I don't understand how the configs are passed around wandb; I'm
    #  following the tutorial here: https://github.com/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb
    with wandb.init(config=config):
        sweep_config = wandb.config
        main(sweep_config)


wandb.agent(sweep_id, train, count=5)
