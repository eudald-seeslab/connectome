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
sweep_id = wandb.sweep(sweep_configuration, project="cnn_1_layer_number")
def train(config=None):
    with wandb.init(config=config):
        sweep_config = wandb.config
        main(sweep_config)

wandb.agent(sweep_id, train, count=5)
