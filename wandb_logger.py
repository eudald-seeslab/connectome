import wandb
from wandb import AlertLevel
import config


MODEL_CONFIG = {
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "base_lr": config.base_lr,
    "num_connectome_passes": config.NUM_CONNECTOME_PASSES,
    "log_transform_weights": config.log_transform_weights,
    "training_data": config.TRAINING_DATA_DIR,
    "testing_data": config.TESTING_DATA_DIR,
    "neurons": config.neurons,
    "voronoi_criteria": config.voronoi_criteria,
    "random_synapses": config.random_synapses,
    "small": config.small,
    "small_length": config.small_length,
}


class WandBLogger:
    def __init__(self, project_name):
        self.project_name = project_name
        self.model_config = MODEL_CONFIG
        self.enabled = config.wandb_
        self.log_images_every = config.wandb_images_every
        self.initialized = False

    def initialize_run(self):
        if self.enabled and not self.initialized:
            wandb.init(project=self.project_name, config=self.model_config)
            self.initialized = True
    
    def initialize_sweep(self, sweep_config):
        if self.enabled:
            return wandb.sweep(sweep_config, project=self.project_name)
        
    def start_agent(self, sweep_id, func):
        if self.enabled:
            wandb.agent(sweep_id, function=func)
    
    @property
    def sweep_config(self):
        return wandb.config

    def log_metrics(self, epoch, iteration, running_loss, total_correct, total):
        if self.enabled:
            try:
                wandb.log(
                    {
                        "epoch": epoch,
                        "iteration": iteration + 1,
                        "loss": running_loss / total,
                        "accuracy": total_correct / total,
                    }
                )
            except Exception as e:
                print(f"Error logging running stats to wandb: {e}. Continuing...")

    def log_image(self, vals, name, title):
        if self.enabled:
            wandb.log(
                {
                    f"{title} image": wandb.Image(
                        vals, caption=f"{title} image {name}"
                    ),
                }
            )

    def log_dataframe(self, df, title):
        if self.enabled:
            wandb.log({title: wandb.Table(dataframe=df)})

    def log_validation_stats(
        self, running_loss_, total_correct_, total_, results_, plots
    ):
        if len(plots) > 0:
            plot_dict = {f"Plot {i}": wandb.Image(plot) for i, plot in enumerate(plots)}
        else:
            plot_dict = {}
        if self.enabled:
            wandb.log(
                {
                    "Validation loss": running_loss_ / total_,
                    "Validation accuracy": total_correct_ / total_,
                    "Validation results": wandb.Table(dataframe=results_),
                }
                | plot_dict
            )

    def send_crash(self, message):
        if self.enabled:
            wandb.alert(
                title=f"Error in run at {self.project_name}",
                text=message,
                level=AlertLevel.ERROR,
            )

    def finish(self):
        if self.enabled:
            wandb.finish()
