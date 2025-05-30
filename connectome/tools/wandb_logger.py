import wandb
from wandb import AlertLevel
from connectome.core.utils import module_to_clean_dict


class WandBLogger:

    def __init__(self, project_name, enabled=True, imgs_every=500):
        self.project_name = project_name
        self.enabled = enabled
        self.log_images_every = imgs_every
        self.initialized = False

    @property
    def run_id(self):
        try:
            return wandb.run.id
        except AttributeError:
            return "NO_RUN_ID"

    def initialize_run(self, config_):
        if self.enabled and not self.initialized:
            model_config = module_to_clean_dict(config_)
            wandb.init(
                project=self.project_name,
                config=model_config,
                group=config_.wandb_group,
            )
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

    def log_metrics(self, epoch, running_loss, total_correct, total, task=None):

        char_ = task if task is not None else ""
        if self.enabled:
            try:
                wandb.log(
                    {
                        "epoch": epoch,
                        f"loss {char_}": running_loss / total,
                        f"accuracy {char_}": total_correct / total,
                    }
                )
            except Exception as e:
                print(f"Error logging running stats to wandb: {e}. Continuing...")

    def log_image(self, vals, name, title, task=None):
        char_ = task if task is not None else ""
        if self.enabled:
            wandb.log(
                {
                    f"{title} image {char_}": wandb.Image(
                        vals, caption=f"{title} image {name}"
                    ),
                }
            )

    def log_dataframe(self, df, title):
        if self.enabled:
            wandb.log({title: wandb.Table(dataframe=df)})

    def log_validation_stats(
        self, running_loss_, total_correct_, total_, results_, plots, task=None
    ):
        char_ = task if task is not None else ""
        if len(plots) > 0:
            plot_dict = {f"Plot {i} {char_}": wandb.Image(plot) for i, plot in enumerate(plots)}
        else:
            plot_dict = {}
        if self.enabled:
            wandb.log(
                {
                    f"Validation loss {char_}": running_loss_ / total_,
                    f"Validation accuracy {char_}": total_correct_ / total_,
                    f"Validation results {char_}": wandb.Table(dataframe=results_),
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
