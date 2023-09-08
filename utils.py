# check missing values
import torch
import wandb
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os


def check_for_missing_values(x, step):
    if torch.isnan(x.parameters()).any():
        raise Exception(f"NaN in parameters at step {step}")


def nan_to_unknown(x):
    # Replace the string "nan" with "Unknown"
    return x.replace("nan", "Unknown")


def check_model_parameters(model, epoch):
    # NOT IN USE AT THE MOMENT
    check_for_missing_values(model, epoch)
    # Clip gradients to avoid exploding gradients
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    # Clip parameters to avoid exploding parameters
    for p in model.parameters():
        p.data.clamp_(-1, 1)


def log_training_images(images, labels, outputs):
    num_images_to_log = 5
    for i in range(num_images_to_log):
        # Convert image and label to numpy for visualization
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        label = labels[i].cpu().numpy()
        predicted_label = torch.argmax(outputs[i]).cpu().numpy()

        # There is a bug in wandb that prevents logging images
        try:
            wandb.log(
                {
                    "Training Image": wandb.Image(
                        image,
                        caption=f"True Label: {label}, Predicted Label: {predicted_label}",
                    )
                }
            )
        except FileNotFoundError:
            logger = logging.getLogger("training_log")
            logger.warning(f"Could not log training image")


def plot_weber_fraction(test_results_df, save_dir):
    # Calculate the percentage of correct answers for each Weber ratio
    test_results_df['yellow'] = test_results_df['Image'].apply(lambda x: x.split('_')[1])
    test_results_df['blue'] = test_results_df['Image'].apply(lambda x: x.split('_')[2])
    test_results_df['weber_ratio'] = test_results_df.apply(
        lambda row: max(int(row['yellow']), int(row['blue'])) / min(int(row['yellow']), int(row['blue'])), axis=1)
    correct_percentage = test_results_df.groupby('weber_ratio')['Correct Prediction'].mean() * 100
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correct_percentage.index, y=correct_percentage.values)
    plt.xlabel('Weber Ratio')
    plt.ylabel('Percentage of Correct Answers')
    plt.title('Percentage of Correct Answers for Each Weber Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save to model directory
    plt.savefig(os.path.join(save_dir, 'weber_fraction_plot.png'))


def print_run_details(config_manager, debug, images_fraction, continue_training):
    logger = logging.getLogger("training_log")
    config_manager.output_model_details()
    if debug:
        logger.warning("WARNING: Running on DEBUG mode, so using 10% of the images")
    elif images_fraction < 1:
        logger.warning(f"WARNING: Using {images_fraction * 100}% of the images")
    if continue_training:
        logger.warning("Warning: I'm training and already trained model")


def handle_log_configs(debug):

    logging.basicConfig(
        filename="training_log.log",
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("training_log")
    logger.addHandler(logging.StreamHandler())

    return logger
