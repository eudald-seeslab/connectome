import re
import torch
import numpy as np
from tqdm import tqdm
from connectome import store_intermediate_output
from connectome import (
    clean_model_outputs,
    get_image_paths,
    get_iteration_number,
    initialize_results_df,
    select_random_images,
    update_results_df,
    update_running_loss,
)


def correct_test_results(test_results):
    # There was a bug in how we get the classes, and the 0-1 labels can be flipped
    # This function corrects the labels if the accuracy is below 0.5
    flipped = False
    if test_results["Is correct"].sum() / len(test_results) < 0.5:
        test_results["Is correct"] = np.abs(test_results["Is correct"] - 1)
        flipped = True

    return test_results, flipped


# test
def manifold_test(model, data_processor, criterion, device, u_config):
    batch_size = u_config.batch_size

    # The hook is in the decision making dropout layer because it is the last layer
    #  before the final layer, and the hook applies _after_ the layer you apply it to
    hook = model.decision_making_dropout.register_forward_hook(
        store_intermediate_output
    )

    testing_images = get_image_paths(u_config.TESTING_DATA_DIR, u_config.small_length)
    already_selected_testing = []
    total_correct, total, running_loss = 0, 0, 0.0
    test_results = initialize_results_df()
    all_intermediate_outputs = []
    all_labels = []

    model.eval()
    iterations = get_iteration_number(len(testing_images), u_config)
    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            batch_files, already_selected_testing = select_random_images(
                testing_images, batch_size, already_selected_testing
            )
            images, labels = data_processor.get_data_from_paths(batch_files)
            inputs, labels = data_processor.process_batch(images, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = model(inputs)
            all_intermediate_outputs.append(model.intermediate_output)
            all_labels.append(labels)
            loss = criterion(out, labels)

            outputs, predictions, labels_cpu, correct = clean_model_outputs(out, labels)
            test_results = update_results_df(
                test_results, batch_files, outputs, predictions, labels_cpu, correct
            )
            test_results, _ = correct_test_results(test_results)
            running_loss += update_running_loss(loss, inputs)
            total += batch_size
            total_correct += correct.sum()


    all_intermediate_outputs = torch.cat(all_intermediate_outputs, dim=0)
    hook.remove()

    print(
        f"Finished testing with loss {running_loss / total} and "
        f"accuracy {total_correct / total}."
    )
    return (
        test_results,
        None,
        total_correct / total,
        all_intermediate_outputs,
        all_labels,
    )


def extract_details(image_path):
    match = re.search(r"all_(\d+)_(\d+)_(\d+)_", image_path)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def reduce_dimension(df, intermediate, algorithm, n_dimensions):
    if algorithm == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_dimensions)

    elif algorithm == "umap":
        import umap

        reducer = umap.UMAP(n_components=n_dimensions)
    elif algorithm == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_dimensions)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    embedding = reducer.fit_transform(intermediate)

    for i in range(n_dimensions):
        df[f"{algorithm}_Component_{int(i + 1)}"] = embedding[:, i]

    return df
