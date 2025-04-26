import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os


def panel_a():
    # Create figure
    fig1, axs = plt.subplots(2, 2, figsize=(7.5, 6), constrained_layout=True)
    axs = axs.flatten()

    # Set a seed for reproducibility
    np.random.seed(42)

    # Generate 3D coordinates for neurons - fixed to exactly n_nodes
    n_nodes = 40
    nodes_2d = np.zeros((n_nodes, 2))

    # Create clusters of nodes
    cluster_centers = [
        [0, 0],  # center
        [-1.5, 1],  # top left
        [1.5, 1],  # top right
        [-1.5, -1],  # bottom left
        [1.5, -1],  # bottom right
    ]

    # Assign nodes to clusters with jitter
    nodes_per_cluster = n_nodes // len(cluster_centers)
    for i, center in enumerate(cluster_centers):
        start_idx = i * nodes_per_cluster
        end_idx = start_idx + nodes_per_cluster if i < len(cluster_centers) - 1 else n_nodes

        for j in range(start_idx, end_idx):
            # Add position with random jitter
            nodes_2d[j, 0] = center[0] + np.random.normal(0, 0.5)
            nodes_2d[j, 1] = center[1] + np.random.normal(0, 0.5)

    # Create fake 3D positions (only needed for distance calculations)
    nodes_3d = np.column_stack((nodes_2d, np.zeros(n_nodes)))

    # Transform 3D to 2D coordinates for visualization
    nodes_2d = nodes_3d[:, :2]

    # Create a distance matrix
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_matrix[i, j] = np.sqrt(np.sum((nodes_3d[i] - nodes_3d[j]) ** 2))

    # Normalize distances
    max_dist = np.max(dist_matrix)
    norm_dist_matrix = dist_matrix / max_dist

    # Define edge densities and colors with explicit wiring lengths
    configs = [
        {
            "title": "Biological Connectome",
            "density": 0.1,
            "node_color": "#4477AA",
            "edge_color": "#77AADD",
            "rel_length": 1.0,
            "prefers_short": True,
        },
        {
            "title": "Unconstrained Random",
            "density": 0.15,
            "node_color": "#EE6677",
            "edge_color": "#EE99AA",
            "rel_length": 2.3,
            "prefers_short": False,
        },
        {
            "title": "Pruned Random",
            "density": 0.1,
            "node_color": "#228833",
            "edge_color": "#66BB66",
            "rel_length": 1.0,
            "prefers_short": False,
        },
        {
            "title": "Distance-Binned Random",
            "density": 0.1,
            "node_color": "#CCBB44",
            "edge_color": "#DDCC77",
            "rel_length": 1.0,
            "prefers_short": True,
        },
    ]

    # Generate the graphs for each configuration
    for idx, config in enumerate(configs):
        # Create a new graph
        G = nx.DiGraph()

        # Add nodes
        for i in range(n_nodes):
            G.add_node(i, pos=nodes_2d[i])

        # Create edges based on configuration
        edges_to_add = []

        if config["title"] == "Biological Connectome":
            # Much stronger preference for short connections
            distance_threshold = 0.6  # Only allow connections below this normalized distance
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and norm_dist_matrix[i, j] < distance_threshold:
                        # Higher probability for shorter connections
                        p_connect = (1 - norm_dist_matrix[i, j]/distance_threshold)**2 * config["density"] * 15
                        if np.random.random() < p_connect:
                            weight = 1.0  # Consistent weight
                            edges_to_add.append((i, j, {'weight': weight, 'distance': dist_matrix[i, j]}))
        
        elif config["title"] == "Unconstrained Random":
            # Random connections regardless of distance
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and np.random.random() < config["density"]:
                        weight = np.random.uniform(0.5, 1)
                        edges_to_add.append(
                            (i, j, {"weight": weight, "distance": dist_matrix[i, j]})
                        )

        elif config["title"] == "Pruned Random":
            # Random connections regardless of distance
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and np.random.random() < config["density"]:
                        weight = np.random.uniform(0.2, 0.7)
                        edges_to_add.append(
                            (i, j, {"weight": weight, "distance": dist_matrix[i, j]})
                        )

        elif config["title"] == "Distance-Binned Random":
            # Divide distances into bins and randomize within each bin
            dist_bins = [
                0,
                0.3,
                0.6,
                1.0,
            ]  # Three distance bins with emphasis on shorter connections
            bin_densities = [0.3, 0.1, 0.05]  # Higher density for shorter connections

            for bin_idx in range(len(dist_bins) - 1):
                bin_min, bin_max = dist_bins[bin_idx], dist_bins[bin_idx + 1]
                bin_density = bin_densities[bin_idx]

                # Find node pairs in this distance bin
                bin_pairs = []
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j and bin_min <= norm_dist_matrix[i, j] < bin_max:
                            bin_pairs.append((i, j))

                # Randomly select pairs from this bin
                np.random.shuffle(bin_pairs)
                bin_edges_count = int(len(bin_pairs) * bin_density)

                for i, j in bin_pairs[:bin_edges_count]:
                    weight = np.random.uniform(0.5, 1.0)
                    edges_to_add.append(
                        (i, j, {"weight": weight, "distance": dist_matrix[i, j]})
                    )

        # Add the edges to the graph
        for u, v, data in edges_to_add:
            G.add_edge(u, v, **data)

        # Get positions and draw
        pos = nx.get_node_attributes(G, "pos")

        # Calculate node sizes based on degree
        node_sizes = [20 + 3 * G.degree(n) for n in G.nodes()]
       
        # Draw the nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=axs[idx],
            node_size=35,
            node_color=config["node_color"],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

        # Draw edges with color and width encoding distance
        # Short connections: thicker and more opaque
        # Long connections: thinner and more transparent
        for u, v, data in G.edges(data=True):
            distance = data["distance"] / max_dist

            # Determine edge properties based on distance
            if config["prefers_short"]:
                # For configurations that prefer short connections
                width = max(0.5, 2.0 * (1.0 - distance))
                alpha = max(0.2, 0.8 * (1.0 - distance))
            else:
                # For configurations that don't differentiate by distance
                width = 1.0
                alpha = 0.6

            nx.draw_networkx_edges(
                G,
                pos,
                ax=axs[idx],
                edgelist=[(u, v)],
                width=width * 2 if config["title"] == "Unconstrained Random" else width,
                alpha=alpha,
                edge_color=config["edge_color"],
                arrows=True,
                arrowsize=6,
                arrowstyle="->",
                connectionstyle="arc3,rad=0.1",
            )

        # Add title and wiring info
        axs[idx].set_title(config["title"], fontsize=16, fontweight="bold")
        axs[idx].text(
            0.,
            0.,
            f"Wiring: {config['rel_length']:.1f}x",
            transform=axs[idx].transAxes,
            fontsize=14,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        # Remove axes
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        axs[idx].spines["top"].set_visible(False)
        axs[idx].spines["right"].set_visible(False)
        axs[idx].spines["bottom"].set_visible(False)
        axs[idx].spines["left"].set_visible(False)

        # plt.suptitle("Network Configuration Comparison", fontsize=10, fontweight="bold")
        plt.savefig(os.path.join(plots_dir, "figure2a_network_configs.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(plots_dir, "figure2a_network_configs.pdf"), bbox_inches="tight")


    return fig1
