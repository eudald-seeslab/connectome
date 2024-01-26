import numpy as np
from scipy.spatial import cKDTree

def get_hex_coordinates(values):
    """ Generate hexagonal coordinates for the given values. """
    # Placeholder function as the actual implementation is not provided
    # This will create a hexagonal grid of points
    n = len(values)
    radius = int(np.ceil(np.sqrt(n)))
    u, v = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    u = u.flatten()
    v = v.flatten()
    mask = np.abs(u - v) <= radius
    return u[mask], v[mask]

def interpolate_between(coord_to_value, p1, p2):
    """ Interpolate between two points. """
    mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    mid_value = (coord_to_value.get(p1, 0) + coord_to_value.get(p2, 0)) / 2
    return mid_point, mid_value

def interpolate_hexagonal(u, v, values, num_points):
    """ Interpolate points in a hexagonal lattice. """
    # Create a mapping from coordinates to values
    coord_to_value = {(u[i], v[i]): values[i] for i in range(len(values))}

    # Define hexagonal neighbors (relative positions)
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    # Loop until we have enough points
    while len(coord_to_value) < num_points:
        new_points = []
        new_values = []

        for (u_coord, v_coord), value in list(coord_to_value.items()):
            for offset in neighbor_offsets:
                neighbor = (u_coord + offset[0], v_coord + offset[1])

                # Check if neighbor is a new point
                if neighbor not in coord_to_value:
                    # Here, instead of averaging with an existing neighbor, we simply replicate the value
                    new_points.append(neighbor)
                    new_values.append(value)

        # Update coord_to_value with newly added points
        coord_to_value.update({pt: val for pt, val in zip(new_points, new_values)})

    # Split the dictionary back into separate arrays
    extended_u, extended_v = zip(*coord_to_value.keys())
    extended_values = list(coord_to_value.values())

    return np.array(extended_u), np.array(extended_v), np.array(extended_values)

def get_voronoi_averages(vector_of_values, n_centers=255):
    """Get the Voronoi averages."""

    u, v = get_hex_coordinates(vector_of_values)

    # Make sure we have enough points to compute the Voronoi averages
    min_num_points = n_centers * 2

    # Extend vector_of_values to match n_centers, if needed
    while len(vector_of_values) < min_num_points:
        u, v, vector_of_values = interpolate_hexagonal(u, v, vector_of_values, min_num_points)

    # Combine u and v to create coordinate pairs and select n random points
    coords = np.column_stack((u, v))
    rand_indices = np.random.choice(len(coords), n_centers, replace=False)
    rand_points = coords[rand_indices]

    # Create a KD-tree for fast lookup of nearest Voronoi cell
    tree = cKDTree(rand_points)

    # Find the nearest Voronoi cell for each point in the lattice
    _, indices = tree.query(coords)

    # Compute the average value for each Voronoi cell
    # Adjust the range of i to match the final length of vector_of_values
    averages = []
    for i in range(n_centers):
        cell_values = vector_of_values[indices == i]
        if len(cell_values) > 0:
            averages.append(cell_values.mean())
        else:
            averages.append(np.nan)  # Handle cells with no points

    return np.array(averages)


if __name__ == "__main__":
    # Generate random values and calculate Voronoi averages
    vector_of_values = np.random.rand(721)
    n_centers = 1200
    avgs = get_voronoi_averages(vector_of_values, n_centers=n_centers)
    nan_count = np.count_nonzero(np.isnan(avgs))
    print(len(avgs))
    print(f"Number of NaNs: {nan_count}")
