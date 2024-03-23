def tsp_to_atsp_tour(tsp_tour, atsp_matrix):
    """
    Transform a TSP tour to an ATSP tour.

    Args:
    - tsp_tour: List representing the TSP tour.
    - atsp_matrix: 2D list representing the ATSP matrix.

    Returns:
    - atsp_tour: List representing the ATSP tour.
    """
    atsp_tour = []
    num_cities = len(atsp_matrix)

    for i in range(len(tsp_tour) - 1):
        from_city = tsp_tour[i]
        to_city = tsp_tour[i + 1]

        forward_cost = atsp_matrix[from_city][to_city]
        reverse_cost = atsp_matrix[to_city][from_city]

        # Choose the direction with the minimum cost
        if forward_cost <= reverse_cost:
            atsp_tour.append(from_city)
        else:
            atsp_tour.append(to_city)

    # Add the last city back to the starting city
    from_city = tsp_tour[-1]
    to_city = tsp_tour[0]
    forward_cost = atsp_matrix[from_city][to_city]
    reverse_cost = atsp_matrix[to_city][from_city]

    if forward_cost <= reverse_cost:
        atsp_tour.append(from_city)
    else:
        atsp_tour.append(to_city)

    return atsp_tour

# Example usage:
tsp_tour = [0, 1, 2, 3, 0]  # TSP tour
atsp_matrix = [
    [0, 10, 15, 20],
    [5, 0, 35, 25],
    [10, 35, 0, 30],
    [15, 20, 25, 0]
]  # ATSP matrix

atsp_tour = tsp_to_atsp_tour(tsp_tour, atsp_matrix)
print("ATSP tour:", atsp_tour)
