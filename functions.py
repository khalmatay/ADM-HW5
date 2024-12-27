def minimum_edge_cut(graph):
    """
    Removes the minimum number of edges to disconnect the graph into two subgraphs.

    Args:
        graph (networkx.Graph): The flight network graph.

    Returns:
        list: The edges removed to disconnect the graph.
        networkx.Graph: The modified graph with the edges removed.
    """
    # Ensure the graph is connected before proceeding
    if not nx.is_connected(graph):
        print("The graph is already disconnected.")
        return [], graph

    min_cut_edges = []
    original_edges = list(graph.edges)

    # Iterate through subsets of edges
    for edge in original_edges:
        # Create a copy of the graph to test disconnection
        temp_graph = graph.copy()
        temp_graph.remove_edge(*edge)

        # Check if removing the edge disconnects the graph
        if not nx.is_connected(temp_graph):
            min_cut_edges.append(edge)
            break  # Stop as soon as the minimum edge cut is found

    # Remove the edges in the cut set
    graph.remove_edges_from(min_cut_edges)

    return min_cut_edges, graph


def visualize_graph(graph, title):
    """
    Visualizes the graph.

    Args:
        graph (networkx.Graph): The graph to visualize.
        title (str): Title for the plot.
    """
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='black'
    )
    plt.title(title)
    plt.show()
