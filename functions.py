import matplotlib.pyplot as plt
import networkx as nx


def minimum_edge_cut(graph):
    if not nx.is_connected(graph):
        print("The graph is already disconnected.")
        return [], list(nx.connected_components(graph))

    # Convert to directed graph with uniform capacities
    directed_graph = nx.DiGraph()
    for u, v in graph.edges:
        directed_graph.add_edge(u, v, capacity=1)
        directed_graph.add_edge(v, u, capacity=1)

    # Use NetworkX's built-in minimum_cut function
    source = list(directed_graph.nodes)[0]  # Select an arbitrary source node
    sink = list(directed_graph.nodes)[-1]  # Select an arbitrary sink node

    cut_value, partition = nx.minimum_cut(directed_graph, source, sink, capacity="capacity")
    reachable, non_reachable = partition

    # Find the edges in the minimum cut
    cut_edges = [(u, v) for u in reachable for v in directed_graph.neighbors(u) if v in non_reachable]

    # Remove the edges in the cut set from the original graph
    graph.remove_edges_from(cut_edges)

    # Get the resulting connected components as subgraphs
    subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    return cut_edges, subgraphs


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


def visualize_subgraphs(subgraphs, title_prefix):
    """
    Visualizes each subgraph in a list of subgraphs.

    Args:
        subgraphs (list): List of subgraphs to visualize.
        title_prefix (str): Prefix for the plot titles.
    """
    for i, subgraph in enumerate(subgraphs):
        title = f"{title_prefix} - Subgraph {i + 1}"
        visualize_graph(subgraph, title)
