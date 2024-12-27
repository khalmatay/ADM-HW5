import matplotlib.pyplot as plt
import networkx as nx


def minimum_cut(graph):
    """
    Finds the minimum edge cut of a graph using the Ford-Fulkerson method (max-flow/min-cut).
    """
    if not nx.is_connected(graph):
        raise ValueError("The graph must be connected.")

    # Convert to a directed graph with uniform capacities
    directed_graph = nx.DiGraph()
    for u, v in graph.edges():
        directed_graph.add_edge(u, v, capacity=1)
        directed_graph.add_edge(v, u, capacity=1)

    # Use the first and last nodes as source and sink
    source = list(directed_graph.nodes)[0]
    sink = list(directed_graph.nodes)[-1]

    #  The minimum cut using Ford-Fulkerson from scratch
    residual = {edge: directed_graph.edges[edge]['capacity'] for edge in directed_graph.edges()}
    max_flow = 0

    def bfs(source, sink, parent):
        visited = {node: False for node in directed_graph.nodes()}
        queue = [source]
        visited[source] = True

        while queue:
            u = queue.pop(0)

            for v in directed_graph.neighbors(u):
                if not visited[v] and residual.get((u, v), 0) > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
        return False

    parent = {}
    while bfs(source, sink, parent):
        # Find the maximum flow through the path found by BFS
        path_flow = float('Inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, residual[(parent[s], s)])
            s = parent[s]

        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            residual[(u, v)] -= path_flow
            residual[(v, u)] = residual.get((v, u), 0) + path_flow
            v = parent[v]

        max_flow += path_flow

    # Find reachable nodes in the residual graph
    visited = {node: False for node in directed_graph.nodes()}
    queue = [source]
    visited[source] = True
    while queue:
        u = queue.pop(0)
        for v in directed_graph.neighbors(u):
            if not visited[v] and residual.get((u, v), 0) > 0:
                queue.append(v)
                visited[v] = True

    # Find the edges that cross the cut
    reachable = [node for node in visited if visited[node]]
    non_reachable = [node for node in directed_graph.nodes() if not visited[node]]
    cut_edges = [(u, v) for u in reachable for v in directed_graph.neighbors(u) if v in non_reachable]

    return max_flow, cut_edges


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


def visualize_subgraphs(graph, cut_edges, title):
    """
    Visualizes the two resulting subgraphs after removing the cut edges.
    """
    graph_copy = graph.copy()
    graph_copy.remove_edges_from(cut_edges)
    components = list(nx.connected_components(graph_copy))

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))

    for component in components:
        subgraph = graph.subgraph(component)
        nx.draw(
            subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='black'
        )

    plt.title(title)
    plt.show()
