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



def compute_degree_centrality(graph):
    """
    Computes the degree centrality for all nodes in a NetworkX graph.
    """
    n = len(graph)  # Total number of nodes
    return {node: graph.degree[node] / (n - 1) for node in graph.nodes}

from collections import deque

def compute_closeness_centrality(graph):
    """
    Computes closeness centrality for all nodes in a graph.
    """
    def bfs_shortest_path_lengths(start_node):
        """Compute shortest path lengths from a start node using BFS."""
        visited = {start_node: 0}  # Store distances from start_node
        queue = deque([start_node])  # Queue for BFS

        while queue:
            current = queue.popleft()
            current_distance = visited[current]

            for neighbor in graph.neighbors(current):  # Use graph.neighbors() for NetworkX graphs
                if neighbor not in visited:
                    visited[neighbor] = current_distance + 1
                    queue.append(neighbor)

        return visited  # Contains shortest path lengths from start_node to reachable nodes

    centrality = {}
    num_nodes = len(graph)  # Total number of nodes in the graph

    for node in graph.nodes:
        shortest_paths = bfs_shortest_path_lengths(node)
        reachable_nodes = len(shortest_paths) - 1  # Exclude the node itself

        if reachable_nodes > 0:
            # Sum of shortest path distances to all other reachable nodes
            total_distance = sum(shortest_paths.values())
            # Closeness centrality formula
            centrality[node] = (reachable_nodes) / total_distance
        else:
            # If the node is isolated, centrality is 0
            centrality[node] = 0.0

    return centrality


from collections import deque, defaultdict

def bfs_paths_and_counts(graph, start_node):
    """
    Perform BFS to calculate shortest paths, path counts, and predecessors for a given start node.
    """
    distances = {node: float('inf') for node in graph.nodes}  # Initialize distances to infinity
    paths = {node: 0 for node in graph.nodes}  # Initialize path counts to 0
    predecessors = {node: [] for node in graph.nodes}  # Initialize predecessors as empty lists

    # BFS initialization
    distances[start_node] = 0  # Distance to itself is 0
    paths[start_node] = 1  # One path to itself
    queue = deque([start_node])  # Start BFS from the start_node

    while queue:
        current_node = queue.popleft()
        current_distance = distances[current_node]

        for neighbor in graph.neighbors(current_node):
            # If visiting the neighbor for the first time
            if distances[neighbor] == float('inf'):
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

            # If this is part of a shortest path
            if distances[neighbor] == current_distance + 1:
                paths[neighbor] += paths[current_node]
                predecessors[neighbor].append(current_node)

    return distances, paths, predecessors


def compute_betweenness_centrality(graph):
    """
    Computes the betweenness centrality for all nodes in a graph.
    """
    centrality = {node: 0.0 for node in graph}
    for source in graph:
        shortest_paths, path_counts, predecessors = bfs_paths_and_counts(graph, source)
        dependency = {node: 0.0 for node in graph}
        nodes_by_distance = sorted(shortest_paths.keys(), key=lambda x: -shortest_paths[x])

        for node in nodes_by_distance:
            for predecessor in predecessors[node]:
                ratio = path_counts[predecessor] / path_counts[node]
                dependency[predecessor] += ratio * (1 + dependency[node])
            if node != source:
                centrality[node] += dependency[node]

    # Normalize
    num_nodes = len(graph)
    normalization_factor = (num_nodes - 1) * (num_nodes - 2)
    for node in centrality:
        centrality[node] /= normalization_factor if normalization_factor > 0 else 1

    return centrality

def compute_pagerank(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    """
    Computes the PageRank of nodes in a graph.
    """
    # Initialize variables
    num_nodes = len(graph)
    pagerank = {node: 1 / num_nodes for node in graph}  # Initial uniform rank
    new_pagerank = pagerank.copy()

    for iteration in range(max_iterations):
        for node in graph:
            rank_sum = 0
            for neighbor in graph.predecessors(node):  # Nodes pointing to the current node
                rank_sum += pagerank[neighbor] / graph.out_degree(neighbor)

            # PageRank formula with damping factor
            new_pagerank[node] = (1 - damping_factor) / num_nodes + damping_factor * rank_sum

        # Check for convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in graph)
        if diff < tolerance:
            break

        pagerank = new_pagerank.copy()  # Update for the next iteration

    return pagerank


def compute_single_source_dijkstra(graph, source, target):
    """
    Computes the shortest path using Dijkstra's algorithm for a graph.
    """
    import heapq

    # Priority queue to select the next node with the smallest distance
    priority_queue = [(0, source, [])]
    visited = set()

    while priority_queue:
        current_distance, current_node, path = heapq.heappop(priority_queue)

        # Skip if the node was already visited
        if current_node in visited:
            continue
        visited.add(current_node)

        # Update the path
        path = path + [current_node]

        # If we reach the target, return the distance and path
        if current_node == target:
            return current_distance, path

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                edge_weight = graph[current_node][neighbor].get('weight', 1)  # Default weight is 1 if not provided
                heapq.heappush(priority_queue, (current_distance + edge_weight, neighbor, path))

    # If no path is found
    return float('inf'), []

