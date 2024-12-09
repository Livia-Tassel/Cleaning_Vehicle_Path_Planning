def dynamic_A_star(graph, cleaner, target_node, k, t):
    def heuristic(node, target_node):
        x1, y1 = graph.node_positions[node]
        x2, y2 = graph.node_positions[target_node]
        return math.hypot(x2 - x1, y2 - y1)

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(cleaner.current_node, target_node), 0, cleaner.current_node))

    came_from = {}
    g_costs = {cleaner.current_node: 0}
    f_costs = {cleaner.current_node: heuristic(cleaner.current_node, target_node)}

    closed_set = set()

    while open_list:
        _, g, current_node = heapq.heappop(open_list)

        if current_node in closed_set:
            continue

        closed_set.add(current_node)

        if current_node == target_node:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(cleaner.current_node)
            return path[::-1]

        for neighbor in graph.graph.neighbors(current_node):
            edge_key = (min(current_node, neighbor), max(current_node, neighbor))
            distance = graph.graph[current_node][neighbor].get("weight", 1)

            dynamic_factor = k * D((current_node, neighbor), t)
            tentative_g = g + distance + dynamic_factor

            if neighbor in closed_set:
                continue

            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                came_from[neighbor] = current_node
                g_costs[neighbor] = tentative_g
                f_costs[neighbor] = tentative_g + heuristic(neighbor, target_node)
                heapq.heappush(open_list, (f_costs[neighbor], tentative_g, neighbor))

    return []  