import heapq
from typing import Dict, List

from entity.path import Path
from entity.station import Station
from geometry.utils import distance
from graph.node import Node


def build_station_nodes_dict(stations: List[Station], paths: List[Path]):
    station_nodes: List[Node] = []
    connections: List[List[Node]] = []
    station_nodes_dict: Dict[Station, Node] = {}

    for station in stations:
        node = Node(station)
        station_nodes.append(node)
        station_nodes_dict[station] = node
    for path in paths:
        if path.is_being_created:
            continue
        connection = []
        for station in path.stations:
            station_nodes_dict[station].paths.add(path)
            connection.append(station_nodes_dict[station])
        connections.append(connection)

    while len(station_nodes) > 0:
        root = station_nodes[0]
        for connection in connections:
            for idx in range(len(connection)):
                node = connection[idx]
                if node == root:
                    if idx - 1 >= 0:
                        neighbor = connection[idx - 1]
                        root.neighbors.add(neighbor)
                        d = distance(root.station.position, neighbor.station.position)
                        root.dist_to_neighbor[neighbor] = min(
                            root.dist_to_neighbor.get(neighbor, float("inf")), d
                        )
                    if idx + 1 <= len(connection) - 1:
                        neighbor = connection[idx + 1]
                        root.neighbors.add(neighbor)
                        d = distance(root.station.position, neighbor.station.position)
                        root.dist_to_neighbor[neighbor] = min(
                            root.dist_to_neighbor.get(neighbor, float("inf")), d
                        )
        station_nodes.remove(root)
        station_nodes_dict[root.station] = root

    return station_nodes_dict


def bfs(start: Node, end: Node) -> List[Node]:
    # Create a queue and enqueue the start node\
    queue = [(start, [start])]

    # While the queue is not empty
    while queue:
        # Dequeue the first node
        (node, path) = queue.pop(0)

        # If the node is the end node, return the path
        if node == end:
            return path

        # Enqueue the neighbors of the node
        for next in node.neighbors:
            if next not in path:
                queue.append((next, path + [next]))

    # If no path was found, return an empty list
    return []


def dijkstra(start: Node, end: Node) -> tuple[List[Node], float]:
    """물리적 거리 기반 최단 경로.

    반환: (노드 경로, 총 거리px).
    경로 없으면 ([], inf). start == end이면 ([start], 0).
    """
    if start == end:
        return [start], 0.0

    counter = 0
    heap: list = [(0.0, counter, start, [start])]
    visited: set = set()

    while heap:
        cost, _, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            return path, cost

        for neighbor in node.neighbors:
            if neighbor in visited:
                continue
            edge_dist = node.dist_to_neighbor.get(neighbor, float("inf"))
            if edge_dist == float("inf"):
                continue
            counter += 1
            heapq.heappush(
                heap,
                (cost + edge_dist, counter, neighbor, path + [neighbor]),
            )

    return [], float("inf")
