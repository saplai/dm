import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx

global numberOfNodes


def getFile():
    text = []
    with open("l4-2.txt", "r") as file:
        lines = file.readlines()
        for i in lines:
            text.append(i.replace('\n', ''))
    for i in range(len(text)):
        text[i] = text[i].split(" ")

    text = [[int(j) if '.' not in j else float(j) for j in i] for i in text]

    global numberOfNodes
    numberOfNodes = text[0][0]
    arrayNumbers = np.empty((0, numberOfNodes), int)
    for i in text[1:]:
        arrayNumbers = np.vstack([arrayNumbers, i])

    return arrayNumbers


class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.ROW = len(graph)

    def BFS(self, s, t, parent):
        visited = [False] * (self.ROW)
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:
            u = queue.pop(0)
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True

        return False

    def FordFulkerson(self, source, sink):
        parent = [-1] * (self.ROW)

        max_flow = 0

        while self.BFS(source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            print('Отримане значення - ', path_flow)
            max_flow += path_flow
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
        print('Алгоритм припиняє виконання через те, '
              'що шляхи витоку не існують. Відповіддю до поставленого '
              'завдання буде сума потоків всіх знайдених шляхів, що збільшуються.')
        return max_flow


if __name__ == '__main__':
    array = getFile()

    res = np.where(array != 0)
    oneArray = np.column_stack((res[0], res[1]))
    arrayDef = np.empty((0, 1), int)
    for i in oneArray:
        newRow = array[i[0]][i[1]]
        arrayDef = np.vstack([arrayDef, newRow])
    oneArray2 = np.column_stack((res[0], res[1], arrayDef))

    graph = nx.DiGraph()
    for i in oneArray2:
        graph.add_edge(i[0], i[1], weight=i[2])
    pos = nx.spring_layout(graph, seed=29)
    labels = nx.get_edge_attributes(graph, 'weight')
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)

    curved_edges = [edge for edge in graph.edges() if reversed(edge)
                    in graph.edges()]
    straight_edges = list(set(graph.edges()) - set(curved_edges))
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.3
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges,
                           connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(graph, 'weight')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {
        edge: edge_weights[edge] for edge in straight_edges}
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, label_pos=0.5, font_size=8, edge_labels=straight_edge_labels,
                                 rotate=False)
    plt.show()
    print(array)

    graph = Graph(array)
    print('Відповідь: ', graph.FordFulkerson(0, numberOfNodes - 1))
