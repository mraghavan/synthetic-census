import matplotlib.pyplot as plt
import sys
import networkx as nx
import lzma
import pickle
import os
from math import cos, sin, pi
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         })

def remove_self_loops(G: nx.Graph):
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    return G

def plot_disconnected(identifier, k):
    fname = f'{identifier}_{k}_reduced_graph.xz'
    k_1 = k + 1
    fname2 = f'{identifier}_{k_1}_reduced_graph.xz'
    save_name = f'{identifier}_{k}_reduced_graph.png'
    with lzma.open(os.path.join(args.mcmc_output_dir, fname), 'rb') as f, lzma.open(os.path.join(args.mcmc_output_dir, fname2), 'rb') as f2:
        graph, sol_map = pickle.load(f)
        next_graph, next_sol_map = pickle.load(f2)
        # Make this undirected because we don't care about weights here
        G = remove_self_loops(nx.Graph(graph))
        next_G = remove_self_loops(nx.Graph(next_graph))
        components = list(nx.connected_components(G))
        reverse_component_map = {node: i for i, component in enumerate(components) for node in component}
        crossing_edges = [(u, v) for u, v in next_G.edges if reverse_component_map[u] != reverse_component_map[v]]
        crossing_nodes = set(u for u, v in crossing_edges)
        crossing_nodes.update(v for u, v in crossing_edges)
        colors = ['xkcd:light purple' if node in crossing_nodes else 'xkcd:grey' for node in G.nodes()]

        pos = {}
        color_map = {}
        scale_factor = 1.3
        for i, component in enumerate(components):
            subgraph = G.subgraph(component)
            pos.update(nx.spring_layout(subgraph, center=(scale_factor*cos(i*2*pi/len(components)), scale_factor*sin(i*2*pi/len(components)))))
            color_map.update({node: i for node in component})

        nx.draw(G, pos, font_weight='bold', node_size=50, arrows=False, node_color=colors)
        nx.draw_networkx_edges(next_G, pos, edgelist=crossing_edges, edge_color='xkcd:light purple', arrows=False, label=f'crossing edges with k={k_1}', style='dashed')

        plt.savefig(os.path.join('./img/', save_name))
        plt.show()

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    identifiers = ['089-002501-2053', '015-000400-1011']
    # identifiers = ['089-002501-2053']
    k = 2
    for identifier in identifiers:
        plot_disconnected(identifier, k)
