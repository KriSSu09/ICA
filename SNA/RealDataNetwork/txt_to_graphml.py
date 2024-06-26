import networkx as nx

author_graph_pre = nx.readwrite.read_adjlist('author_network_pre.txt')
author_graph_post = nx.readwrite.read_adjlist('author_network_post.txt')
keyword_graph_pre = nx.readwrite.read_adjlist('keyword_network_pre.txt')
keyword_graph_post = nx.readwrite.read_adjlist('keyword_network_post.txt')
nx.write_graphml(author_graph_pre, 'author_network_pre.graphml')
nx.write_graphml(author_graph_post, 'author_network_post.graphml')
nx.write_graphml(keyword_graph_pre, 'keyword_network_pre.graphml')
nx.write_graphml(keyword_graph_post, 'keyword_network_post.graphml')