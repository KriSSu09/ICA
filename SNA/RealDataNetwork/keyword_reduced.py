import networkx as nx
import requests

API_KEY = "xdf9HSiXK8510t7HU1bVdpF2R36I0UQ4"
BASE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"


def fetch_articles(year, month):
    """Fetch articles from NYT Archive API for a given year and month."""
    url = BASE_URL.format(year=year, month=month)
    params = {'api-key': API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    return data['response']['docs']


def build_networks(articles):
    """Build and return both author and keyword networks from a list of articles."""
    keyword_graph = nx.Graph()

    articles = articles[:len(articles) // 8]
    print(len(articles))
    for idx, article in enumerate(articles):
        # Add nodes for each article, using URL as unique identifier
        url = article['web_url']
        keyword_graph.add_node(url)

        # Add edges based on common keywords
        keywords = {keyword['value'] for keyword in article['keywords']}
        for other_url, other_keywords in keyword_graph.nodes.data('keywords', default=[]):
            if keywords.intersection(other_keywords):
                keyword_graph.add_edge(url, other_url)

        # Update node attributes
        keyword_graph.nodes[url]['keywords'] = keywords

        if idx % 100 == 0:
            print(f'Processed {idx} articles')

    return keyword_graph


def save_graph(graph, filename):
    """Save the graph to a file."""
    nx.readwrite.write_adjlist(graph, filename)


# Process articles for 6 specific months
months_pre = [(2019, 4)]
months_post = [(2020, 4)]
articles_pre = [article for year, month in months_pre for article in fetch_articles(year, month)]
articles_post = [article for year, month in months_post for article in fetch_articles(year, month)]

# Build networks
keyword_graph_pre = build_networks(articles_pre)
keyword_graph_post = build_networks(articles_post)

# Save networks to files
save_graph(keyword_graph_pre, 'keyword_network_pre.txt')
save_graph(keyword_graph_post, 'keyword_network_post.txt')
