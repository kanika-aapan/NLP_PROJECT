import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from kwQnA._getentitypair import GetEntity


class GraphEnt:
    """docstring for graphEnt."""

    def __init__(self):
        super(GraphEnt, self).__init__()
        self.x = GetEntity()

    def createGraph(self, dataEntities):
        entity_list = dataEntities.values.tolist()
        source, relations, target = [], [], []

        for i in entity_list:
            # Skip empty entries
            if i[0] == "" or i[3] == "":
                continue
                
            source.append(i[0])
            relations.append(i[1])
            target.append(i[3])

        # Create DataFrame for graph creation
        kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
        
        # Create directed multigraph from pandas DataFrame
        G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr="edge", create_using=nx.MultiDiGraph())
        
        # Create edge labels dictionary
        edge_labels = {}
        for src, dst, data in G.edges(data=True):
            if (src, dst) not in edge_labels:
                edge_labels[(src, dst)] = data['edge']
            else:
                # Handle multiple edges between same nodes
                if isinstance(edge_labels[(src, dst)], list):
                    if data['edge'] not in edge_labels[(src, dst)]:
                        edge_labels[(src, dst)].append(data['edge'])
                else:
                    if data['edge'] != edge_labels[(src, dst)]:
                        edge_labels[(src, dst)] = [edge_labels[(src, dst)], data['edge']]

        # Visualization settings
        plt.figure(figsize=(16, 16))
        
        # Use spring layout with larger k for better spacing
        pos = nx.spring_layout(G, k=2.5, seed=42)  # Fixed seed for reproducibility
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='skyblue', 
                              node_size=2000, 
                              alpha=0.9)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=12, 
                               font_weight='bold')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              width=2, 
                              alpha=0.7, 
                              edge_color='gray',
                              arrowsize=20)
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, 
                                     edge_labels=edge_labels, 
                                     font_size=10,
                                     font_color='red',
                                     alpha=0.7)
        
        # Add title and adjust margins
        plt.title("Knowledge Graph with Relation Labels", fontsize=20, pad=20)
        plt.margins(0.1)
        plt.tight_layout()
        plt.axis('off')
        
        # Show the graph
        plt.show()
        
        # Save the graph to a file
        plt.savefig("knowledge_graph.png", format="PNG", dpi=300, bbox_inches='tight')
        print("Knowledge graph visualization saved as 'knowledge_graph.png'")
        
        return G  # Return the graph object for potential further analysis

if __name__ == '__main__':
    test = GraphEnt()
    print("Can't Test directly")