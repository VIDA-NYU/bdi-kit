from streamlit_agraph import Config, Edge, Node, agraph


def draw_recommandation_graph(top_k_results):
    nodes = []
    edges = []
    for result in top_k_results:
        candidate_column = result["Candidate column"]
        nodes.append(
            Node(id=candidate_column, label=candidate_column, size=25, shape="hexagon")
        )
        for name, score in result["Top k columns"]:
            child_name = f"{candidate_column}-{name}"
            nodes.append(Node(id=child_name, label=name, size=5, shape="dot"))
            edges.append(
                Edge(
                    source=candidate_column,
                    label=score,
                    target=child_name,
                )
            )

    return nodes, edges
