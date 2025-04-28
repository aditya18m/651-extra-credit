import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def extract_graph_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=10, maxRadius=30
    )

    G = nx.Graph()
    pos = {}

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i, (x, y, r) in enumerate(circles):
            node_id = str(i)
            G.add_node(node_id)
            pos[node_id] = (x, y)

    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            closest_nodes = []
            for node_id, (nx_, ny_) in pos.items():
                d1 = np.hypot(x1 - nx_, y1 - ny_)
                d2 = np.hypot(x2 - nx_, y2 - ny_)
                if d1 < 50 or d2 < 50:
                    closest_nodes.append((node_id, min(d1, d2)))

            closest_nodes = sorted(set(closest_nodes), key=lambda x: x[1])
            if len(closest_nodes) >= 2:
                node1 = closest_nodes[0][0]
                node2 = closest_nodes[1][0]
                if node1 != node2:
                    G.add_edge(node1, node2)

    return G, pos
    
def visualize_graph(G, pos, title="Graph from Image"):
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    image_paths = ["image1.png", "image2.png", "image3.jpg"]

    for img_path in image_paths:
        G, pos = extract_graph_from_image(img_path)
        visualize_graph(G, pos, title=f"Graph from {img_path}")