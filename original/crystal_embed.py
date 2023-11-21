from pymatgen.core import Structure
import plotly.graph_objects as go
from collections import Counter
import networkx as nx
import re
from PIL import Image
import io
import plotly.io as pio
import numpy as np


def visualize_structure(structure):
    """
    Visualizes the crystal structure using a graph representation.

    Args:
    - structure: pymatgen.core.structure.Structure object representing the crystal structure.

    Returns:
    - data: numpy.ndarray of shape (n_images, 32, 32, 3) representing the RGB data of the images.
    """

    # Set the radius cutoff for identifying neighbors
    radius_cutoff = 4  # angstroms

    # Create a list of neighbors for each site in the structure
    neighbors_list = []
    elements = []
    for i, site in enumerate(structure):
        site_id = f"{site.specie}-{i}"
        neighbors = structure.get_sites_in_sphere(site.coords, radius_cutoff)
        neighbor_indices = list(
            set([f"{neighbor.specie}-{neighbor.index}" for neighbor in neighbors])
        )
        neighbors_list.append(neighbor_indices)
        elements.append(site_id)

    # Create a list of edges between neighboring sites
    edges = []
    for i in range(len(elements)):
        for j in range(len(neighbors_list[i])):
            edges.append([elements[i], neighbors_list[i][j]])

    # Flatten the list of lists
    flattened = [item for sublist in edges for item in sublist]

    # Count the occurrences of each element in the flattened list
    counts = Counter(flattened)

    # Get all elements that appear more than once
    duplicates = [element for element in counts if counts[element] > 1]

    # Filter the list of paired lists to only include items that appear more than once
    edges = [edge for edge in edges if edge[0] in duplicates and edge[1] in duplicates]

    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for sublist in edges:
        for node in sublist:
            G.add_node(node)

    # Add edges to the graph
    for sublist in edges:
        G.add_edge(sublist[0], sublist[1])

    # Give a random position to each node
    pos = nx.spring_layout(G)
    for node in G.nodes():
        G.nodes[node]["pos"] = pos[node].tolist()

    # Create a list of x and y coordinates for each edge
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]["pos"]
        x1, y1 = G.nodes[edge[1]]["pos"]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Create a scatter plot of the edges
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#000"),
        hoverinfo="none",
        mode="lines",
    )

    # Create a list of x and y coordinates for each node
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)

    # Assign colors to each node based on its element
    node_colors = []
    import plotly.colors as pc

    # Create a dictionary of colors based on valency
    valency_table = {
        "H": 1,
        "He": 0,
        "Li": 1,
        "Be": 2,
        "B": 3,
        "C": 4,
        "N": 5,
        "O": 6,
        "F": 7,
        "Ne": 8,
        "Na": 1,
        "Mg": 2,
        "Al": 3,
        "Si": 4,
        "P": 5,
        "S": 6,
        "Cl": 7,
        "Ar": 8,
        "K": 1,
        "Ca": 2,
        "Sc": 3,
        "Ti": 4,
        "V": 5,
        "Cr": 6,
        "Mn": 7,
        "Fe": 8,
        "Co": 9,
        "Ni": 10,
        "Cu": 11,
        "Zn": 12,
        "Ga": 3,
        "Ge": 4,
        "As": 5,
        "Se": 6,
        "Br": 7,
        "Kr": 8,
        "Rb": 1,
        "Sr": 2,
        "Y": 3,
        "Zr": 4,
        "Nb": 5,
        "Mo": 6,
        "Tc": 7,
        "Ru": 8,
        "Rh": 9,
        "Pd": 10,
        "Ag": 11,
        "Cd": 12,
        "In": 3,
        "Sn": 4,
        "Sb": 5,
        "Te": 6,
        "I": 7,
        "Xe": 8,
        "Cs": 1,
        "Ba": 2,
        "La": 3,
        "Ce": 4,
        "Pr": 5,
        "Nd": 6,
        "Pm": 7,
        "Sm": 8,
        "Eu": 9,
        "Gd": 10,
        "Tb": 11,
        "Dy": 12,
        "Ho": 13,
        "Er": 14,
        "Tm": 15,
        "Yb": 16,
        "Lu": 17,
        "Hf": 4,
        "Ta": 5,
        "W": 6,
        "Re": 7,
        "Os": 8,
        "Ir": 9,
        "Pt": 10,
        "Au": 11,
        "Hg": 12,
        "Tl": 3,
        "Pb": 4,
        "Bi": 5,
        "Th": 4,
        "Pa": 5,
        "U": 6,
        "Np": 6,
        "Pu": 6,
        "Am": 6,
        "Cm": 6,
        "Bk": 6,
        "Cf": 6,
        "Es": 6,
        "Fm": 6,
        "Md": 6,
        "No": 6,
        "Lr": 3,
        "Rf": 4,
        "Db": 5,
        "Sg": 6,
        "Bh": 7,
        "Hs": 8,
        "Mt": 9,
        "Ds": 10,
        "Rg": 11,
        "Cn": 12,
        "Nh": 1,
        "Fl": 2,
        "Mc": 3,
        "Lv": 4,
        "Ts": 5,
        "Og": 6,
        "Ac": 3,
    }

    # Create a list of colors based on the periodic table
    colors = pc.qualitative.Light24

    # Create a dictionary that maps valency to colors
    color_dict = {}
    for valency, number in valency_table.items():
        color_dict[number] = colors[number % len(colors)]

    # Assign a color to each node based on its valency
    node_colors = []
    for node in G.nodes():
        element = re.sub("[^a-zA-Z]", "", node.split("-")[0])
        valency = valency_table[element]
        color = color_dict[valency]
        node_colors.append(color)

    # Create a scatter plot of the nodes
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=list(G.nodes()),
        hoverinfo="text",
        marker=dict(
            color=node_colors,
            size=4,
            line_width=0.5,
        ),
    )

    # Add the number of connections to each node's text
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(list(G.nodes)[node])

    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            width=32,
            height=32,
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            font_size=20,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
    )
    # Debug: save to png or show
    # fig.write_image("fig1.png")
    # fig.show()

    # Convert the figure to bytes
    fig_bytes = pio.to_image(fig, format="png")

    # Convert bytes to an image
    image = Image.open(io.BytesIO(fig_bytes))

    # Convert the image to a NumPy array
    data = np.array(image)

    # Extract only the RGB data
    rgb_data = data[:, :, :3]

    # Reshape the array to the desired shape
    data = rgb_data.reshape((-1, 32, 32, 3))

    # Debug: create an image from the array
    # image = Image.fromarray(data[0].astype(np.uint8))
    # image.save("fig2.png")

    # return data
    return data


# Run test case
# visualize_structure(Structure.from_file("NaCl.cif"))
