import numpy as np
import plotly.graph_objects as go

from d_and_c.utils import benchmark, runif_in_tetrahedron

n = 1000
vertices = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0.5, np.sqrt(3)/2, 0],
                       [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
                       ])

solid_tetrahedron, runtime_original = benchmark(runif_in_tetrahedron,
                                                n, vertices)

print(f'Runtime: {runtime_original}')

# Compute the closest vertex for each point in sims
# Use NumPy broadcasting to compute distance between the i-th point and the j-th vertex
distances = np.linalg.norm(
    solid_tetrahedron[:, None, :] - vertices[None, :, :], axis=2)
closest_vertex = np.argmin(distances, axis=1)

# Define a color map
colors_map = np.array(['red', 'green', 'blue', 'orange'])
point_colors = colors_map[closest_vertex]

# Plot Data in Tetrahedron Mesh
# i, j, k are the indices of the tetrahedron vertices that make up the x, y, z coordinates of the faces' vertices
fig = go.Figure(data=[
    go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=[0, 0, 0, 1],
        j=[1, 1, 2, 2],
        k=[2, 3, 3, 3],
        color='grey',
        opacity=0.2,
        name='Tetrahedron'
    ),
    go.Scatter3d(
        x=solid_tetrahedron[:, 0],
        y=solid_tetrahedron[:, 1],
        z=solid_tetrahedron[:, 2],
        mode='markers',
        marker=dict(size=3, color=point_colors),
        name='Random Points'
    )
])

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'
    ),
    title="Solid regular tetrahedron: Data"
)

fig.show()
