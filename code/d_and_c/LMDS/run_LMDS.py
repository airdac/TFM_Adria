import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.datasets import make_swiss_roll

from scipy.spatial.distance import pdist, squareform
from d_and_c.methods import local_mds

n, r = 3000, 2
np.random.seed(42)
X, color = make_swiss_roll(n_samples=n, random_state=42)

k = 10
tau = 0.01
itmax = 20000
principal_components = True
embedding = local_mds(
    X, r, principal_components=principal_components, k=k, tau=tau, itmax=itmax, verbose=2)

# Plot embedding
plt.scatter(embedding[:, 0], embedding[:, 1],
            c=color, cmap=plt.cm.Spectral)
plot_path_elements = [f'dataset=swiss_roll',
                      f'n={n}',
                      f'method=Local_MDS',
                      f'k={k}',
                      f'tau={tau}',
                      f'itmax={itmax}',
                      f'principal_components={principal_components}']
title_lines = [', '.join(plot_path_elements[:3]),
               ', '.join(plot_path_elements[3:])]
title = '\n'.join(title_lines)
plt.suptitle(title, fontsize=12, y=0.98)
plt.tight_layout()
output_path = os.path.join('d_and_c', 'LMDS', 'plots')
plots_path = os.path.join(output_path, *plot_path_elements)
os.makedirs(plots_path, exist_ok=True)
plt.savefig(os.path.join(plots_path, 'embedding.png'))
plt.close()
