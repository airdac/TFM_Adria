import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.datasets import make_swiss_roll

from d_and_c.methods import tsne
from d_and_c.utils import benchmark

n, r = 1000, 2
np.random.seed(42)
X, color = make_swiss_roll(n_samples=n, random_state=42)

perplexity = 30
learning_rate = "auto"
n_iter = 250
principal_components = True
embedding, runtime = benchmark(tsne,
    X, r, principal_components=principal_components, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate, verbose=2)

# Plot embedding
plt.scatter(embedding[:, 0], embedding[:, 1],
            c=color, cmap=plt.cm.Spectral)
plot_path_elements = [f'dataset=swiss_roll',
                      f'n={n}',
                      f'method=t-SNE',
                      f'perplexity={perplexity}',
                      f'learning_rate={learning_rate}',
                      f'n_iter={n_iter}',
                      f'principal_components={principal_components}']
title_lines = [', '.join(plot_path_elements[:3]),
               ', '.join(plot_path_elements[3:]),
               f'Runtime: {runtime:.2f} s']
title = '\n'.join(title_lines)
plt.suptitle(title, fontsize=12, y=0.98)
plt.tight_layout()
output_path = os.path.join('d_and_c', 't-SNE', 'plots')
plots_path = os.path.join(output_path, *plot_path_elements)
os.makedirs(plots_path, exist_ok=True)
plt.savefig(os.path.join(plots_path, 'embedding.png'))
plt.close()
