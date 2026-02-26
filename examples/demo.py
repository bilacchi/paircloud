import matplotlib.pyplot as plt
import numpy as np

import paircloud


def main():
    # Make some sample data: two overlapping distributions
    np.random.seed(42)
    group1 = np.random.normal(loc=5, scale=1.5, size=400)
    group2 = np.random.normal(loc=7, scale=1.2, size=300)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1. Faded Dotplot
    ax = axes[0]
    paircloud.faded_dotplot(group1, ax=ax, color='#E63946', position=1, orientation='v')
    paircloud.faded_dotplot(group2, ax=ax, color='#457B9D', position=2, orientation='v')
    ax.set_title('Faded Dotplots')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Group 1', 'Group 2'])

    # 2. Shadeplot
    ax = axes[1]
    paircloud.shadeplot(group1, ax=ax, color='#E63946', position=1, orientation='v')
    paircloud.shadeplot(group2, ax=ax, color='#457B9D', position=2, orientation='v')
    ax.set_title('Shadeplots')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Group 1', 'Group 2'])

    # 3. Raincloud
    ax = axes[2]
    paircloud.raincloud(group1, ax=ax, color='#E63946', position=1, orientation='v')
    paircloud.raincloud(group2, ax=ax, color='#457B9D', position=2, orientation='v')
    ax.set_title('Raincloud Plots')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Group 1', 'Group 2'])

    # 4. Paired Raincloud
    ax = axes[3]
    # filter to identical lengths to act as "repeated measures" data
    paired1 = group1[:300]
    paired2 = group2[:300]
    paircloud.paired_raincloud(
        paired1,
        paired2,
        ax=ax,
        colors=('#E63946', '#457B9D'),
        positions=(1, 2),
        orientation='v',
    )
    ax.set_title('Paired Raincloud Plots')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Time 1', 'Time 2'])

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('demo.png', dpi=300)
    print('Saved demo.png')


if __name__ == '__main__':
    main()
