import os
import numpy as np
from matplotlib import pyplot as plt, colors, cm


def plot_metric(trains: list, 
                tests: list, 
                epochs: int, 
                model_name: str | None = "model",
                dataset_name: str | None = "dataset",
                metric_name: str | None = "metric",
                figsize: tuple[int, int] | None | None = (25, 3),
                save_graph: bool | None = True,
                graph_dir: str | None = "graphs"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        [i+1 for i in range(epochs)],
        trains, 
        "b.-", alpha=0.6, lw=0.3, ms=0.8,
        label="train " + metric_name
    )
    ax.plot(
        [i+1 for i in range(epochs)],
        tests, 
        "r.-", alpha=0.8, lw=0.3,
        label="test " + metric_name
    )
    ax.set(
        title=f"{model_name} {metric_name} on {dataset_name} for {epochs} epochs",
        xlabel="epochs", xlim=(-1, epochs+1), xticks=[i for i in range(0, epochs+1, 5)],
        ylabel=metric_name
    )
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend()

    if save_graph:
        plt.savefig(os.path.join(graph_dir, f"regnet_{metric_name}_{epochs}epochs.png"))
    

def centralize_colormap(values, 
                        center: float | None = 0):
    values_min = np.min(values)
    values_max = np.max(values)
    border = max(abs(values_min), abs(values_max))
    
    return colors.TwoSlopeNorm(
        vmin=-border,
        vcenter=center, 
        vmax=border
    )


def plot_heatmap(values,
                 cmap_name: str | None = "seismic",
                 title: str | None = "Weights heatmap",
                 figsize: tuple[int, int] | None = (25, 3),
                 save_graph: bool | None = True,
                 graph_dir: str | None = "graphs"):
    plt.figure(figsize=figsize)
    plt.imshow(
        values, 
        origin="lower",
        cmap=cmap_name, 
        norm=centralize_colormap(values),
        extent=(0, len(values[0]) + 1, 0, len(values) + 1)
    )  
    plt.title(title)  
    plt.colorbar()  

    if save_graph:
        plt.savefig(os.path.join(graph_dir, f"regnet_{title}.png"))


def plot_colored_barplot(values,
                         values_color,
                         values_color_name: str,
                         cmap_name: str | None = "seismic",
                         edge_color: str | None = "black",
                         title: str | None = "Values barplot",
                         xlabel: str | None = "values",
                         ylabel: str | None = "values number",
                         xticks: list | None = [],
                         xticklabels: list | None = [],   
                         figsize: tuple[int, int] | None = (25, 3),
                         save_graph: bool | None = True,
                         graph_dir: str | None = "graphs"):
    fig, ax = plt.subplots(figsize=figsize)
    
    values_cmap = plt.get_cmap(cmap_name)
    norm = centralize_colormap(values_color)
    ax.bar(
        [i for i in range(len(values))],
        values, 
        color=values_cmap(norm(values_color)),
        edgecolor=edge_color
    )
    
    scalar_map = cm.ScalarMappable(
        cmap=values_cmap, 
        norm=norm
    )
    scalar_map.set_array([])
    colorbar = plt.colorbar(scalar_map, ax=ax)
    colorbar.set_label(values_color_name)
    
    if len(xticks):
        ax.set(
            title=title,
            xlabel=xlabel,
            xticks=xticks,
            ylabel=ylabel
        )
        ax.set_xticklabels(xticklabels, rotation=-45)
    else:
        ax.set(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel
        )
    plt.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()

    if save_graph:
        plt.savefig(os.path.join(graph_dir, f"regnet_{title}.png"))
