import os
import numpy as np
from matplotlib import pyplot as plt, colors, cm
from matplotlib.patches import Patch
from typing import Iterable, Callable
import pandas as pd


# AUX FUNCTIONS


def ax_set_grid(axis, 
                alpha: float | None = 0.3, 
                zorder: int | None = 0):
    axis.grid(True, alpha=alpha, zorder=zorder)


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


# BASIC PLOTS


def plot_metric(trains: list, 
                tests: list, 
                epochs: int, 
                model_name: str | None = "model",
                dataset_name: str | None = "dataset",
                metric_name: str | None = "metric",
                figsize: tuple[int, int] | None = (25, 3),
                save_graph: bool | None = False,
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
    ax_set_grid(ax)
    ax.legend()

    if save_graph:
        plt.savefig(os.path.join(graph_dir, f"regnet_{metric_name}_{epochs}epochs.png"))


def plot_heatmap(values,
                 cmap_name: str | None = "seismic",
                 title: str | None = "Weights heatmap",
                 figsize: tuple[int, int] | None = (25, 3),
                 save_graph: bool | None = False,
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
                         save_graph: bool | None = False,
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
    ax_set_grid(ax)
    plt.tight_layout()

    if save_graph:
        plt.savefig(os.path.join(graph_dir, f"regnet_{title}.png"))


def plot_histogram(values: np.ndarray,
                   bins: int | None = 50,
                   title: str | None = "Histogram of values",
                   xlabel: str | None = "value",
                   ylabel: str | None = "frequency",
                   cmap_name: str | None = "berlin",
                   edge_color: str | None = "black",
                   alpha: float | None = 0.7,
                   figsize: tuple | None = (25, 3),
                   save_graph: bool | None = False,
                   graph_dir: str | None = "graphs"):
    values_flat = values.flatten()
    fig, ax = plt.subplots(figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        values_flat, 
        bins=bins, 
        edgecolor=edge_color, 
        alpha=alpha
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cmap = plt.get_cmap(cmap_name)
    norm = centralize_colormap(values_flat)          # symmetric normalization around 0
    for center, patch in zip(bin_centers, patches):
        patch.set_facecolor(cmap(norm(center)))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax_set_grid(ax)
    plt.tight_layout()

    if save_graph:
        os.makedirs(graph_dir, exist_ok=True)
        plt.savefig(os.path.join(graph_dir, f"{title}.png"))


# SPECIFIC PLOTS


def plot_metrics_from_df(metrics: pd.DataFrame,
                         x_column: str,
                         title: str,
                         x_label: str,
                         y_label: str,
                         columns_to_plot: list[str] | None = None,
                         emph_columns: list[str] | None = None,
                         emph_condition: Callable | None = None,
                         emph_condition_name: str | None = "<0",
                         cmap_name: str = "winter",
                         emph_cmap_name: str = "spring",
                         figsize: tuple = (20, 7)):
    columns_to_skip = [x_column] \
        if (emph_columns is None) or len(set(columns_to_plot).intersection(emph_columns)) else \
                      [x_column] + emph_columns
    metric_names = columns_to_plot or [
        column for column in metrics.columns if (column not in columns_to_skip)
    ]
    x_values = metrics[x_column].values
    unique_x = sorted(metrics[x_column].unique())

    vlines_x = []
    vlines_colors = []
    column_met_condition = {
        column: False for column in emph_columns
    } if emph_columns else {}
    all_conds_met = False

    if emph_columns is not None:
        n_emph = len(emph_columns)
        emph_cmap = plt.get_cmap(emph_cmap_name)
        for i, row in metrics.iterrows():
            conds = [emph_condition(row[column]) for column in emph_columns]
            if any(conds):
                x_pos = row[x_column]
                vlines_x.append(x_pos)
                if all(conds):
                    vlines_colors.append("red")
                    all_conds_met = True
                else:
                    first_idx = next(i for i, c in enumerate(conds) if c)
                    vlines_colors.append(emph_cmap(first_idx / (n_emph - 1)))
                    column_met_condition[emph_columns[first_idx]] = True

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(cmap_name)
    max_metric = 0
    min_metric = -1000

    for i, column in enumerate(metric_names):
        y_values = metrics[column].values
        max_metric = max(max_metric, y_values.max())
        min_metric = min(min_metric, y_values.min())
        ax.plot(
            x_values, 
            y_values, 
            marker='o', markersize=4,
            color=cmap(i / len(metric_names)), 
            label=column
        )

    for x, color in zip(vlines_x, vlines_colors):
        ax.axvline(
            x, 
            color=color, alpha=0.8, linewidth=1.2, 
            zorder=3
        )

    emph_legend_elements = []
    if all_conds_met:
        emph_legend_elements.append(Patch(
            facecolor="red", 
            edgecolor="red",
            label="All conditions met"
        ))
    if emph_columns:
        emph_cmap = plt.get_cmap(emph_cmap_name)
        for i, column in enumerate(emph_columns):
            if column_met_condition[column]:
                color = emph_cmap(i / (n_emph - 1))
                emph_legend_elements.append(Patch(facecolor=color, edgecolor=color,
                                                  label=column + emph_condition_name))

    if max_metric > 90:
        y_ticks = list(range(0, 101, 5))
    elif max_metric <= 2:
        y_ticks = [i * 0.1 for i in range(-5, 20)]
    elif max_metric < 5:
        y_ticks = [i * 0.2 for i in range(0, 25)]
    elif max_metric < 10:
        y_ticks = [i * 0.5 for i in range(0, int(max_metric) * 2)]
    else:
        y_ticks = list(range(0, int(max_metric) + 6, 5))

    ax.set(
        xticks=unique_x,
        xlabel=x_label,
        yticks=y_ticks,
        ylabel=y_label,
        title=title,
    )
    # ax.set_xticklabels(ax.get_xticks(), rotation=-90)
    ax_set_grid(ax, 0.5)

    curve_legend = ax.legend(loc='upper left')
    if emph_legend_elements:
        ax.legend(handles=emph_legend_elements, loc='lower left')
        ax.add_artist(curve_legend)

    plt.tight_layout()



def plot_bin_scale_k_metrics(metric_dicts: list[dict],
                             metric_names: list[str],
                             truebin_scale_ks: Iterable | None = [],
                             cmap_name: str | None = "winter",
                             coeff_name: str | None = "",
                             figsize: tuple = (20, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    metric_cmap = plt.get_cmap(cmap_name)
    max_metric = 0
    
    for i, metric_dict in enumerate(metric_dicts):
        scale_ks = sorted(metric_dict.keys())
        metric_values = [metric_dict[scale_k] for scale_k in scale_ks]
        max_metric = max(max_metric, max(metric_values))
        
        ax.plot(
            scale_ks, 
            metric_values, 
            marker='o', 
            color=metric_cmap(i / len(metric_dicts)),
            label=metric_names[i]
        )
    for truebin_scale_k in truebin_scale_ks:
        ax.axvline(
            truebin_scale_k, 
            alpha=0.8, lw=0.3,
            color = "red"
        )
    
    if max_metric > 90:
        y_ticks_metrics = [i for i in range(0, 101, 5)]
    elif max_metric < 5:
        y_ticks_metrics = [i*0.2 for i in range(0, 25)]
    elif max_metric < 10:
        y_ticks_metrics = [i*0.5 for i in range(0, int(max_metric) * 2)]
    else:
        y_ticks_metrics = [i for i in range(0, int(max_metric) + 6, 5)]
        
    ax.set(
        xticks=scale_ks,
        xlim=(min(scale_ks) - 0.01, max(scale_ks) + 0.01),
        xlabel="Scaling coefficient",
        yticks=y_ticks_metrics,
        ylabel="Accuracy (in %)",
        title="Accuracy dependance on " + coeff_name + " scaling coefficient"
    )
    ax.set_xticklabels(ax.get_xticks(), rotation=-90)
    ax_set_grid(ax)
    ax.legend()

    plt.tight_layout()
    
    
def plot_svd_metrics(metrics: dict,
                     n_components_options: Iterable,
                     n_components_options_sparse: Iterable,
                     signific_thresholds: Iterable,
                     cmap_name: str | None = "Dark2"):
    fig, axs = plt.subplots(1, 3, figsize=(25, 7))

    axs[0].plot(n_components_options, metrics["explained_variances"], "b", alpha=0.6, label="explained variances")
    axs[0].plot(n_components_options, metrics["reconstruction_rmse"], "g", alpha=0.6, label="reconstruction RMSE")
    axs[0].plot(n_components_options, metrics["decomposition_time"],  "r", alpha=0.6, label="decomposition time, s")

    axs[0].set(
        xticks=n_components_options_sparse,
        xlabel="number of SVD components", 
        ylabel="metrics",
        title="3 metrics dependance on SVD n_components"
    )

    n_components_cmap = plt.get_cmap(cmap_name)
    for i in range(len(signific_thresholds)):
        axs[1].plot(
                n_components_options, 
                metrics[f"n_significant_features_{signific_thresholds[i]}"],  
                color=n_components_cmap(i / len(signific_thresholds)),
                marker="o", 
                alpha=0.6, label=f"n significant features >= {signific_thresholds[i]}"
        )
    axs[1].set(
        xticks=n_components_options_sparse,
        xlabel="number of SVD components", 
        yticks=[0] + [
                n_s_fs[-1] 
                for metric_name, n_s_fs in metrics.items()
                if ("n_significant_features" in metric_name) and (n_s_fs[-1] > 1000) 
        ],
        ylabel="number of significant features",
        title="Significant features number dependance on SVD n_components"
    )

    for i in range(len(signific_thresholds)):
        axs[2].plot(
                n_components_options, 
                metrics[f"n_significant_features_{signific_thresholds[i]}"],  
                color=n_components_cmap(i / len(signific_thresholds)),
                marker="o", 
                alpha=0.6, label=f"n significant features >= {signific_thresholds[i]}"
        )
    axs[2].set(
        xticks=n_components_options_sparse,
        xlabel="number of SVD components", 
        yticks=[0] + [
                n_fs
                for metric_name, n_s_fs in metrics.items()
                if ("n_significant_features" in metric_name) 
                for n_fs in [n_s_fs[0], n_s_fs[-1]]
        ],
        ylabel="number of significant features",
        ylim=(0, 2000),
        title="Significant features number dependance on SVD n_components (zoom)"
    )

    for ax in axs:
        ax.set_xticklabels(ax.get_xticks(), rotation=-90)
        ax_set_grid(ax)
        ax.legend()

    plt.tight_layout()
    plt.show()
    

def plot_svd_metric(metrics: dict,
                    metric_name: dict,
                    n_components_options: list,
                    n_components_options_sparse: list,
                    signific_thresholds: list,
                    color: str | None = "g"):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        n_components_options, 
        metrics[metric_name], 
        color, 
        alpha=0.6)

    ax.set(
        xticks=n_components_options_sparse,
        xlabel="number of SVD components", 
        ylabel="RMSE",
        title="Reconstruction RMSE dependance on SVD n_components (zoomed)"
    )
    ax.grid(zorder=0)