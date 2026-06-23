import os
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt, colors, cm


def ax_set_grid(axis, 
                alpha: float | None = 0.3, 
                zorder: int | None = 0):
    axis.grid(True, alpha=alpha, zorder=zorder)


def parse_params_str(params_values_str: str) -> dict:
    parts = params_values_str.split('_')
    if len(parts) % 2 != 0:
        raise ValueError("Params string must have an even number of underscore-separated parts")
    return {parts[i]: parts[i+1] for i in range(0, len(parts), 2)}


def plot_metric(trains: list, 
                tests: list, 
                epochs: int, 
                model_name: str | None = "model",
                dataset_name: str | None = "dataset",
                metric_name: str | None = "metric",
                figsize: tuple[int, int] | None | None = (25, 3),
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


def plot_log_from_npz(npz_path: str,
                      model_name: str | None = "model",
                      dataset_name: str | None = "dataset",
                      metric_ids_to_plot: list[int] | None = [],
                      metric_ids_to_skip: list[int] | None = [],
                      start_step: int | None = 0,
                      classes_ids: str | list | None = None,
                      are_one_plot: bool | None = False,
                      figsize: tuple[int, int] | None = (10, 30),
                      alpha: float | None = 0.8,
                      linewidth: float | None = 1,
                      cmap_name: str | None = "jet",
                      save_graph: bool | None = False,
                      graph_dir: str | None = "graphs"):
    metrics_log = np.load(npz_path, allow_pickle=True)
    
    metrics = metrics_log["metrics"][start_step:]
    metric_names = metrics_log["metric_names"]
    
    metric_ids = metric_ids_to_plot or [ metric_id for metric_id in range(metrics.shape[1]) ]
    for metric_id_to_skip in metric_ids_to_skip:
        if metric_id_to_skip in metric_ids:
            metric_ids.remove(metric_id_to_skip)
    n_metrics = len(metric_ids)

    start_step_comment = f"(from step {start_step})" if start_step else ""
    classes_ids_comment = f"[classes: {classes_ids}]" if classes_ids else ""
    if are_one_plot:
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.get_cmap(cmap_name)
        
        for metric_i, metric_id in enumerate(metric_ids):
            ax.plot(
                metrics_log["steps"][start_step:], 
                metrics[:, metric_id], 
                lw=linewidth, alpha=alpha,
                color=colors(metric_i / n_metrics),
                label=metric_names[metric_id]
            )
        ax.set(
            title=f"Inversion training losses of {model_name} "\
                  f"on {dataset_name} {classes_ids_comment} {start_step_comment}",
            xlabel="steps",
            ylabel="loss"
        )
        ax_set_grid(ax)
        ax.legend()
        
    else:
        fig, axs = plt.subplots(n_metrics, 1, figsize=figsize)
        for metric_i, metric_id in enumerate(metric_ids):
            axs[metric_i].plot(
                metrics_log["steps"][start_step:], 
                metrics[:, metric_id], 
                lw=linewidth, alpha=alpha
            )
            axs[metric_i].set(
                title=f"Inversion training loss {metric_names[metric_id]} of {model_name} "\
                      f"on {dataset_name} {classes_ids_comment} {start_step_comment}",
                xlabel="steps",
                ylabel=metric_names[metric_id]
            )
            
            ax_set_grid(axs[metric_i])
            axs[metric_i].legend()

    plt.tight_layout()
    if save_graph:
        plt.savefig(os.path.join(graph_dir, f"npz_log_{model_name}_{dataset_name}.png"))
  

def plot_logs_network_grid(network_name: str,
                           all_logs_dir: str,
                           log_filename: str,
                           dataset_name: str = "ImageNet",
                           metric_ids_to_skip: list[int] | None = [],
                           start_step: int = 0,
                           figsize: tuple[int, int] | None = None,
                           alpha: float = 0.6,
                           linewidth: float = 1.0,
                           cmap_name: str = "Set3",
                           save_graph: bool = False,
                           graph_dir: str = "graphs"):
    network_logs_dir = os.path.join(all_logs_dir, f"inversion_{network_name}_{dataset_name}")

    # load metrics for every class & batch_size assuming dir structure: bs_<batch_size>_classes_<class_id>/training_log.npz
    data = defaultdict(lambda: defaultdict(list))  # class_id : { batch_size : metrics array }

    # get all classes & batch_sizes to determine grid size
    all_classes = set()
    all_batch_sizes = set()

    for dir_name in os.listdir(network_logs_dir):
        dir_path = os.path.join(network_logs_dir, dir_name)
        
         # skip folders & files with invalid names
        if not os.path.isdir(dir_path):
            continue
        try:
            params = parse_params_str(dir_name)
        except ValueError:
            continue
        if "bs" not in params or "classes" not in params:
            continue
        
        class_id = params["classes"]
        batch_size = params["bs"]
        npz_path = os.path.join(dir_path, log_filename)
        if not os.path.exists(npz_path):
            continue

        metrics_log = np.load(npz_path, allow_pickle=True)
        metrics = metrics_log["metrics"]  # shape (steps, n_metrics)
        metric_names = metrics_log["metric_names"]

        metrics = metrics[start_step:, :]
        data[class_id][batch_size] = metrics
        all_classes.add(class_id)
        all_batch_sizes.add(batch_size)

    # sort classes & batch sizes
    classes_sorted = sorted(all_classes, key=lambda x: int(x) if x.isdigit() else x)
    batch_sizes_sorted = sorted(all_batch_sizes, key=lambda x: int(x) if x.isdigit() else x)
    n_classes = len(classes_sorted)
    n_metrics_total = metrics.shape[1]
    
    # filter metrics
    all_metric_ids = list(range(n_metrics_total))
    metric_ids_to_plot = [i for i in all_metric_ids if i not in metric_ids_to_skip]
    n_metrics = len(metric_ids_to_plot)
    metric_names_filtered = [metric_names[i] for i in metric_ids_to_plot]

    figsize = figsize or (4 * n_classes, 3 * n_metrics)

    fig, axs = plt.subplots(n_metrics, n_classes, figsize=figsize, squeeze=False)
    fig.suptitle(f"Inversion training logs for {network_name} on {dataset_name}", fontsize=24, y=1.02)

    n_batch = len(batch_sizes_sorted)
    cmap = plt.get_cmap(cmap_name, n_batch)
    colors = [cmap(i) for i in range(n_batch)]

    for row, metric_id in enumerate(metric_ids_to_plot):
        for col, class_id in enumerate(classes_sorted):
            ax = axs[row, col]
            for bs, color in zip(batch_sizes_sorted, colors):
                if bs in data[class_id]:
                    metrics_arr = data[class_id][bs]
                    steps = np.arange(start_step, start_step + metrics_arr.shape[0])
                    ax.plot(steps, metrics_arr[:, metric_id],
                            lw=linewidth, alpha=alpha, color=color)
            ax_set_grid(ax)
            if not row:
                ax.set_title(f"Class {class_id}")
            if row == n_metrics - 1:
                ax.set_xlabel("steps")
            if not col:
                ax.set_ylabel(metric_names_filtered[row])

    # shared legend for all batch_sizes
    legend_handles = [
        plt.Line2D(
            [0], [0], 
            color=colors[i], 
            lw=linewidth, 
            label=f"bs={bs}"
        )
        for i, bs in enumerate(batch_sizes_sorted)
    ]
    fig.legend(
        handles=legend_handles, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(n_batch, 8), title="Batch size"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for the upper legend

    if save_graph:
        os.makedirs(graph_dir, exist_ok=True)
        save_path = os.path.join(graph_dir, f"npz_log_grid_{network_name}_{dataset_name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()
    plt.close(fig)


