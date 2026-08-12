"""Visualization helpers for truss learning notebooks."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


def plot_training_history(history, train_key=None, val_key=None, log_y=False):
    train_key = train_key or ("train" if "train" in history else "train_loss")
    val_key = val_key or ("val" if "val" in history else "validation_loss")
    epochs = np.arange(1, len(history[train_key]) + 1)
    validation = np.asarray(history[val_key], dtype=float)
    validated = np.isfinite(validation)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, history[train_key], label="Train")
    axis.plot(epochs[validated], validation[validated], marker="o", label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Graph-balanced Smooth L1 loss")
    axis.set_title("Training history")
    if log_y:
        axis.set_yscale("log")
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure


def plot_axial_force_results(predictions, r2, seed=42, max_points=40_000, bins=80):
    rng = np.random.default_rng(seed)

    # -------------------------
    # Data for parity plot
    # -------------------------
    num_edges = len(predictions["axial_true"])
    shown = (
        rng.choice(num_edges, size=max_points, replace=False)
        if num_edges > max_points
        else np.arange(num_edges)
    )

    true_force = predictions["axial_true"][shown]
    pred_force = predictions["axial_pred"][shown]

    limit = max(
        float(np.max(np.abs(true_force))),
        float(np.max(np.abs(pred_force))),
        1e-8,
    )

    # -------------------------
    # Data for error histogram
    # -------------------------
    axial_error = (
        predictions["axial_pred"]
        - predictions["axial_true"]
    )

    # -------------------------
    # Figure
    # -------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5.5),
        constrained_layout=True,
    )

    # Parity
    ax = axes[0]
    ax.scatter(true_force, pred_force, s=8, alpha=0.35)
    ax.plot(
        [-limit, limit],
        [-limit, limit],
        linestyle="--",
        linewidth=1.5,
        color="red"
    )

    ax.set_xlabel("True axial force")
    ax.set_ylabel("Predicted axial force")
    ax.set_title(
        f"Edge axial force parity, test set (R²={r2:.3f})"
    )

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # Same x/y physical scale for parity
    ax.set_aspect("equal", adjustable="box")

    # Force physical panel to be square
    ax.set_box_aspect(1)

    # Error distribution
    ax = axes[1]
    ax.hist(axial_error, bins=bins)

    ax.set_xlabel("(Predicted − true) axial force")
    ax.set_ylabel("Edge count")
    ax.set_title("Axial force error distribution")

    ax.axvline(
        0.0,
        linestyle="--",
        color="red"
    )

    # Same physical panel dimensions as parity plot
    ax.set_box_aspect(1)

    plt.show()

    return fig


def plot_static_action_results(predictions, r2, bins=60):
    true_static = predictions["static_true"]
    pred_static = predictions["static_pred"]

    low = float(min(true_static.min(), pred_static.min()))
    high = float(max(true_static.max(), pred_static.max()))

    relative_error = (
        (pred_static - true_static)
        / np.maximum(np.abs(true_static), 1e-8)
    ) * 100.0

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5.5),
        constrained_layout=True,
    )

    # -------------------------
    # Parity
    # -------------------------
    ax = axes[0]

    ax.scatter(
        true_static,
        pred_static,
        s=18,
        alpha=0.55,
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1.5,
        color="red"
    )

    ax.set_xlabel("True static action")
    ax.set_ylabel("Static action readout")
    ax.set_title(
        f"Static action parity, test set (R²={r2:.3f})"
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    # -------------------------
    # Relative error
    # -------------------------
    ax = axes[1]

    ax.hist(relative_error, bins=bins)
    ax.axvline(
        0.0,
        linestyle="--",
        color="red"
    )

    ax.set_xlabel("Static action relative error (%)")
    ax.set_ylabel("Truss count")
    ax.set_title("Static action error distribution")

    ax.set_box_aspect(1)

    plt.show()

    return fig


def predict_single_graph(model, graph, device):
    model.eval()
    graph_device = graph.to(device)
    with torch.no_grad():
        output = model(graph_device)
    return output["axial_force"].detach().cpu().numpy()


def _add_member_field(axis, graph, values, title, norm, cmap):
    positions = graph.pos_raw.cpu().numpy()
    member_index = graph.member_index.cpu().numpy()
    segments = [positions[member_index[:, idx]] for idx in range(member_index.shape[1])]
    collection = LineCollection(segments, linewidths=3, cmap=cmap, norm=norm)
    collection.set_array(np.asarray(values))
    axis.add_collection(collection)
    axis.scatter(positions[:, 0], positions[:, 1], s=18, color="black")
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_title(title)
    return collection


def plot_member_force_comparison(graph, true_values, predicted_values, cmap="bwr_r"):
    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)
    shared_bound = max(
        float(np.max(np.abs(true_values))),
        float(np.max(np.abs(predicted_values))),
        1e-8,
    )
    norm = TwoSlopeNorm(vmin=-shared_bound, vcenter=0.0, vmax=shared_bound)
    figure, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    _add_member_field(axes[0], graph, true_values, "True member axial forces", norm, cmap)
    collection = _add_member_field(
        axes[1], graph, predicted_values, "Predicted edge axial forces", norm, cmap
    )
    figure.colorbar(collection, ax=axes, label="Axial force", shrink=0.9)
    plt.show()
    return figure


def plot_displacement_parity(true_nodes, predicted_nodes, true_magnitude, predicted_magnitude, plane):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    parity_items = [
        (true_nodes[:, 0], predicted_nodes[:, 0], f"U{plane[0]}"),
        (true_nodes[:, 1], predicted_nodes[:, 1], f"U{plane[1]}"),
        (true_magnitude, predicted_magnitude, "Magnitude"),
    ]

    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12

    for ax, (truth, prediction_values, label) in zip(axes, parity_items):

        ax.scatter(
            truth,
            prediction_values,
            s=8,
            alpha=0.35
        )

        lower = min(truth.min(), prediction_values.min())
        upper = max(truth.max(), prediction_values.max())

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            color="red"
        )

        ax.set_title(
            f"{label}: predicted vs true",
            fontsize=TITLE_FONTSIZE,
            pad=10
        )

        if label not in "Magnitude" :
            ax.set_xlabel(
                "True displacement",
                fontsize=LABEL_FONTSIZE
            )

            ax.set_ylabel(
                "Predicted displacement",
                fontsize=LABEL_FONTSIZE
            )
        else :
            ax.set_xlabel(
                "True magnitude",
                fontsize=LABEL_FONTSIZE
            )

            ax.set_ylabel(
                "Predicted magnitude",
                fontsize=LABEL_FONTSIZE
            )

        # Increase numerical tick labels
        ax.tick_params(
            axis="both",
            labelsize=TICK_FONTSIZE
        )

    plt.tight_layout()
    plt.show()
    return fig


def plot_displacement_residuals(true_nodes, predicted_nodes, true_magnitude, predicted_magnitude, plane, bins=60):
    residuals = predicted_nodes - true_nodes

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12

    for component, ax in enumerate(axes):

        ax.hist(
            residuals[:, component],
            bins=bins
        )

        ax.axvline(
            0.0,
            linestyle="--",
            color="red"
        )

        ax.set_title(
            f"Residual distribution for U{plane[component]}",
            fontsize=TITLE_FONTSIZE,
            pad=10
        )

        ax.set_xlabel(
            "(Predicted − true) displacement",
            fontsize=LABEL_FONTSIZE
        )

        ax.set_ylabel(
            "Free nodes",
            fontsize=LABEL_FONTSIZE
        )

        ax.tick_params(
            axis="both",
            labelsize=TICK_FONTSIZE
        )

    plt.tight_layout()
    plt.show()
    return fig



def plot_deformed_truss(graph, prediction, plane="xz", title="Test example"):
    position = graph.pos_raw.cpu().numpy()
    truth = graph.displacement.cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    physical_edges = graph.edge_index[:, 0::2].cpu().numpy().T

    span = np.ptp(position, axis=0).max()

    max_displacement = max(
        np.linalg.norm(truth, axis=1).max(),
        np.linalg.norm(prediction, axis=1).max(),
        1e-12,
    )

    # Slightly reduced deformation amplification
    amplification = 0.09 * span / max_displacement

    true_deformed = position + amplification * truth
    predicted_deformed = position + amplification * prediction

    fig, ax = plt.subplots(figsize=(11, 5))

    for source, target in physical_edges:
        # Undeformed structure: grey
        ax.plot(
            position[[source, target], 0],
            position[[source, target], 1],
            color="grey",
            linewidth=1.0,
            alpha=0.6,
        )

        # True deformation: solid black
        ax.plot(
            true_deformed[[source, target], 0],
            true_deformed[[source, target], 1],
            color="black",
            linewidth=1.8,
        )

        # Predicted deformation: dashed green
        ax.plot(
            predicted_deformed[[source, target], 0],
            predicted_deformed[[source, target], 1],
            color="green",
            linestyle="--",
            linewidth=1.8,
        )

    support = ~graph.free_mask.cpu().numpy()

    ax.scatter(
        position[support, 0],
        position[support, 1],
        marker="s",
        s=55,
        color="black",
        label="Supports",
    )

    # Legend handles
    ax.plot(
        [],
        [],
        color="grey",
        linewidth=1.0,
        label="Undeformed",
    )
    ax.plot(
        [],
        [],
        color="black",
        linewidth=1.8,
        label="True deformation",
    )
    ax.plot(
        [],
        [],
        color="green",
        linestyle="--",
        linewidth=1.8,
        label="Predicted deformation",
    )

    ax.set_aspect("equal")
    ax.set_title(
        f"{title} — deformation"
    )
    ax.set_xlabel(plane[0].upper())
    ax.set_ylabel(plane[1].upper())
    ax.legend()

    plt.tight_layout()
    plt.show()
