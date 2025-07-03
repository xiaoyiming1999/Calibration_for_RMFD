import numpy as np
from matplotlib import pyplot as plt


def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):

    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (num_bins > 0)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)

    return {"accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece}


def _reliability_diagram_subplot(ax, bin_data, x_label="Confidence", y_label="Expected Accuracy"):

    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    widths = bin_size
    positions = bins[:-1] + bin_size / 2.0

    acc_plt = ax.bar(positions, accuracies, bottom=0, width=widths,
                     edgecolor="black", color="#75bbfd", alpha=1, linewidth=2,
                     label="Accuracy")

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences),
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor='#e50000', color='#ff81c0', linewidth=2, label="Gap", alpha=0.5)

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=3)

    ece = (bin_data["expected_calibration_error"] * 100)
    bbox = {"facecolor": "white", "alpha": 1}
    ax.text(0.95, 0.05, "ECE=%.2f%%" % ece, color="black",
            ha="right", va="bottom", transform=ax.transAxes, size=35, bbox=bbox)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend(handles=[gap_plt, acc_plt])


def _reliability_diagram_combined(bin_data, fig_size, dpi):

    plt.rc('font', family='Times New Roman', size=22)

    figsize = (fig_size[0] * 1.1, fig_size[0] * 2)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi,
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.35, left=0.2, bottom=0.1)

    _reliability_diagram_subplot(ax[0], bin_data)

    # Draw the confidence histogram upside down
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax[1], bin_data)

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(np.int32)
    ax[1].set_yticklabels(new_ticks)

    # plt.savefig(r'C:\Users\xiaoyiming\Desktop\Papers\后验不确定性估计\figures\图8\FIG8d.svg', )

    plt.show()


def _confidence_histogram_subplot(ax, bin_data, x_label="Confidence", y_label="Count"):

    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size / 2.0

    ax.bar(positions, counts, width=bin_size, edgecolor="black", color="#75bbfd", alpha=1, linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=2,
                         c="black", label="Accuracy")
    conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=2,
                          c="#444", label="Avg. conf.")
    ax.legend(handles=[acc_plt, conf_plt], loc='lower left')


def _reliability_diagram(true_labels, pred_labels, confidences, num_bins=10, fig_size=(6, 6), dpi=100):

    bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)

    return _reliability_diagram_combined(bin_data, fig_size=fig_size, dpi=dpi)










