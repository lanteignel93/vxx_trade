import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import cm, colors
from scipy.stats import kurtosis, skew

from vxx_trade.data_generator import generate_data_for_strategy

plt.style.use("dark_background")

MATPLOTLIB_MAPPING = {
    "vix_cp": {
        "title": "VIX",
        "x_lims": (0, 100),
        "more": 1111,
        "more": 1111,
        "more": 1111,
        "more": 1111,
    },
    "vvix_cp": {
        "title": "VVIX",
        "x_lims": (50, 225),
        "more": 1111,
        "more": 1111,
        "more": 1111,
    },
    "vol_ts": {
        "title": "VIX/VIXM Term-Structure",
        "x_lims": (0, 2),
        "more": 1111,
        "more": 1111,
        "more": 1111,
    },
}


def main():
    scatter_plot_vxx_ret(column="vix_cp")
    scatter_plot_vxx_ret(column="vvix_cp")
    scatter_plot_vxx_ret(column="vol_ts")
    histogram_vxx_ret()


def histogram_vxx_ret():
    df = generate_data_for_strategy(verbose=False)

    x = df.drop_nulls().get_column("vxx_log_ret").to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    ax.hist(x, bins=bins)

    stats = {
        "mean": np.mean(x),
        "std": np.std(x),
        "median": np.median(x),
        "min": np.min(x),
        "max": np.max(x),
        "skew": skew(x),
        "kurtosis": kurtosis(x),
    }

    textstr = "\n".join(
        [
            r"Mean " + f"= {stats['mean']:.4f}",
            r"Std " + f"= {stats['std']:.4f}",
            r"Median " + f"= {stats['median']:.4f}",
            r"Min " + f"= {stats['min']:.4f}",
            r"Max " + f"= {stats['max']:.4f}",
            r"Skew " + f"= {stats['skew']:.4f}",
            r"Kurtosis " + f"= {stats['kurtosis']:.4f}",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=1)
    ax.text(
        0.6,
        0.85,
        textstr,
        transform=ax.transAxes,
        color="black",
        fontsize=16,
        verticalalignment="top",
        bbox=props,
    )
    ax.set_xticks(np.linspace(-0.2, 0.4, 13))

    print(bins)
    ax.set_xlabel("VXX Returns")
    ax.set_ylabel("VXX Returns Frequency Count")
    ax.set_title("VXX Returns Histogram 2012-2024")
    plt.savefig("hist_test.png")


def scatter_plot_vxx_ret(column: str):
    title = MATPLOTLIB_MAPPING[column]["title"]
    x_lims = MATPLOTLIB_MAPPING[column]["x_lims"]

    df = generate_data_for_strategy(verbose=False)
    df = df.with_columns(pl.col("date").dt.year().alias("year"))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.8)
    ax.scatter(
        x=df.get_column(column),
        y=df.get_column("vxx_log_ret"),
        c=df.get_column("year"),
        cmap="Spectral",
    )
    tmp = df.select([column, "vxx_log_ret"]).drop_nulls()
    x = tmp.get_column(column).to_numpy()
    y = tmp.get_column("vxx_log_ret").to_numpy()
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    x_graph = np.linspace(x_lims[0], x_lims[1], 1000)
    y_pred = c + m * x_graph
    ax.plot(x_graph, y_pred, color="white", ls="--", lw=2)

    y_pred = c + m * x
    ss_tot = np.power(y - y.mean(), 2).sum()
    ss_res = np.power(y - y_pred, 2).sum()
    r2 = 1 - ss_res / ss_tot

    textstr = "\n".join(
        [
            r"$R^2$ " + f"= {r2:.4f}",
            r"$\alpha$ " + f"= {c:.4f}",
            r"$\beta$ " + f"= {m:.4f}",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=1)
    ax.text(
        0.05,
        0.15,
        textstr,
        transform=ax.transAxes,
        color="black",
        fontsize=16,
        verticalalignment="top",
        bbox=props,
    )

    cmap = plt.get_cmap("Spectral")
    norm = colors.Normalize(2012, 2024)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(-0.4, 0.4)
    ax.set_xticks(np.linspace(x_lims[0], x_lims[1], 11))
    ax.set_yticks(np.linspace(-0.4, 0.4, 9))

    ax.set_xlabel(f"{title} Level")
    ax.set_ylabel("VXX Returns")
    ax.set_title(f"VXX Returns versus {title} Level 2012-2024")
    plt.savefig("scatter_test.png")


if __name__ == "__main__":
    main()

