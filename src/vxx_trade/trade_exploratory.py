from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import cm, colors
from scipy.stats import kurtosis, skew

from vxx_trade import EXPLORATORY_PARAMETERS, MATPLOTLIB_EXPLORATORY, MATPLOTLIB_STYLE
from vxx_trade._utils import MatplotlibAxesLimit, MatplotlibFigSize, YearsResearch
from vxx_trade.data_generator import DataGenerator, generate_data_for_strategy

plt.style.use(MATPLOTLIB_STYLE["style"])


@dataclass
class TradeExploratoryParameters:
    target_title: str
    target_col: str
    figsize: MatplotlibFigSize
    years: YearsResearch
    y_lims: MatplotlibAxesLimit
    y_tick_count: int
    data_generator: DataGenerator


class TradeExploratory(TradeExploratoryParameters):
    def __init__(
        self,
        parameters: TradeExploratoryParameters,
        target_title: str | None = None,
        target_col: str | None = None,
        figsize: MatplotlibFigSize | list | tuple | None = None,
        years: YearsResearch | list | tuple | None = None,
        y_lims: MatplotlibFigSize | list | tuple | None = None,
        y_tick_count: int | None = None,
        data_generator: DataGenerator | None = None,
    ):
        super().__init__(**asdict(parameters))

        self._handle_data_generator(parameters.data_generator)
        self._handle_cmap()

        if target_title is not None:
            self.target_title = target_title

        if target_col is not None:
            self.target_col = target_col

        if figsize is not None:
            if isinstance(figsize, list) or isinstance(figsize, tuple):
                self.figsize = MatplotlibFigSize(*figsize)
            self.figsize = figsize

        if years is not None:
            if isinstance(years, list) or isinstance(years, tuple):
                self.years = YearsResearch(*years)

            self.years = years

        if y_lims is not None:
            if isinstance(y_lims, list) or isinstance(y_lims, tuple):
                self.y_lim = MatplotlibAxesLimit(*y_lims)

            self.y_lims = y_lims

        if y_tick_count is not None:
            self.y_tick_count = y_tick_count

        if data_generator is not None:
            self._handle_data_generator(data_generator)

    def _handle_data_generator(self, data_generator: DataGenerator) -> None:
        self.df = data_generator.df
        self.zscore_period = data_generator.zscore_period
        self.analysis_columns = data_generator._volatility_columns

    def _handle_cmap(self) -> None:
        self.cmap_name = MATPLOTLIB_STYLE["cmap"]
        self.cmap = plt.get_cmap(self.cmap_name)
        return

    def barplot_vxx_ret_zscore(self, column: str):
        title = MATPLOTLIB_EXPLORATORY[column]["title"]

        tmp = (
            self.df.select([self.target_col, f"{column}_zscore_bucket"])
            .drop_nulls()
            .group_by(f"{column}_zscore_bucket")
            .mean()
            .sort(f"{column}_zscore_bucket")
        )

        fig, ax = plt.subplots(figsize=(self.figsize.width, self.figsize.height))
        colours = self.cmap(np.linspace(0, 1, tmp.shape[0])[::-1])
        ax.bar(
            tmp.get_column(f"{column}_zscore_bucket"),
            tmp.get_column(self.target_col),
            color=colours,
        )
        ax.set_xlabel(f"Zscore bucket of {title} - EWMA {title}")
        ax.set_ylabel(f"Average {self.target_title} Returns in bucket.")
        ax.set_title(
            f"Average {self.target_title} Returns per different {title} - EWMA {title} {self.zscore_period}days Zscore buckets"
        )

        plt.savefig(f"{title.lower().replace('/','_').replace(' ', '_')}_zscore.png")

    def barplot_vxx_ret_decile(self, column: str):
        title = MATPLOTLIB_EXPLORATORY[column]["title"]

        tmp = (
            self.df.select([f"{self.target_col}", f"{column}_rank", column])
            .drop_nulls()
            .group_by(f"{column}_rank")
            .mean()
            .sort(f"{column}_rank")
        )

        x_label = [f"{x:.2f}" for x in tmp.get_column(column).to_numpy()]

        fig, ax = plt.subplots(figsize=(self.figsize.width, self.figsize.height))
        colours = self.cmap(np.linspace(0, 1, tmp.shape[0])[::-1])
        ax.bar(x_label, tmp.get_column(self.target_col), color=colours)
        ax.set_xlabel(f"Average {title} per rank")
        ax.set_ylabel(f"Average {self.target_title} Returns in bucket.")
        ax.set_title(
            f"Average {self.target_title} Returns per different {title} buckets"
        )

        plt.savefig(f"{title.lower().replace('/','_').replace(' ', '_')}_decile.png")

    def histogram_vxx_ret(self):
        x = self.df.drop_nulls().get_column(self.target_col).to_numpy()
        fig, ax = plt.subplots(figsize=(self.figsize.width, self.figsize.height))
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
        ax.set_xticks(np.linspace(self.y_lims.min, self.y_lims.max, self.y_tick_count))

        ax.set_xlabel(f"{self.target_title} Returns")
        ax.set_ylabel(f"{self.target_title} Returns Frequency Count")
        ax.set_title(
            f"{self.target_title} Returns Histogram {self.years[0]}-{self.years[1]}"
        )
        plt.savefig(f"{self.target_col.lower()}_histogram.png")

    def scatter_plot_vxx_ret(self, column: str):
        title = MATPLOTLIB_EXPLORATORY[column]["title"]
        x_lims = MATPLOTLIB_EXPLORATORY[column]["x_lims"]

        df = self.df
        df = df.with_columns(pl.col("date").dt.year().alias("year"))

        fig, ax = plt.subplots(figsize=(self.figsize.width, self.figsize.height))
        fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.8)
        ax.scatter(
            x=df.get_column(column),
            y=df.get_column(self.target_col),
            c=df.get_column("year"),
            cmap=self.cmap_name,
        )
        tmp = df.select([column, self.target_col]).drop_nulls()
        x = tmp.get_column(column).to_numpy()
        y = tmp.get_column(self.target_col).to_numpy()
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        x_graph = np.linspace(x_lims.min, x_lims.max, 1000)
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

        norm = colors.Normalize(self.years[0], self.years[1])

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=self.cmap), ax=ax)

        ax.set_xlim(x_lims[0], x_lims[1])
        ax.set_ylim(self.y_lims.min, self.y_lims.max)
        ax.set_xticks(np.linspace(x_lims.min, x_lims.max, 11))
        ax.set_yticks(np.linspace(self.y_lims.min, self.y_lims.max, self.y_tick_count))

        ax.set_xlabel(f"{title} Level")
        ax.set_ylabel(f"{self.target_title} Returns")
        ax.set_title(
            f"{self.target_title} Returns versus {title} Level {self.years[0]}-{self.years[1]}"
        )
        plt.savefig(f"scatter_{title.lower().replace('/','_').replace(' ', '_')}.png")

    def plot_analysis(self):
        self.histogram_vxx_ret()
        for col in self.analysis_columns:
            self.scatter_plot_vxx_ret(column=col)
            self.barplot_vxx_ret_decile(column=col)
            self.barplot_vxx_ret_zscore(column=col)


def main():
    data_gen = generate_data_for_strategy(verbose=False)
    EXPLORATORY_PARAMETERS["data_generator"] = data_gen
    exploratory_parameters = TradeExploratoryParameters(**EXPLORATORY_PARAMETERS)
    trade_explorator = TradeExploratory(parameters=exploratory_parameters)
    trade_explorator.plot_analysis()


if __name__ == "__main__":
    main()
