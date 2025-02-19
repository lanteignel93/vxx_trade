import polars as pl
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from vxx_trade.data_generator import generate_data_for_strategy


def foo():
    datagen = generate_data_for_strategy(verbose=False)
    df = datagen()

    features = [
        "volume",
        "vix_cp",
        "vvix_cp",
        "vol_ts",
        "vix_cp_ewma_zscore",
        "vvix_cp_ewma_zscore",
        "vol_ts_ewma_zscore",
    ]

    cols_to_keep = features + [
        "date",
        "vix_cp_rank",
        "vol_ts_rank",
        "vvix_cp_rank",
        "vix_cp_zscore_bucket",
        "vvix_cp_zscore_bucket",
        "vol_ts_zscore_bucket",
        "cc_ret",
    ]

    scaling_features = features[:4]

    df = df.with_columns(
        [(pl.col(name) - pl.mean(name)) / pl.std(name) for name in scaling_features]
    )
    df = df.select(cols_to_keep).drop_nulls()
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(df.select(features).to_numpy())
    df = df.with_columns(pl.Series(kmeans.labels_.astype("int")).alias("KMeansCluster"))

    hierarchical_clustering = AgglomerativeClustering(n_clusters=10)
    hierarchical_clustering.fit(df.select(features).to_numpy())
    df = df.with_columns(
        pl.Series(hierarchical_clustering.labels_.astype("int")).alias(
            "HierarchicalCluster"
        )
    )

    gmm = GaussianMixture(n_components=10, random_state=42)
    gmm.fit(df.select(features).to_numpy())
    df = df.with_columns(
        pl.Series(gmm.predict(df.select(features).to_numpy()).astype("int")).alias(
            "GMMCluster"
        )
    )

    return df


def plot_clusters(df):
    cluster_cols = ["KMeansCluster", "HierarchicalCluster", "GMMCluster", "vix_cp_rank"]
    df = foo()

    for col in cluster_cols:
        df.group_by(col).agg(
            [
                pl.std("cc_ret").alias("std_cc_ret"),
                (pl.col("cc_ret").abs() * 252**0.5)
                .mean()
                .alias("annualized_cc_ret_vol"),
                pl.col("cc_ret").count().alias("GroupCount"),
            ]
        ).sort("std_cc_ret").show()


def compute_ret_sharpe(df):
    cluster_cols = ["KMeansCluster", "HierarchicalCluster", "GMMCluster", "vix_cp_rank"]

    df = df.with_columns(
        [
            (pl.col("cc_ret") * 252 / (pl.col("cc_ret").abs() * 252**0.5).mean())
            .alias(f"{col}_sharpe")
            .over(col)
            for col in cluster_cols
        ]
    )

    return df


if __name__ == "__main__":
    df = foo()
    plot_clusters(df)
