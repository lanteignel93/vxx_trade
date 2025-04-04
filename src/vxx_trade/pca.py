import numpy as np 
import polars as pl
from sklearn.decomposition import (
    PCA,
    KernelPCA,
    SparsePCA
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import Isomap
from vxx_trade.data_generator import generate_data_for_strategy

# def sparse_pca():
#     datagen = generate_data_for_strategy()
#     df = datagen() 
#
#     features = ['vvix_cp', 'vvix_cp_ewma_zscore']
#
#     df = df.drop_nulls().sample(fraction=1)
#     df_test = df.tail(-2000)
#     train = df.head(2000)
#     test = df.tail(-2000)
#
#     X_train = train.select(features)
#
#     X_test = test.select(features)
#
#     X_train = X_train.to_numpy()
#     X_test = X_test.to_numpy()
#
#     spca = SparsePCA(
#         n_components=X_train.ndim,
#         alpha=1,
#         ridge_alpha=0.1,
#         method='lars'
#     ) 
#     spca.fit(X_train)
#     
#     train_spca_output = spca.transform(X_train)[:, 0]
#     test_spca_output = spca.transform(X_test)[:, 0]
#
#     print(train_spca_output, test_spca_output, sep='\n')

# def kernel_pca():
#     datagen = generate_data_for_strategy()
#     df = datagen() 
#
#     features = ['vvix_cp', 'vvix_cp_ewma_zscore']
#
#     df = df.drop_nulls().sample(fraction=1)
#     df_test = df.tail(-2000)
#     train = df.head(2000)
#     test = df.tail(-2000)
#
#     X_train = train.select(features)
#
#     X_test = test.select(features)
#
#     X_train = X_train.to_numpy()
#     X_test = X_test.to_numpy()
#
#     kpca = KernelPCA(n_components=X_train.ndim, kernel='poly', degree=3)
#     kpca.fit(X_train)
#
#     train_kpca_output = kpca.transform(X_train)[:, 0]
#     test_kpca_output = kpca.transform(X_test)[:, 0]
#
#     print(train_kpca_output, test_kpca_output, sep='\n')



def isomap():
    datagen = generate_data_for_strategy()
    df = datagen() 

    features = ['vvix_cp', 'vvix_cp_ewma_zscore']

    df = df.drop_nulls().sample(fraction=1)
    df_test = df.tail(-2000)
    train = df.head(2000)
    test = df.tail(-2000)

    X_train = train.select(features)

    X_test = test.select(features)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    n_iso_components = X_train.ndim 
    
    reconstruction_errors = {}
    for k in range(4,12):
        try:
            iso = Isomap(n_neighbors=k,
                         n_components=n_iso_components,
                         n_jobs=-1)
            iso.fit(X_train)
            reconstruction_errors[k] = iso.reconstruction_error()
        except Exception as e:
            print(f"    k={k}, Failed: {e}")
            reconstruction_errors[k] = float('inf')

    best_k = min(reconstruction_errors, key=reconstruction_errors.get)

    iso = Isomap(
        n_neighbors=best_k,
        n_components=n_iso_components,
        n_jobs=-1
    )
    iso.fit(X_train)
    train_iso_output = iso.transform(X_train)[:, 0] 
    test_iso_output = iso.transform(X_test)[:, 0]

    print(train_iso_output, test_iso_output, sep='\n')

# def pls():
#     datagen = generate_data_for_strategy()
#     df = datagen() 
#
#     features = ['vvix_cp', 'vvix_cp_ewma_zscore']
#
#     df = df.drop_nulls().sample(fraction=1)
#     df_test = df.tail(-2000)
#     target = 'vvix_cp_rank'
#     train = df.head(2000)
#     test = df.tail(-2000)
#
#     y_train = train.get_column(target)
#     X_train = train.select(features)
#
#     y_test = test.get_column(target)
#     X_test = test.select(features)
#
#     X_train = X_train.to_numpy()
#     X_test = X_test.to_numpy()
#
#     pls = PLSRegression(n_components=X_train.ndim, scale=False)
#     pls.fit(X_train, y_train)
#
#     train_pls_output = pls.transform(X_train)[:, ]
#     test_pls_output = pls.transform(X_test)[:, 0]
#
#     print(train_pls_output, test_pls_output, sep='\n')


# def pca():  
#     datagen = generate_data_for_strategy()
#     df = datagen() 
#     features = ['vvix_cp', 'vvix_cp_ewma_zscore']
#
#     df = df.drop_nulls().sample(fraction=1)
#     df_test = df.tail(-2000)
#     X_train = df.select(features).head(2000)
#     X_test = df.select(features).tail(-2000)
#
#     X_train = X_train.to_numpy()
#     X_test = X_test.to_numpy()
#
#     pca = PCA(n_components=X_train.ndim)
#     pca.fit(X_train)
#
#     train_pca_output = pca.transform(X_train)[:, 0]
#     test_pca_output = pca.transform(X_test)[:, 0]
#
#     pca_feature_name = features[0].split("_")[0] + '_pca_output'
#
#     df_test = df_test.with_columns(pl.Series(name=pca_feature_name, values=test_pca_output))
#     print(df_test.tail())


# def lda():
#     datagen = generate_data_for_strategy()
#     df = datagen() 
#
#     features = ['vvix_cp', 'vvix_cp_ewma_zscore']
#
#     df = df.drop_nulls().sample(fraction=1)
#     df_test = df.tail(-2000)
#     target = 'vvix_cp_rank'
#     train = df.head(2000)
#     test = df.tail(-2000)
#
#     y_train = train.get_column(target)
#     X_train = train.select(features)
#
#     y_test = test.get_column(target)
#     X_test = test.select(features)
#
#     X_train = X_train.to_numpy()
#     X_test = X_test.to_numpy()
#
#     lda = LinearDiscriminantAnalysis(solver='eigen')
#     lda.fit(X_train, y_train)
#
#     lda_train_output = lda.transform(X_train)[:, 0]
#     lda_test_output = lda.transform(X_test)[:, 0]
#     
#     lda_feature_name = features[0].split("_")[0] + "_lda_output"
#
#     df_test = df_test.with_columns(pl.Series(name=lda_feature_name, values=lda_test_output))
#     print(df_test.tail())
#
    
sparse_pca()
