from sklearn.cluster import KMeans, AgglomerativeClustering


def perform_kmeans(X, k=3):

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)

    return labels


def perform_hierarchical(X, k=3):

    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)

    return labels