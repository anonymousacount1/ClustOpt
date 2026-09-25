import matplotlib.pyplot as plt


def create_preview(X, y, feature_names, save_path):

    if X.shape[1] < 2:
        return

    plt.figure(figsize=(6, 6))

    labs = sorted(set(y))

    for lab in labs:
        mask = y == lab
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            s=3,
            label=str(lab),
            alpha=0.7
        )

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Dataset preview")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()