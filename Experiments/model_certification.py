from Models import Smooth

smoothing_sigmas = [.1, .25, .5, 1, 2]


def cert_model(model, dataset, sigma, n0, n, alpha, batch):
    smooth_model = Smooth(model, 10, sigma)
    c_hat = []
    corr = []
    radii = []
    for i in range(len(dataset)):
        sample = dataset[i][0]
        label = dataset[i][1]
        c, r = smooth_model.certify(sample, n0, n, alpha, batch)
        c_hat.append(c)
        radii.append(r)
        corr.append(c == label)
    return {'pred' : c_hat,
            'is_cor' : corr,
            'radii' : radii}

