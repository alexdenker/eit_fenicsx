"""
Written by KTC2023 challenge organizers:
https://zenodo.org/record/8252370

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps

def Otsu(image, nvals, figno):
    # binary Otsu's method for finding the segmentation level for sigma
    histogramCounts, x = np.histogram(image.ravel(), nvals)
    # plt.figure(figno)
    # plt.clf()
    # plt.hist(image.ravel(), 256)
    # plt.hold(True)

    total = np.sum(histogramCounts)
    top = 256
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.dot(np.arange(top), histogramCounts)
    for ii in range(1, top):
        wF = total - wB
        if wB > 0 and wF > 0:
            mF = (sum1 - sumB) / wF
            val = wB * wF * (((sumB / wB) - mF) ** 2)
            if val >= maximum:
                level = ii
                maximum = val
        wB = wB + histogramCounts[ii]
        sumB = sumB + (ii - 1) * histogramCounts[ii]

    # plt.plot([x[level]] * 2, [0, np.max(histogramCounts)], linewidth=2, color='r')
    # plt.title('histogram of image pixels')
    # plt.gcf().set_size_inches(9, 5)
    # plt.show()

    return level, x

def Otsu2(image, nvals, figno):
    # three class Otsu's method to find the semgentation point of sigma
    histogramCounts, tx = np.histogram(image.ravel(), nvals)
    x = (tx[0:-1] + tx[1:])/2
    # plt.figure(figno)
    # plt.clf()
    # plt.stairs(histogramCounts, tx)
    # plt.hold(True)

    #total = np.sum(histogramCounts)
    #top = 256
    maximum = 0.0
    muT = np.dot(np.arange(1, nvals+1), histogramCounts) / np.sum(histogramCounts)
    for ii in range(1, nvals):
        for jj in range(1, ii):
            w1 = np.sum(histogramCounts[:jj])
            w2 = np.sum(histogramCounts[jj:ii])
            w3 = np.sum(histogramCounts[ii:])
            if w1 > 0 and w2 > 0 and w3 > 0:
                mu1 = np.dot(np.arange(1, jj+1), histogramCounts[:jj]) / w1
                mu2 = np.dot(np.arange(jj+1, ii+1), histogramCounts[jj:ii]) / w2
                mu3 = np.dot(np.arange(ii+1, nvals+1), histogramCounts[ii:]) / w3

                val = w1 * ((mu1 - muT) ** 2) + w2 * ((mu2 - muT) ** 2) + w3 * ((mu3 - muT) ** 2)
                if val >= maximum:
                    level = [jj-1, ii-1]
                    maximum = val

    # plt.plot([x[level[0]]] * 2, [0, np.max(histogramCounts)], linewidth=2, color='r')
    # plt.plot([x[level[1]]] * 2, [0, np.max(histogramCounts)], linewidth=2, color='r')
    # plt.title('histogram of image pixels')
    # plt.gcf().set_size_inches(9, 5)
    # plt.show()

    return level, x

def scoringFunction(groundtruth, reconstruction):
    
    if (np.any(groundtruth.shape != np.array([256, 256]))):
        raise Exception('The shape of the given ground truth is not 256 x 256!')
    
    if (np.any(reconstruction.shape != np.array([256, 256]))):
        return 0
    
    truth_c = np.zeros(groundtruth.shape)
    truth_c[np.abs(groundtruth - 2) < 0.1] = 1
    reco_c = np.zeros(reconstruction.shape)
    reco_c[np.abs(reconstruction - 2) < 0.1] = 1

    score_c = ssim(truth_c, reco_c)

    truth_d = np.zeros(groundtruth.shape)
    truth_d[np.abs(groundtruth - 1) < 0.1] = 1
    reco_d = np.zeros(reconstruction.shape)
    reco_d[np.abs(reconstruction - 1) < 0.1] = 1

    score_d = ssim(truth_d, reco_d)

    score = 0.5*(score_c + score_d)

    return score


def ssim(truth, reco):

    c1 = 1e-4
    c2 = 9e-4
    r = 80

    ws = np.ceil(2*r)
    wr = np.arange(-ws, ws+1)
    X, Y = np.meshgrid(wr, wr)
    ker = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * np.divide((np.square(X) + np.square(Y)), r**2))
    correction = sps.convolve2d(np.ones(truth.shape), ker, mode='same')

    gt = np.divide(sps.convolve2d(truth, ker, mode='same'), correction)
    gr = np.divide(sps.convolve2d(reco, ker, mode='same'), correction)

    mu_t2 = np.square(gt)
    mu_r2 = np.square(gr)
    mu_t_mu_r = np.multiply(gt, gr)

    sigma_t2 = np.divide(sps.convolve2d(np.square(truth), ker, mode='same'), correction) - mu_t2
    sigma_r2 = np.divide(sps.convolve2d(np.square(reco), ker, mode='same'), correction) - mu_r2
    sigma_tr = np.divide(sps.convolve2d(np.multiply(truth, reco), ker, mode='same'), correction) - mu_t_mu_r;

    num = np.multiply((2*mu_t_mu_r + c1), (2*sigma_tr + c2))
    den = np.multiply((mu_t2 + mu_r2 + c1), (sigma_t2 + sigma_r2 + c2))
    ssimimage = np.divide(num, den)

    score = np.mean(ssimimage)

    return score

