

import numpy as np 
import scipy.signal as sps

def Otsu2(image, nvals, figno):
    # three class Otsu's method to find the semgentation point of sigma
    histogramCounts, tx = np.histogram(image.ravel(), nvals)
    x = (tx[0:-1] + tx[1:])/2
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
    return level, x


def segment_recon_otsu(recon):
    # threshold the image histogram using Otsu's method
    level, x = Otsu2(recon.flatten(), 256, 7)

    deltareco_pixgrid_segmented = np.zeros_like(recon)

    ind0 = recon < x[level[0]]
    ind1 = np.logical_and(recon >= x[level[0]],recon <= x[level[1]])
    ind2 = recon > x[level[1]]
    inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
    bgclass = inds.index(max(inds)) #background class

    match bgclass:
        case 0:
            deltareco_pixgrid_segmented[ind1] = 2
            deltareco_pixgrid_segmented[ind2] = 2
        case 1:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind2] = 2
        case 2:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind1] = 1
    return deltareco_pixgrid_segmented

def FastScoringFunction(groundtruth, reconstruction):
    if (np.any(groundtruth.shape != np.array([256, 256]))):
        raise Exception('The shape of the given ground truth is not 256 x 256!')
    if (np.any(reconstruction.shape != np.array([256, 256]))):
        return 0
    truth_c = np.zeros(groundtruth.shape)
    truth_c[np.abs(groundtruth - 2) < 0.1] = 1
    reco_c = np.zeros(reconstruction.shape)
    reco_c[np.abs(reconstruction - 2) < 0.1] = 1
    score_c = ssim(truth_c, reco_c)#, data_range = 1)
    truth_d = np.zeros(groundtruth.shape)
    truth_d[np.abs(groundtruth - 1) < 0.1] = 1
    reco_d = np.zeros(reconstruction.shape)
    reco_d[np.abs(reconstruction - 1) < 0.1] = 1
    score_d = ssim(truth_d, reco_d)#, data_range = 1)
    score = 0.5*(score_c + score_d)
    return score

def ssim(truth, reco):
    c1 = 1e-4
    c2 = 9e-4
    r = 80
    ws = np.ceil(2*r)
    wr = np.arange(-ws, ws+1)
    ker =  (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * np.divide(wr**2, r**2))[None]
    correction = convolve2d(np.ones(truth.shape), ker, mode='same')
    gt = np.divide(convolve2d(truth, ker, mode='same'), correction)
    gr = np.divide(convolve2d(reco, ker, mode='same'), correction)
    mu_t2 = np.square(gt)
    mu_r2 = np.square(gr)
    mu_t_mu_r = np.multiply(gt, gr)
    sigma_t2 = np.divide(convolve2d(np.square(truth), ker, mode='same'), correction) - mu_t2
    sigma_r2 = np.divide(convolve2d(np.square(reco), ker, mode='same'), correction) - mu_r2
    sigma_tr = np.divide(convolve2d(np.multiply(truth, reco), ker, mode='same'), correction) - mu_t_mu_r
    num = np.multiply((2*mu_t_mu_r + c1), (2*sigma_tr + c2))
    den = np.multiply((mu_t2 + mu_r2 + c1), (sigma_t2 + sigma_r2 + c2))
    ssimimage = np.divide(num, den)
    score = np.mean(ssimimage)
    return score

def convolve2d(img, kernel, mode):
    return sps.convolve(sps.convolve(img, kernel, mode=mode), kernel.T, mode=mode)




