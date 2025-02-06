import numpy as np
from scipy.interpolate import interpn, NearestNDInterpolator
from skimage.filters import threshold_multiotsu

from dolfinx.fem import assemble_scalar, form
import ufl


def compute_relative_l1_error(sigma_rec, sigma_gt):
    diff = abs(sigma_rec - sigma_gt) * ufl.dx
    diff = assemble_scalar(form(diff))

    norm = abs(sigma_gt) * ufl.dx
    norm = assemble_scalar(form(norm))

    return diff / norm




def multi_level_otsu_segmentation(image):
    # Apply thresholding
    thresholds = threshold_multiotsu(image, classes=3)
    #print(thresholds)
    # Segment image into three classes based on the thresholds
    class0 = image < thresholds[0]  # Lower conductivity
    class1 = (image >= thresholds[0]) & (image < thresholds[1])  # Background
    class2 = image >= thresholds[1]  # Higher conductivity
    
    # However, sometimes this does not work and class1 is actually the background class
    # We work with the heuristic that the class with the most number of elements is the background
    inds = [np.count_nonzero(class0),np.count_nonzero(class1),np.count_nonzero(class2)]
    bgclass = inds.index(max(inds)) #background class

    segmented_image = np.zeros_like(image)
    if bgclass == 0: # background is class0, join class1 + class2 for higher
        segmented_image[class0] = 0  
        segmented_image[class1] = 1
        segmented_image[class2] = 1  
    elif bgclass == 1: # background is class1, then class0 is lower, class2 is higher
        segmented_image[class0] = -1  
        segmented_image[class1] = 0
        segmented_image[class2] = 1  
    elif bgclass == 2: # background is 2, join class0 + class1 for lower
        segmented_image[class0] = -1  
        segmented_image[class1] = -1
        segmented_image[class2] = 0
    else:
        print("Segmentation failed, return zero image")

    return segmented_image, thresholds

def compute_dice_per_class(pred, gt):
    dice_scores = []

    class_sizes = []
    for class_id in [-1, 0, 1]:
        # Create binary masks for each class
        pred_class = (pred == class_id).astype(np.uint8)
        gt_class = (gt == class_id).astype(np.uint8)
        class_sizes.append(np.count_nonzero(gt_class))
        # Compute intersection and sum of predictions/ground truth
        intersection = np.sum(pred_class * gt_class)
        dice_score = 2 * intersection / (np.sum(pred_class) + np.sum(gt_class))

        # If this sum is zero, we do not have this class in this image
        if np.sum(pred_class) + np.sum(gt_class) > 0.0:
            dice_scores.append(dice_score)
    

    return np.mean(dice_scores) 

def mean_dice_score(pred, gt, backCond):
    """
    Compute the mean dice score over classes. 
    class = -1 : lower conductivity 
    class = 0  : background 
    class = 1  : higher conductivity 
    
    This method takes in the predicted conductivity values, performs a 
    segmentation using Otsus method and computes the mean dice score.

    We segment the ground truth using the knowledge of the background conductivity. 

    Note: This method is not batched 
    TODO: Deal with batches.

    pred: predicted conductivity values, np.array 
    gt: groundtruth conductivity values, np.array
    """

    pred_seg, _ = multi_level_otsu_segmentation(pred.flatten())

    # segment ground truth 
    class0 = gt.flatten() < (backCond - 0.2)   # Lower conductivity
    class1 = (gt.flatten() >= (backCond - 0.2)) & (gt.flatten() < (backCond + 0.2))  # Background
    class2 = gt.flatten() >= (backCond + 0.2)  # Higher conductivity

            
    segmented_gt = np.zeros_like(gt.flatten())
    segmented_gt[class0] = -1
    segmented_gt[class1] = 0 
    segmented_gt[class2] = 1

    dice_score = compute_dice_per_class(pred_seg, segmented_gt)

    return dice_score

def image_to_mesh(x, mesh_pos):
    radius = np.max(np.abs(mesh_pos))

    pixcenter_x = pixcenter_y = np.linspace(-radius, radius, 256)
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")

    sigma = interpn(
        [pixcenter_x, pixcenter_y],
        np.flipud(x).T,
        mesh_pos,
        bounds_error=False,
        fill_value=1.0,
        method="nearest",
    )

    return sigma


def interpolate_mesh_to_mesh(x, mesh_pos1, mesh_pos2):
    interpolator = NearestNDInterpolator(mesh_pos1, x)

    sigma = interpolator(mesh_pos2[:, 0], mesh_pos2[:, 1])

    return sigma


def current_method(L, l, method=1, value=1):
    """
    Create a numpy array (or a list of arrays) that represents the current pattern in the electrodes.

    Taken from: https://github.com/HafemannE/FEIT_CBM34/blob/main/CBM/FEIT_codes/FEIT_onefile.py


    :param L: Number of electrodes.
    :type L: int
    :param l: Number of measurements.
    :type l: int
    :param method: Current pattern. Possible values are 1, 2, 3, or 4 (default=1).
    :type method: int
    :param value: Current density value (default=1).
    :type value: int or float

    :returns: list of arrays or numpy array -- Return list with current density in each electrode for each measurement.

    :Method Values:
        1. 1 and -1 in opposite electrodes.
        2. 1 and -1 in adjacent electrodes.
        3. 1 in one electrode and -1/(L-1) for the rest.
        4. For measurement k, we have: (sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16)).
        5. All against 1

    :Example:

    Create current pattern 1 with 4 measurements and 4 electrodes:

    >>> I_all = current_method(L=4, l=4, method=1)
    >>> print(I_all)
        [array([ 1.,  0., -1.,  0.]),
        array([ 0.,  1.,  0., -1.]),
        array([-1.,  0.,  1.,  0.]),
        array([ 0., -1.,  0.,  1.])]

    Create current pattern 2 with 4 measurements and 4 electrodes:

    >>> I_all = current_method(L=4, l=4, method=2)
    >>> print(I_all)
        [array([ 1., -1.,  0.,  0.]),
        array([ 0.,  1., -1.,  0.]),
        array([0.,  0.,  1., -1.]),
        array([ 1.,  0.,  0., -1.])]

    """
    I_all = []
    # Type "(1,0,0,0,-1,0,0,0)"
    if method == 1:
        if L % 2 != 0:
            raise Exception("L must be odd.")

        for i in range(l):
            if i <= L / 2 - 1:
                I = np.zeros(L)
                I[i], I[i + int(L / 2)] = value, -value
                I_all.append(I)
            elif i == L / 2:
                print(
                    "This method only accept until L/2 currents, returning L/2 currents."
                )
    # Type "(1,-1,0,0...)"
    if method == 2:
        for i in range(l):
            if i != L - 1:
                I = np.zeros(L)
                I[i], I[i + 1] = value, -value
                I_all.append(I)
            else:
                I = np.zeros(L)
                I[0], I[i] = -value, value
                I_all.append(I)
    # Type "(1,-1/15, -1/15, ....)"
    if method == 3:
        for i in range(l):
            I = np.ones(L) * -value / (L - 1)
            I[i] = value
            I_all.append(I)
    # Type "(sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16))"
    if method == 4:
        for i in range(l):
            I = np.ones(L)
            for k in range(L):
                I[k] = I[k] * np.sin((i + 1) * (k + 1) * 2 * np.pi / L)
            I_all.append(I)

    if method == 5:
        for i in range(l):
            if i <= L - 1:
                I = np.zeros(L)
                I[0] = -value
                I[i + 1] = value
                I_all.append(I)
            else:
                print(
                    "This method only accept until L-1 currents, returning L-1 currents."
                )

    if l == 1:
        I_all = I_all[0]
    return np.array(I_all)


if __name__ == "__main__":
    Inj_ref = current_method(L=16, l=16, method=2)

    print(Inj_ref)
    print(Inj_ref.shape)
