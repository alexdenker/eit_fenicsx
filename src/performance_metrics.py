from abc import ABC, abstractmethod

from dolfinx.fem import assemble_scalar, form
from skimage.filters import threshold_multiotsu
import ufl
import numpy as np


class PerformanceMetric(ABC):
    def __init__(self, name="default"):
        self.name = name

    @abstractmethod
    def __call__(self, sigma_pred, sigma_gt):
        """
        Enforce that all subclasses implement the `__call__` method.

        """
        pass


class RelativeL1Error(PerformanceMetric):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, sigma_pred, sigma_gt):
        """
        sigma_pred: prediction, FenicsX function
        sigma_gt: ground truth, FenicsX function

        """

        diff = abs(sigma_pred - sigma_gt) * ufl.dx
        diff = assemble_scalar(form(diff))

        norm = abs(sigma_gt) * ufl.dx
        norm = assemble_scalar(form(norm))

        return diff / norm


class RelativeL2Error(PerformanceMetric):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, sigma_pred, sigma_gt):
        """
        sigma_pred: prediction, FenicsX function
        sigma_gt: ground truth, FenicsX function

        """

        diff = abs(sigma_pred - sigma_gt)**2 * ufl.dx
        diff = assemble_scalar(form(diff))

        norm = abs(sigma_gt)**2 * ufl.dx
        norm = assemble_scalar(form(norm))

        return diff / norm


class DiceScore(PerformanceMetric):
    def __init__(self, name, backCond: float):
        super().__init__(name)

        self.backCond = backCond  # background conductivity

    def __call__(self, sigma_pred, sigma_gt):
        """
        Compute the mean dice score over classes.
        class = -1 : lower conductivity
        class = 0  : background
        class = 1  : higher conductivity

        This method takes in the predicted conductivity values, performs a
        segmentation using Otsus method and computes the mean dice score.

        We segment the ground truth using the knowledge of the background conductivity.

        sigma_pred: prediction, FenicsX function
        sigma_gt: ground truth, FenicsX function

        """
        sigma_pred_np = np.array(sigma_pred.x.array[:]).flatten()
        sigma_gt_np = np.array(sigma_gt.x.array[:]).flatten()

        # segment prediction
        pred_seg = self.multi_level_otsu_segmentation(sigma_pred_np)

        # segment ground truth
        class0 = sigma_gt_np < (self.backCond - 0.2)  # Lower conductivity
        class1 = (sigma_gt_np >= (self.backCond - 0.2)) & (
            sigma_gt_np < (self.backCond + 0.2)
        )  # Background
        class2 = sigma_gt_np >= (self.backCond + 0.2)  # Higher conductivity

        segmented_gt = np.zeros_like(sigma_gt_np)
        segmented_gt[class0] = -1
        segmented_gt[class1] = 0
        segmented_gt[class2] = 1

        dice_score = self.compute_dice_per_class(pred_seg, segmented_gt)

        return dice_score

    def multi_level_otsu_segmentation(self, image):
        # Apply thresholding

        try:
            thresholds = threshold_multiotsu(image, classes=3)
        except ValueError:
            print("Image has less grayscale values than classes, return zero image")
            return np.zeros_like(image)

        # Segment image into three classes based on the thresholds
        class0 = image < thresholds[0]  # Lower conductivity
        class1 = (image >= thresholds[0]) & (image < thresholds[1])  # Background
        class2 = image >= thresholds[1]  # Higher conductivity

        # However, sometimes this does not work and class1 is actually the background class
        # We work with the heuristic that the class with the most number of elements is the background
        inds = [
            np.count_nonzero(class0),
            np.count_nonzero(class1),
            np.count_nonzero(class2),
        ]
        bgclass = inds.index(max(inds))  # background class

        segmented_image = np.zeros_like(image)
        if bgclass == 0:  # background is class0, join class1 + class2 for higher
            segmented_image[class0] = 0
            segmented_image[class1] = 1
            segmented_image[class2] = 1
        elif (
            bgclass == 1
        ):  # background is class1, then class0 is lower, class2 is higher
            segmented_image[class0] = -1
            segmented_image[class1] = 0
            segmented_image[class2] = 1
        elif bgclass == 2:  # background is 2, join class0 + class1 for lower
            segmented_image[class0] = -1
            segmented_image[class1] = -1
            segmented_image[class2] = 0
        else:
            print("Segmentation failed, return zero image")

        return segmented_image

    def compute_dice_per_class(self, pred, gt):
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


class DynamicRange(PerformanceMetric):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, sigma_pred, sigma_gt):
        """
        sigma_pred: prediction, FenicsX function
        sigma_gt: ground truth, FenicsX function

        """
        sigma_pred_np = np.array(sigma_pred.x.array[:]).flatten()
        sigma_gt_np = np.array(sigma_gt.x.array[:]).flatten()

        max_sigma_pred = np.max(sigma_pred_np)
        min_sigma_pred = np.min(sigma_pred_np)

        max_sigma_gt = np.max(sigma_gt_np)
        min_sigma_gt = np.min(sigma_gt_np)

        return (max_sigma_pred - min_sigma_pred) / (max_sigma_gt - min_sigma_gt)


class MeasurementError(PerformanceMetric):
    def __init__(self, name, solver):
        super().__init__(name)

        self.solver = solver

    def __call__(self, sigma_pred, Umeas):
        """
        sigma_pred: prediction, FenicsX function
        Umeas: measurements numpy array

        """
        _, Upred = self.solver.forward_solve(sigma_pred)

        Upred = np.array(Upred)

        return np.sum((Upred - Umeas) ** 2) / np.sum(Umeas**2)
