"""Evaluation Metrics for Semantic Segmentation"""
import paddle
import numpy as np

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']



class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):

            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, paddle.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = paddle.zeros([self.nclass,1])
        self.total_union = paddle.zeros([self.nclass,1])
        self.total_correct = 0
        self.total_label = 0


# paddle version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = paddle.argmax(output.astype('long'), 1) + 1
    target = target.astype('long') + 1
    pixel_labeled = paddle.sum(target > 0).item()
    # try:
    pixel_correct = paddle.sum((predict == target) * (target > 0)).item()
    # except:
    #     print("predict size: {}, target size: {}, ".format(predict.shape, target.shape))
    # assert pixel_correct < pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = paddle.argmax(output, 1) + 1  # [N,H,W]
    target = target.astype('float32') + 1            # [N,H,W]

    predict = predict.astype('float32') * (target > 0).astype('float32')
    intersection = predict * (predict == target).astype('float32')
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = paddle.histogram(intersection, bins=nbins, min=mini, max=maxi)
    area_pred = paddle.histogram(predict, bins=nbins, min=mini, max=maxi)
    area_lab = paddle.histogram(target, bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert paddle.sum(area_inter > area_union) == 0, "Intersection area should be smaller than Union area"
    return area_inter.astype('float32'), area_union.astype('float32')


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc



if __name__ == '__main__':
    import numpy as np
    import paddle
    from reprod_log import ReprodLogger

    # # Metric Align
    # reprod_logger = ReprodLogger()
    # model = ICNet()
    # model_file = 'paddle_from_torch.pdparams'
    # model.load_dict((paddle.load(model_file)))
    # model.eval()
    # fake_data = np.load('../models/fake_data.npy', allow_pickle=True)
    # fake_label = np.load('../models/fake_label.npy', allow_pickle=True)
    # print(fake_label)
    # input = paddle.to_tensor(fake_data)
    # label = paddle.to_tensor(fake_label)
    # outputs = model(input)
    # metric = SegmentationMetric(19)
    # metric.update(outputs[0], label)
    # pixAcc, mIoU = metric.get()
    # print(pixAcc)
    # print(mIoU)
    #
    # print(np.array([mIoU.item()]))
    # reprod_logger.add("mIoU", np.array([mIoU.item()]))
    # reprod_logger.save("metric_paddle.npy")
