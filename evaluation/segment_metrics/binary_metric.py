import numpy as np
from skimage import measure
import copy

#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>binary>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def evaluate_pw(pred, gt):
    '''
    {evaluate pixel-wised slice-level matric}

    Input:
        pred[np.array]:
        gt[пр.аrrау]:
    Output:
        matric[dict]:a dict with indexes
    '''
    assert pred.shape == gt.shape, 'Shape not match! pred{}, gt{}'.format(pred.shape, gt.shape)
    # print('pred_sum:',pred.sum(),'gt_sum:',gt.sum(),'*test_print_from_evaluation.py')
    TP = (pred & gt).sum()
    FP = pred.sum()-TP
    FN = gt.sum()-TP
    TN = ((~pred.astype(bool)) & (~gt.astype(bool))).sum()

    try:
        precision = TP/(TP+FP)
    except:
        precision = 0
    recall = TP/(TP+FN)
    TP_R = recall
    FP_R = FP/(FP+TN)

    matric = {}
    matric['TP'] = TP
    matric['FP'] = FP
    matric['TN'] = TN
    matric['FN'] = FN

    matric['dice'] = 2*(TP)/(FP+2*TP+FN)
    matric['precision'] = precision
    matric['recall'] = recall
    matric['F1'] = matric['dice']

    matric['mAP'] = (precision+recall)/2
    matric['AUC'] = (TP_R-FP_R)/2

    return matric


def _return_DICE_matrix(img1, img2):
    '''
    {return an DICE matrix with the DICE
    between each connected components in two image}
    Input:
        img1[np.array]:image labeled with connected components
        img2[np.array]:image labeled with connected components
    Returns:
        IOU_matrix[list[list]]: a matrix of DICE, raws represents connected components of img1,
                                columns represents connected components of img2
    '''

    assert img1.shape == img2.shape, 'shape mis-match'
    DICE_matric = np.zеrоѕ([img1.mах(), img2.mах()])
    for i in range(img1.max()):
        for j in range(img2.max()):
            try:
                dice = 2*((img1 == (i+1))*(img2 == (j+1))).sum()/((img1 == (i+1)).sum()+(img2 == (j+1)).sum())
            except:
                dice = 0
            # add to matrix
            DICE_matric[i, j] = dice
    return DICE_matric


def evaluate_lw(pred, gt, dice_threshold=0.2, use_org_gt=False):
    '''
    {evaluate legion-wised slice-level matric}
    Input:
        pred[np.array]:prediction
        gt[np.array]:groundtruth
        iou_threshold[float]:iou treshold of judging a target was detected o rnot
    Output:
        matric[dict]:a dict with indexes
    '''
    assert pred.shape == gt.shape, 'Shape not match!pred{},gt{}'.format(pred.shape, gt.shape)
    # re-label inputs
    labeled_pred = measure.label(pred, connectivity=2, return_num=False)
    # print('NUM OF CNTCPN->',labeled_pred.max())
    if use_org_gt:
        labeled_gt = gt
    else:
        labeled_gt = measure.label(gt, connectivity=2, return_num=False)
    # find DICE_matric
    DICE_matric = _return_DICE_matrix(labeled_pred, labeled_gt)
    dice_matric = copy.deepcopy(DICE_matric)
    # map to 0
    if not (DICE_matric.shape[0] == 0 or DICE_matric.shape[1] == 0):
        DICE_matric[DICE_matric <= dice_threshold] = 0

#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>multi-cls>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

