from skimage import measure
import numpy as np
import time


def _get_mcls_dice_matrix(lb1, lb2, **kwarges):
    '''
    {return multi-class dice matrix}
    PARAMS:
        lbl[np.array]:label1withmulti-class
        lb2[np.array]:label2withmulti-class
    RETURNS:
        dice_matrix[np.array]:
        shape->[n_1+1,n_2+1]
        n_1->connected components of labell
        n__2->connected components of label2
        dice_matrix[1:,0]->classindices of labell
        dice_matrix[0,l:]->class indices ofl abel2
        dice_matrix[1:,l:]->dice scores
    '''

    ts = time.time()

    connectivity = kwarges.get('connectivity', 1)
    lbl_labeled = measure.label(lb1, connectivity=connectivity)
    lb2_labeled = measure.label(lb2, connectivity=connectivity)
    # print('>>[_get_mcls_dice_matrix_cncnpt] Time %3.3f'%(time.time()-ts),)
    dice_matrix = np.zeros([lbl_labeled.max()+1, lb2_labeled.max()+1])
    for lb1_i in range(0, lbl_labeled.max()+1):
        for lb2_i in range(0, lb2_labeled.max()+1):
            if lb1_i == 0 and lb2_i == 0:
                pass
            elif lb1_i != 0 and lb2_i == 0:
                dice_matrix[lb1_i, lb2_i] = lb1[lbl_labeled == lb1_i][0]
            elif lb1_i == 0 and lb2_i != 0:
                dice_matrix[lb1_i, lb2_i] = lb2[lb2_labeled == lb2_i][0]
            else:
                dice = 2*((lbl_labeled == lb1_i)*(lb2_labeled == lb2_i)).sum()/ \
                       ((lbl_labeled == lb1_i).sum()+(lb2_labeled == lb2_i).sum())
                dice_matrix[lb1_i, lb2_i] = dice
    # print('>> [_get_mcls_dice_matrix] Time %3.3f'%(time.time()-ts),)
    return dice_matrix


def _map_rule(dc_mtrc, rule, dice_threshold):
    '''
    {map dice_matrix to bool matrix base on different rule}
    PARAMS:
        dc_mtrc[np.array]:
        rule[str]: 'lbl1_prime','lbl2_prime'
        dice_threshold[float]
    RETURNS:
        hit_indices[list[tuple]]: list of indices in tuple
    NOTE:
        if hit right label -> set hit to right label
        if hit wrong label -> set hit to label with max dice
        if hit nothing -> set hit to 0
    '''
    hit_indices = []
    if dc_mtrc.shape[0] == 1:
        for c in dc_mtrc[0, 1:]:
            hit_indices.append((0, c))
    elif dc_mtrc.shape[1] == 1:
        for c in dc_mtrc[1:, 0]:
            hit_indices.append((c, 0))
    else:
        for i in range(1, dc_mtrc.shape[0]):
            cls = dc_mtrc[i, 0]
            if (dc_mtrc[i, 1:] > dice_threshold).sum() == 0:
                hit_indices.append((cls, 0))
            elif rule == 'lbl1_prime':
                max_d = 0
                for dc, c in zip(dc_mtrc[i, 1:], dc_mtrc[0, 1:]):
                    if dc > dice_threshold and c == cls:
                        idx_hit = (cls, cls)
                        break
                    else:
                        if dc > max_d:
                            max_d = dc
                            idx_hit = (cls, c)
                hit_indices.append(idx_hit)
        for i in range(1, dc_mtrc.shape[1]):
            cls = dc_mtrc[0, i]
            if (dc_mtrc[1:, i] > dice_threshold).sum() == 0:
                hit_indices.append((0, cls))
            elif rule == 'lbl2_prime':
                max_d = 0
                for dc, c in zip(dc_mtrc[1:, i], dc_mtrc[1:, 0]):
                    if dc > dice_threshold and c == cls:
                        idx_hit = (cls, cls)
                        break
                    else:
                        if dc > max_d:
                            max_d = dc
                            idx_hit = (c, cls)
                hit_indices.append(idx_hit)
    return hit_indices


def eval_lesion_mcls_efficient(lb1, lb2, dice_threshold, **kwarges):
    '''
    {evaluation for multi-class lesion wise index}
    PARAMS:
        lb1[np.array]: label 1 with multi-class
        lb2[np.array]: label 2 with multi-class
        dice_threshold[float]: threshold of dice score of being a valid match
    RETURNS:
        confusion_matrix1[np.array]: shape -> [ncls,ncls] 'lbl1_prime'
        confusion_matrix2[np.array]: shape -> [ncls,ncls] 'lbl2_prime'
    '''
    _ndims = len(lb1.shape)
    connectivity = kwarges.get('connectivity', 1)
    ncls = kwarges.get('ncls', max(lb1.max(), lb2.max())+1)  # contains background
    total_cncnpts = measure.label((lb1+lb2) > 0, connectivity=connectivity)
    total_props = measure.regionprops(total_cncnpts)
    confusion_matrix1 = np.zeros([ncls, ncls])
    confusion_matrix2 = np.zeros([ncls, ncls])
    for prop in total_props:
        i_cncnpt = prop.label
        bbox = prop.bbox
        lb1_tmp = (lb1*(total_cncnpts == i_cncnpt))[
                  bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5],
                  ]
        lb2_tmp = (lb2*(total_cncnpts == i_cncnpt))[
                  bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5],
                  ]
        dc_mtrc = _get_mcls_dice_matrix(lb1_tmp, lb2_tmp, **kwarges)
        hit_mtrc = _map_rule(dc_mtrc, rule='lbl1_prime', dice_threshold=dice_threshold)
        for idx1, idx2 in hit_mtrc:
            confusion_matrix1[int(idx1), int(idx2)] += 1
        hit_mtrc = _map_rule(dc_mtrc, rule='lbl2_prime', dice_threshold=dice_threshold)
        for idx1, idx2 in hit_mtrc:
            confusion_matrix2[int(idx1), int(idx2)] += 1
    return confusion_matrix1, confusion_matrix2


if __name__ == '__main__':
    a = np.array([[
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 3, 0, 0, 4],
        [1, 0, 0, 0, 0, 0, 3, 4],
        [1, 0, 0, 0, 3, 0, 0, 4],
        [1, 0, 0, 0, 3, 0, 0, 0],
    ]])

    b = np.array([[
        [1, 0, 0, 4, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 5, 0, 3, 3, 0, 4],
        [1, 0, 0, 0, 3, 3, 3, 0],
        [1, 0, 0, 0, 3, 3, 0, 4],
        [0, 0, 0, 0, 3, 0, 0, 0],
    ]])
    confusion_matrix1, confusion_matrix2 = eval_lesion_mcls_efficient(a, b, dice_threshold=.1)
