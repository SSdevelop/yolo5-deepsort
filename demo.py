from AIDetector_pytorch import Detector
import imutils
import cv2
import pandas as pd
import numpy as np


# 保存模式
# list_txt(path='savelist.txt', list=List1)
# 读取模式
# List_rd = list_txt(path='savelist.txt')
def list_txt(path, list=None):
    '''
    :param path: 储存list的位置
    :param list: list数据
    :return: None/re将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''

    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

def _nn_euclidean_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return np.maximum(0.0, r2.min(axis=0))

def distance(features, targets):
    cost_matrix = np.zeros((len(targets), len(features)))
    for i, target in enumerate(targets):
        cost_matrix[i, :] = _nn_euclidean_distance(target, features)
    return cost_matrix

def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    import pytest
    # print("cosine2")
    # pytest.set_trace()
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

def _pdist(a, b, single_embedding):
    new = np.asarray(a)
    known = np.asarray(b)
    if len(new) == 0 or len(known) == 0:
        return np.zeros((len(new), len(known)))
    new2, known2 = np.square(new).sum(axis=1), np.square(known).sum(axis=1)
    r2 = -2. * np.dot(new, known.T) + new2[:, None] + known2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    # import pytest
    # pytest.set_trace()
    return r2

def _nn_euclidean_distance(x, y, single_embedding):
    distances = _pdist(x, y, single_embedding)
    return np.maximum(0.0, distances.min(axis=0))

def main():

    name = 'demo'
    # change suitcase / phone / others
    item_to_detect = ['person', 'suitcase']
    det = Detector(item_to_detect)
# 
    name_list = []
    known_embedding = []
    name_list, known_embedding = det.loadIDFeats()
    print(name_list,known_embedding)
    # known_embedding = np.array(known_embedding)
    list_txt(path='name_list.txt', list=name_list)

    fw = open('known_embedding.txt', 'w')
    for line in known_embedding:
        for a in line:
            fw.write(str(a))
            fw.write('\t')
        fw.write('\n')
    fw.close()

    # cap = cv2.VideoCapture('IMG_6761.mp4')
    cap = cv2.VideoCapture('IMG_1752.mp4')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(100/fps)

    videoWriter = None
    targetLocked=False
    minIndex=None
    trackId=None
    while True:
        # RES_DIR = set_res_dir()
        # if TRAIN:
        #     !python train.py --data ../data.yaml --weights yolo5s.pt \
        #     --img 640 --epochs {EPOCHS} --batch-size 16 --name {RES_DIR}

        # try:
        success, im = cap.read()
        import pytest
        if im is None:
            break
        DetFeatures = []
        DetFeatures,img_input,box_input= det.loadDetFeats(im)
        # detFeatures = np.array(DetFeatures)
        if len(DetFeatures) > 0 and not targetLocked:
            # pytest.set_trace()
            dist_matrix = _nn_euclidean_distance(known_embedding,DetFeatures, known_embedding[0])
            import pytest
            pytest.set_trace()
            minIndex=dist_matrix.argmin()
            if trackId is None:
                trackId = minIndex + 1
            det.targetTrackId=trackId
            targetLocked = True
            # result = det.feedCap(im,draw=False)
            # pytest.set_trace()
        
            # _nn_cosine_distance(DetFeatures, known_embedding)
        # pytest.set_trace()
        # distance(known_embedding,detFeatures)
        # targetFeatures = np.array(known_embedding[3])
        # pytest.set_trace()
        result = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        import pytest
        if det.isLost is True:
            # pytest.set_trace()
            # cv2.imwrite(f'./test-{det.frameCounter/fps}-second.png', result)
            print('lost')

            # todo: quit after write.
        cv2.imshow(name, result)
        cv2.waitKey(t)

        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出
        #     break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()