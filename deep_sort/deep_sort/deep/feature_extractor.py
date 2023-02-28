import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
    known_embedding = []
    with open('./known_embedding.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            # 将字符串以空格和'\r\n'分割，然后转换为int类型的数组赋值给elem
            elem = list(map(int, line.split()))
            known_embedding.append(elem)
    f.close()


    def list_txt(path, list=None):
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


    name_list = list_txt(path='name_list.txt')
    cost_matrix = distance(known_embedding, feature)
    print('cost_matrix')

