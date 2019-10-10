import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

F = 270
class FrameProcessor(object):

    def __init__(self, w, h):
        self.W = w
        self.H = h
        self.orb =  cv2.ORB_create(100)
        self.last = None
        self.K = np.array([[F, 0, self.W//2], [0, F, self.H//2], [0, 0, 1]])
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.Kinv = np.linalg.inv(self.K)

    def extractRt(self, E):
        W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        U,d,Vt = np.linalg.svd(E)
        assert np.linalg.det(U) > 0
        if np.linalg.det(Vt) < 0:
            Vt *= -1.0
        R = np.dot(np.dot(U, W), Vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vt)
        t = U[:, 2]
        pose = np.concatenate([R, t.reshape(3,1)], axis=1)
        print(pose)
        return pose

    def add_ones(self, x):
        return np.concatenate([x , np.ones((x.shape[0], 1))], axis=1)

    def normalize(self, pts):
        return np.dot(self.Kinv, self.add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        ret  = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

    def processFrame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # feature detection
        corners = cv2.goodFeaturesToTrack(image, 3000, 0.01, 3)
        corners = np.int0(corners)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in corners]

        # feature extraction
        kps, des = self.orb.compute(frame, kps)
        matches = None
        pose    = None
        # feature matching
        ret = []

        if self.last is not None:
            matches = self.flann.knnMatch(des, self.last['descriptors'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        self.last = {'kps':kps, 'descriptors':des}

        if matches is None:
            return np.array([]), None, corners, pose

        if(len(ret) > 0):
            ret = np.array(ret)

            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            model, inliers = ransac((ret[:, 0], ret[:,1]), EssentialMatrixTransform, min_samples=8, residual_threshold=0.005, max_trials=200)
            ret = ret[inliers]
            pose = self.extractRt(model.params)
        return ret, matches, corners, pose