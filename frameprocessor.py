import cv2
import numpy as np

class FrameProcessor(object):
    def __init__(self, w, h):
        self.W = w
        self.H = h
        self.orb =  cv2.ORB_create(100)
        self.last = None
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def processFrame(self, frame):

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # feature detection
        corners = cv2.goodFeaturesToTrack(image, 3000, 0.01, 3)
        corners = np.int0(corners)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in corners]

        # feature extraction
        kps, des = self.orb.compute(frame, kps)
        matches = None

        # feature matching
        query = []
        train = []

        if self.last is not None:
            matches = self.flann.knnMatch(des, self.last['descriptors'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    query.append(kps[m.queryIdx])
                    train.append(self.last['kps'][m.trainIdx])
        self.last = {'kps':kps, 'descriptors':des}

        if matches is None:
            return [],[],None,corners

        return query, train , matches, corners