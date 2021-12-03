import numpy as np
import cv2


def match_images(
    kp1,
    des1,
    kp2,
    des2,
    bf_matcher_cross_check: bool = True,
    bf_matcher_norm: str = "NORM_L2",
    lowe: float = 0.7,
    ransack_threshold: float = 7.0,
    matcher_type: str = "brute_force",
    flann_trees: int = 5,
    flann_index: int = 1,
    flann_checks: int = 50,
    min_matches: int = 10,
):

    if matcher_type == "brute_force":
        K = 1 if bf_matcher_cross_check else 2
        # TODO refactor me!
        if bf_matcher_norm == "NORM_L2":
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=bf_matcher_cross_check)
        elif bf_matcher_norm == "NORM_L1":
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=bf_matcher_cross_check)
        elif bf_matcher_norm == "NORM_HAMMING":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=bf_matcher_cross_check)
        else:
            raise NotImplementedError(f"bf_matcher_norm={bf_matcher_norm}")
    else:

        K = 2
        index_params = dict(algorithm=flann_index, trees=flann_trees)
        search_params = dict(checks=flann_checks)
        bf = cv2.FlannBasedMatcher(index_params, search_params)

        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

    matches = bf.knnMatch(des1, des2, k=K)

    if K == 1:
        good = [m[0] for m in matches if len(m) == 1]
    else:
        good = []
        for m, n in matches:
            if m.distance < lowe * n.distance:
                good.append(m)

    if len(good) < min_matches:
        return None, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransack_threshold)
    matchesMask = mask.ravel().tolist()

    matches = [p for p, m in zip(good, matchesMask) if m == 1]

    return H, matches


def select_matching_points(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return src_pts, dst_pts


def mask_bbox(img):
    a = np.where(img > 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    ymin, ymax, xmin, xmax = bbox
    return ymin, ymax, xmin, xmax
