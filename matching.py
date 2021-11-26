import numpy as np
import cv2
from kivy.lang import Builder
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog

KV = '''
<MatchingConfiguration>
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "120dp"
    
    MDBoxLayout:
        MDLabel:
            text: "Match Cross check (BF only)"
        MDSwitch:
    
    MDBoxLayout:
        MDLabel:
            text: "Detector Type"
        MDDropdownMenu:
            id: detector_type 
            text: "SIFT"
            items: ["SIFT", "ORB", "BRISK"]
    
'''

Builder.load_string(KV)


class MatchingConfiguration(MDBoxLayout):
    pass


def create_configuration_dialog():
    conf = MatchingConfiguration()


    dialog = MDDialog(
        title="Address:",
        type="custom",
        content_cls=conf,
        buttons=[
            MDFlatButton(
                text="CANCEL",
            ),
            MDFlatButton(
                text="OK",
            ),
        ],
    )
    return dialog

def match_images(
    kp1,
    des1,
    kp2,
    des2,
    cross_check: bool = True,
    detector_type: str = "SIFT",
    lowe: float = 0.7,
    ransack_threshold: float = 7.0,
    matcher_type: str = "brute_force"
):

    if matcher_type == "brute_force":
        K = 1 if cross_check else 2

        if detector_type == "SIFT":
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
        elif detector_type in ["ORB", "BRISK"]:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        elif "ORB" in detector_type:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        else:
            raise NotImplementedError(f"detector_type={detector_type}")
    else:

        K = 2
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
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

    if len(good) < 10:
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
