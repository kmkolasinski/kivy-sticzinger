import cv2 as cv
import numpy as np


def transform(src_pts, H):
    # src = [src_pts 1]
    src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
    # pts = H * src
    pts = np.dot(H, src.T).T
    # normalize and throw z=1
    pts = (pts / pts[:, -1].reshape(-1, 1))[:, 0:2]
    return pts


# find the ROI of a transformation result
def warpRect(rect, H):
    x, y, w, h = rect
    corners = [[x, y], [x, y + h - 1], [x + w - 1, y], [x + w - 1, y + h - 1]]
    extremum = transform(corners, H)
    minx, miny = np.min(extremum[:, 0]), np.min(extremum[:, 1])
    maxx, maxy = np.max(extremum[:, 0]), np.max(extremum[:, 1])
    xo = int(np.floor(minx))
    yo = int(np.floor(miny))
    wo = int(np.ceil(maxx - minx))
    ho = int(np.ceil(maxy - miny))
    outrect = (xo, yo, wo, ho)
    return outrect


def size2rect(size):
    return (0, 0, size[1], size[0])


# homography matrix is translated to fit in the screen
def coverH(rect, H):
    # obtain bounding box of the result
    x, y, _, _ = warpRect(rect, H)
    # shift amount to the first quadrant
    xpos, ypos = int(0), int(0)
    if x < 0:
        xpos = int(-x)
    if y < 0:
        ypos = int(-y)
    # correct the homography matrix so that no point is thrown out
    T = np.array([[1, 0, xpos], [0, 1, ypos], [0, 0, 1]])
    H_corr = T.dot(H)
    return (H_corr, (xpos, ypos))


# only the non-zero pixels are weighted to the average
def mean_blend_v2(img1, img2):

    assert img1.shape == img2.shape

    locs1 = np.where(cv.cvtColor(img1, cv.COLOR_RGB2GRAY) != 0)
    blended1 = np.copy(img2)
    blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]

    return blended1


def cv_blend_images(imageA, imageB, H):
    # move origin to cover the third quadrant
    H_corr, pos = coverH(size2rect(imageA.shape), H)
    xpos, ypos = pos
    # warp the image and paste the original one
    # result = cv.warpPerspective(imageA, H_corr, dst=imageB)
    result = cv.warpPerspective(imageA, H_corr, (5000, 5000))

    bottom, right = int(0), int(0)
    if ypos + imageB.shape[0] > result.shape[0]:
        bottom = ypos + imageB.shape[0] - result.shape[0]
    if xpos + imageB.shape[1] > result.shape[1]:
        right = xpos + imageB.shape[1] - result.shape[1]

    result = cv.copyMakeBorder(
        result, 0, bottom, 0, right, cv.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # mean value blending
    idx = np.s_[ypos : ypos + imageB.shape[0], xpos : xpos + imageB.shape[1]]

    result[idx] = mean_blend_v2(result[idx], imageB)

    # crop extra paddings
    x, y, w, h = cv.boundingRect(cv.cvtColor(result, cv.COLOR_RGB2GRAY))
    result = result[0 : y + h, 0 : x + w]
    # return the resulting image with shift amount
    return (result, (xpos, ypos))


def homography_transform(array: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """
    Transform list of points (x, y) with homography matrix.
    Args:
        array: a numpy array of shape [num_points, 2]
        homography: a homography matrix of shape [3, 3]

    Returns:
        transformed array
    """

    if array.shape[0] == 0:
        return array

    points = np.expand_dims(array, axis=0)
    points = cv.perspectiveTransform(points, m=homography)[0]
    points = points.reshape([-1, 2])

    return points


def draw_transformed_image_borders(
    stitched_img: np.ndarray,
    current_img_frame: np.ndarray,
    H: np.ndarray,
    line_thickness=2,
):
    height, width = stitched_img.shape[:2]
    lines = [
        ([[0, 0], [width, 0]], (255, 0, 0)),
        ([[width, 0], [width, height]], (0, 255, 0)),
        ([[width, height], [0, height]], (255, 0, 0)),
        ([[0, height], [0, 0]], (0, 255, 0)),
    ]

    for line, color in lines:
        line = np.array(line, dtype=np.float32)
        line = homography_transform(line, H).astype(np.int32)
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(current_img_frame, (x1, y1), (x2, y2), color, thickness=line_thickness)


def draw_left_to_right_overlap_line(
    stitched_img: np.ndarray,
    current_img_frame: np.ndarray,
    H: np.ndarray,
    line_thickness=2,
):
    height, width = stitched_img.shape[:2]

    line = np.array([[width, 0], [width, height]], dtype=np.float32)
    line = homography_transform(line, H).astype(np.int32)
    x1, y1 = line[0]
    x2, y2 = line[1]

    cv.line(
        current_img_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=line_thickness
    )
