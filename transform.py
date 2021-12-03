from operator import sub

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


# pad image to cover ROI, return the shift amount of origin
def addBorder(img, rect):
    top, bottom, left, right = int(0), int(0), int(0), int(0)
    x, y, w, h = rect
    tl = (x, y)
    br = (x + w, y + h)
    if tl[1] < 0:
        top = -tl[1]
    if br[1] > img.shape[0]:
        bottom = br[1] - img.shape[0]
    if tl[0] < 0:
        left = -tl[0]
    if br[0] > img.shape[1]:
        right = br[0] - img.shape[1]
    img = cv.copyMakeBorder(
        img, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0]
    )
    orig = (left, top)
    return img, orig


def check_limits(pts, size):
    np.clip(pts[:, 0], 0, size[1] - 1, pts[:, 0])
    np.clip(pts[:, 1], 0, size[0] - 1, pts[:, 1])
    return pts


################################################################################
# Stitching functions                                                          #
################################################################################


def warpImage(img, H):
    # tweak the homography matrix to move the result to the first quadrant
    H_cover, pos = coverH(size2rect(img.shape), H)
    # find the bounding box of the output
    x, y, w, h = warpRect(size2rect(img.shape), H_cover)
    width, height = x + w, y + h
    assert width * height < 1e8  # do not exceed 300 MB for 8 GB RAM
    # warp the image using the corrected homography matrix
    # all the fuss is because of the indexing conventions of numpy and cv2
    # warped = cv.warpPerspective(img, H_corr, (width, height))
    idx_pts = np.mgrid[0:width, 0:height].reshape(2, -1).T
    map_pts = transform(idx_pts, np.linalg.inv(H_cover))
    map_pts = map_pts.reshape(width, height, 2).astype(np.float32)
    warped = cv.remap(img, map_pts, None, cv.INTER_CUBIC).transpose(1, 0, 2)
    # make the external boundary solid black, useful for masking
    warped = np.ascontiguousarray(warped, dtype=np.uint8)
    gray = cv.cvtColor(warped, cv.COLOR_RGB2GRAY)
    _, bw = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    # https://stackoverflow.com/a/55806272/12447766
    major = cv.__version__.split(".")[0]
    if major == "3":
        _, cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    else:
        cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    warped = cv.drawContours(warped, cnts, 0, [0, 0, 0], lineType=cv.LINE_4)
    return (warped, pos)


# only the non-zero pixels are weighted to the average
def mean_blend(img1, img2):
    assert img1.shape == img2.shape
    locs1 = np.where(cv.cvtColor(img1, cv.COLOR_RGB2GRAY) != 0)
    blended1 = np.copy(img2)
    blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
    locs2 = np.where(cv.cvtColor(img2, cv.COLOR_RGB2GRAY) != 0)
    blended2 = np.copy(img1)
    blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
    # blended = cv.addWeighted(blended1, 0.5, blended2, 0.5, 0)
    blended = cv.addWeighted(blended1, 1.0, blended2, 0.0, 0)
    return blended


# only the non-zero pixels are weighted to the average
def mean_blend_v2(img1, img2):

    assert img1.shape == img2.shape

    locs1 = np.where(cv.cvtColor(img1, cv.COLOR_RGB2GRAY) != 0)
    blended1 = np.copy(img2)
    blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
    # locs2 = np.where(cv.cvtColor(img2, cv.COLOR_RGB2GRAY) != 0)
    # blended2 = np.copy(img1)
    # blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
    #
    # blended = cv.addWeighted(blended1, 1.0, blended2, 0.0, 0)
    return blended1


def blend_images(imageA, imageB, H):
    return warpPano(imageA, imageB, H, (0, 0))


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


def warpPano(prevPano, img, H, orig):
    # corret homography matrix
    T = np.array([[1, 0, -orig[0]], [0, 1, -orig[1]], [0, 0, 1]])
    H_corr = H.dot(T)
    # warp the image and obtain shift amount of origin
    result, pos = warpImage(prevPano, H_corr)
    xpos, ypos = pos
    # zero pad the result
    rect = (xpos, ypos, img.shape[1], img.shape[0])
    result, _ = addBorder(result, rect)
    # mean value blending
    idx = np.s_[ypos : ypos + img.shape[0], xpos : xpos + img.shape[1]]
    result[idx] = mean_blend(result[idx], img)
    # crop extra paddings
    x, y, w, h = cv.boundingRect(cv.cvtColor(result, cv.COLOR_RGB2GRAY))
    result = result[y : y + h, x : x + w]
    # return the resulting image with shift amount
    return (result, (xpos - x, ypos - y))


# no warping here, useful for combining two different stitched images
# the image at given origin coordinates must be the same
def patchPano(img1, img2, orig1=(0, 0), orig2=(0, 0)):
    # bottom right points
    br1 = (img1.shape[1] - 1, img1.shape[0] - 1)
    br2 = (img2.shape[1] - 1, img2.shape[0] - 1)
    # distance from orig to br
    diag2 = tuple(map(sub, br2, orig2))
    # possible pano corner coordinates based on img1
    extremum = np.array(
        [(0, 0), br1, tuple(map(sum, zip(orig1, diag2))), tuple(map(sub, orig1, orig2))]
    )
    bb = cv.boundingRect(extremum)
    # patch img1 to img2
    pano, shift = addBorder(img1, bb)
    orig = tuple(map(sum, zip(orig1, shift)))
    idx = np.s_[
        orig[1] : orig[1] + img2.shape[0] - orig2[1],
        orig[0] : orig[0] + img2.shape[1] - orig2[0],
    ]
    subImg = img2[orig2[1] : img2.shape[0], orig2[0] : img2.shape[1]]
    pano[idx] = mean_blend(pano[idx], subImg)
    return (pano, orig)


# base image is the last image in each iteration
def blend_multiple_images(images, homographies):
    N = len(images)
    assert N >= 2
    assert len(homographies) == N - 1
    pano = np.copy(images[0])
    pos = (0, 0)
    for i in range(N - 1):
        # get homography matrix
        img = images[i + 1]
        H = homographies[i]
        # warp pano onto image
        pano, pos = warpPano(pano, img, H, pos)
    return (pano, pos)


################################################################################
# Miscellaneous                                                                #
################################################################################


def color_list(N, colormap=cv.COLORMAP_HSV):
    cmap = cv.applyColorMap(np.array(range(256), np.uint8), colormap)
    return list(tuple(int(c) for c in cmap[int(256 * n / N)][0]) for n in range(N))


def mark_points(img, x, colormap=cv.COLORMAP_HSV, bgr=False):
    marked = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    if colormap == "invert":
        colors = list(tuple(int(256 - v) for v in img[p]) for p in x)
    elif type(colormap) is list:
        if len(colormap) >= len(x):
            colors = colormap
        else:
            colors = color_list(len(x))
    else:
        colors = color_list(len(x), colormap)
    for i in range(len(x)):
        marked = cv.drawMarker(
            marked,
            (int(x[i][0]), int(x[i][1])),
            colors[i],
            cv.MARKER_CROSS,
            markerSize=30,
            thickness=2,
        )
    if bgr:
        return marked
    else:
        return cv.cvtColor(marked, cv.COLOR_BGR2RGB)


def match_points(img1, img2, pts1, pts2, colors=None, hstack=True):
    if colors is None:
        colors = color_list(len(pts1))
    if img1.shape != img2.shape:
        img1b = cv.copyMakeBorder(
            img1,
            0,  # up
            max(img2.shape[0] - img1.shape[0], 0),  # down
            0,  # left
            max(img2.shape[1] - img1.shape[1], 0),  # right
            cv.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        img2b = cv.copyMakeBorder(
            img2,
            0,  # up
            max(img1.shape[0] - img2.shape[0], 0),  # down
            0,  # left
            max(img1.shape[1] - img2.shape[1], 0),  # right
            cv.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        img1 = img1b
        img2 = img2b
    img = np.concatenate((img1, img2), axis=int(hstack))
    for i in range(len(pts1)):
        pt1 = tuple(pts1[i])
        if hstack:
            pt2 = (pts2[i][0] + img1.shape[1], pts2[i][1])
        else:
            pt2 = (pts2[i][0], pts2[i][1] + img1.shape[0])
        img = cv.line(img, pt1, pt2, colors[i], 3, cv.LINE_AA)
    return img


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
    stitched_img: np.ndarray, current_img_frame: np.ndarray, H: np.ndarray
):
    height, width = stitched_img.shape[:2]
    lines = [
        ([[0, 0], [width, 0]], (255, 0, 0)),
        ([[width, 0], [width, height]], (0, 255, 0)),
        ([[width, height], [0, height]], (255, 0, 0)),
        ([[0, height], [0, 0]], (0, 255, 0)),
    ]
    line_thickness = 2

    for line, color in lines:
        line = np.array(line, dtype=np.float32)
        line = homography_transform(line, H).astype(np.int32)
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(current_img_frame, (x1, y1), (x2, y2), color, thickness=line_thickness)