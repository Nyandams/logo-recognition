import cv2
from matplotlib import pyplot as plt


def imread_resize(filename, flags=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(filename, flags)
    if img is None:
        print('Could not open or find the image {}'.format(filename))
        exit(0)

    width = img.shape[1]
    height = img.shape[0]
    min_shape = min(width, height)

    if min_shape < 300:
        scale_percent = 300 / min_shape
        dim = (int(width * scale_percent), int(height * scale_percent))
        return cv2.resize(img, dim, cv2.INTER_AREA)
    else:
        return img


def show_keypoints_img(img, kp):
    img_kp = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_kp), plt.show()


def show_match_imgs(img1, img2, akaze):
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    #  Filter matches using the Lowe's ratio test
    ratio_thresh = 0.65
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.title('good matching post AKAZE')
    plt.imshow(img_matches), plt.show()


def get_features(img, akaze):
    kp, des = akaze.detectAndCompute(img, None)
    return kp, des


akaze = cv2.AKAZE_create()
"""
img1 = imread_resize('img/ce/ce-logo.png')  # queryImage
img2 = imread_resize('img/ce/ce-logo50.png')  # trainImage
show_match_imgs(img1, img2, akaze)
"""

list_img_path = ['img/ce/ce-logo50.png',
                 'img/ce/ce-logo.png',
                 'img/ce/ce-flou.jpg',
                 'img/ce/ce-logo2.png',
                 'img/ce/ce-logo3.png',
                 'img/ce/ce-logo4.jpg',
                 'img/bnp/bnp2.png',
                 'img/bnp/bnp.jpg',
                 'img/bnp/bnp5.jpg',
                 'img/bnp/bnp3.jpg',
                 'img/bnp/bnp4.jpg']

list_img = [imread_resize(filename) for filename in list_img_path]
list_kp_des = [get_features(img, akaze) for img in list_img]
list_kp = [kp_des[0] for kp_des in list_kp_des]
list_des = [kp_des[1] for kp_des in list_kp_des]

[show_match_imgs(img1, img2, akaze) for img1 in list_img for img2 in list_img]