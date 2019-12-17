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


img1 = imread_resize('img/bp/bp3.jpg')  # queryImage
img2 = imread_resize('img/bp/bp1.jpg')  # trainImage

# AKAZE
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

img1_kp = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(img1_kp), plt.show()

img2_kp = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(img2_kp), plt.show()

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

plt.title('good matching post SURF')
plt.imshow(img_matches), plt.show()