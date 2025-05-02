import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("images/0000001.png",cv2.IMREAD_GRAYSCALE)

ret1,thresh1 = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)
thresh2 = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,6)
image_blur = cv2.GaussianBlur(image,(5,5),0)
ret, thresh3 = cv2.threshold(image, 40, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)


images = [thresh1,thresh2]
titles = ['Global Thresholding', 'Gaussian Adaptive Thresholding']

'''for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.savefig('proposal_example.png')'''

image2 = cv2.imread("Swastika_1.webp", cv2.IMREAD_GRAYSCALE)
ret4, image2BW = cv2.threshold(image2,200,255,cv2.THRESH_BINARY_INV)

srcTri = np.array([[0, 0], [image2BW.shape[1] - 1, 0], [0, image2BW.shape[0] - 1]]).astype(np.float32)
dstTri = np.array([[0, image2BW.shape[1] * 0.33], [image2BW.shape[1] * 0.85, image2BW.shape[0] * 0.25],
                   [image2BW.shape[1] * 0.15, image2BW.shape[0] * 0.7]]).astype(np.float32)

warp_mat = cv2.getAffineTransform(srcTri, dstTri)

warp_dst = cv2.warpAffine(image2BW, warp_mat, (image2BW.shape[1], image2BW.shape[0]))


cv2.imshow("WebP Image", warp_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("example_warp.png", warp_dst)