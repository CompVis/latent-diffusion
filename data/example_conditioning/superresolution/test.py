import cv2
  
# read the images
img1 = cv2.imread('face64_1.png')
img2 = cv2.imread('face64_2.png')
img3 = cv2.imread('face64_3.png')
img4 = cv2.imread('face64_4.png')

im_h = cv2.hconcat([img1, img2, img3, img4])
# cv2.imshow('man_image.jpeg', im_h)

cv2.imwrite('hconcat_resize.png', im_h)
