# %%

import cv2
import numpy as np
import os

def read_image(img_path):
    """
    Return the original image and the binary image in grayscale
    img_path: image's path
    """
    ori_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    
    # Convert to binary image
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i][j] < 128:
                gray_img[i][j] = 0
            else:
                gray_img[i][j] = 255

    return ori_img, gray_img


def candidates(img):
    """
    Return the filtered image and the candidates
    img: binary image
    This function will groups pixels that are connected to each other and filter them based on their area and width/height ratio
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    gray = np.zeros_like(img)
    # Iterate through the connected components and filter them based on their area
    count = 0
    candidates = []

    for i in range(1, num_labels):  # 0 is background element
        x, y, width, height, area = stats[i]
    
        ratio = float(width) / height
        if ratio > 0.1 and ratio < 10 and area > (0.003 * img.shape[0] * img.shape[1]):
            count += 1
            candidates.append([x,y,width,height,area])
            gray[labels == i] = 255
    return gray, candidates


def create_Imgfiles():
    """
    Create a new directory to store the cropped characters in each car plate images
    """

    if not os.path.exists('Cropped_Characters'):
        os.makedirs('Cropped_Characters')
    num_of_imgs =10

    for i in range(1,num_of_imgs+1):
        if not os.path.exists('Cropped_Characters/' + 'Carplate_{}'.format(i)):
            os.makedirs('Cropped_Characters/' + 'Carplate_{}'.format(i))

        ori_img, gray_img = read_image('{}.jpg'.format(i))
        gray_img, my_candidates = candidates(gray_img)

        # # Draw possible candidates (For visualization of possible candidate purposes)
        # pos_img = ori_img
        # for stat in my_candidates:
        #     x, y, w, h, area = stat
        #     cv2.rectangle(pos_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the candidates in each images and save it to as a new image
        for j in range(len(my_candidates)):
            x, y, w, h, area = my_candidates[j]
            cv2.imwrite("Cropped_Characters/"+ "Carplate_" + str(i) +"/{}.jpg".format(j), gray_img[y:y+h, x:x+w])

    # cv2.imshow("Candidate", pos_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

create_Imgfiles()
