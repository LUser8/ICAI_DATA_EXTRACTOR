import cv2
import numpy as np
import imutils
import pytesseract
import os
import sys


FONT = cv2.FONT_HERSHEY_SIMPLEX


def segment_doc(img):
    vert_seg = {}
    # loop to travel each image exist in extracted image folder
    img_clone = img.copy()  # load image to the open cv library
    img_clone = img.copy()
    gray = cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)  # convert colored image to gray scale image

    # size of the image
    shape = img_clone.shape[:2]
    # print(shape)
    # conversion of the image into binary image
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # for image size
    rows, cols = thresh.shape[:2]
    scale = 30

    # extract horizontal lines
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    erode = cv2.erode(thresh.copy(), horizontalStructure)
    dilate = cv2.dilate(erode, horizontalStructure)

    # extract vertical lines
    verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    erode1 = cv2.erode(thresh.copy(), verticalKernel)
    dilate1 = cv2.dilate(erode1, verticalKernel)

    # find vertical line location blocks
    _, contours, hierarchy = cv2.findContours(dilate1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[::-1]
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        points = (x, 0, x+w, y+h)
        point = [x-2, shape[0]-1]
        # cv2.rectangle(img_clone, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 1)
        # cv2.putText(img_clone, str(i), (points[0]-5, points[1]), FONT, 3, (0, 0, 255), 3, cv2.LINE_4)
        vert_seg[str(i)] = point
        # print('points' + str(i), point)

    # print('Contours_len:', len(contours))

    # creating mask for extracting all horizontal and vertical lines from the original image
    mask = dilate + dilate1

    # apply mask using and operation on the original image
    dst = cv2.bitwise_and(img_clone, img_clone, mask=255 - mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    dst2 = cv2.bitwise_or(mask, img_clone)

    segment = []
    start_points = [0, 0]
    final_point = list(shape[::-1])
    end_point = list()
    if len(vert_seg.keys()) == 2:
        for key in vert_seg.keys():
            end_point = vert_seg[key]
            segment.append(start_points + end_point)
            start_points = [vert_seg[key][0], 0]
        segment.append(start_points+final_point)
        # print(segment)

    # For testing
    # dst2 = imutils.resize(dst2, height=800)
    # cv2.imshow('extract', dst2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return segment, dst2


def make_blocks(img_c):
    img_clone = img_c.copy()
    gray = cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (470, 12))
    grad1 = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    _bw, thresh = cv2.threshold(grad1.copy(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    connected = thresh.copy()
    _, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = contours[::-1]

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 16600:
            x, y, w, h = cv2.boundingRect(cnt)
            roi1 = img_clone[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi1)
            tex = text.split(' ')
            print(text)
            print('================================')
            # cv2.rectangle(img_clone, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # cv2.imshow('roi', img_clone)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def extract_data(image, segments):
    img_clone = image.copy()
    for i, d in enumerate(segments):
        roi = img_clone[d[1]:d[3], d[0]:d[2]]

        # text = pytesseract.image_to_string(roi)
        print('ROI_'+str(i))
        print('================================')
        make_blocks(roi)
        print('================================')
        # cv2.imshow('roi', roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def load_document(doc_path):
    file_list = [file for file in os.listdir(doc_path)]
    file_list = sorted(file_list, key=lambda x: int(x.split('-')[1].split('.')[0]))
    for file in file_list:
        img_loc = doc_path + '/' + file
        img = cv2.imread(img_loc)
        segments, img_c = segment_doc(img)
        extract_data(img_c, segments)


if __name__ == '__main__':
    # img_loc = '/Users/macbot/Workspace/Development/OpenCV-ImageProcessing/ICAI_DATA_CAPTURE/test_image.jpeg'
    doc_loc = '/Users/macbot/Workspace/Development/OpenCV-ImageProcessing/Pdf_tasks/Document/ICAI_DATA/Sole/center/extracted_images'
    # img = cv2.imread(img_loc)
    # segments, img_c = segment_doc(img)
    # extract_data(img_c, segments)
    load_document(doc_loc)

