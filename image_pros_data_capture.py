import cv2
import pytesseract
import os
import imutils


folder_path = '/Users/macbot/Workspace/Development/OpenCV-ImageProcessing/Pdf_tasks/Document/ICAI_DATA/Sole/north/' \
              'extracted_images'

test_image_path = 'test_image.jpeg'


def load_image(file):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = imutils.resize(img, height=1000)
    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img.shape)


if __name__ == '__main__':
    load_image(test_image_path)


