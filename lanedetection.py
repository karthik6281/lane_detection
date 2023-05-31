import cv2
import numpy as np

path = "resources/images/06. whiteCarLaneSwitch.jpg"
img = cv2.imread(path)
imgCopy = img.copy()


def grey(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


def blur(img):
    return cv2.GaussianBlur(img,(7,7),1)


def canny(img):
    edges = cv2.Canny(img,50,50)
    return edges


def roi(img):
    mask = np.zeros_like(img)
    print(mask.shape)
    rows,cols = img.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img,mask)
    return masked_img


def hough_transform(image):
    rho = 1
    theta = np.pi/180
    threshold = 50
    minLineLength = 40
    maxLineGap = 5
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)


def average(lines):
    left = []
    right = []
    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        if x1 == x2:
            continue
        slope = (y2-y1)/(x2-x1)
        intercept = y1-(slope*x1)
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    return left_avg,right_avg


def points(img,avg):
    slope,intercept = avg
    y1 = img.shape[0]
    y2 = (y1*0.6)
    x1 = int((y1-intercept)//slope)
    x2 = int((y2-intercept)//slope)
    return np.array([x1,int(y1),x2,int(y2)])


def draw(img,lines):
    lines_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line
            cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),12)
    return lines_image


imgGray = grey(img)
imgBlur = blur(imgGray)
imgCanny = canny(imgBlur)

img_lanes = roi(imgCanny)
lines = hough_transform(img_lanes)
left,right = average(lines)
left_line = points(img,left)
right_line = points(img,right)
lane_lines = np.array([left_line,right_line])
lanes_image = draw(img,lane_lines)
lanes = cv2.addWeighted(imgCopy, 0.8, lanes_image, 1, 1)
cv2.imshow("lane detection",lanes)
cv2.waitKey(0)
