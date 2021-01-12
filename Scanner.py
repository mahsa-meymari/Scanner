import cv2
import numpy as np


def det(a, b):
    return a[0] * b[1] - b[0] * a[1]


def line_intersection_cramer(line1, line2, imgShape):
    b1 = line1[0][0] - line1[1][0]
    b2 = line2[0][0] - line2[1][0]
    a1 = -1 * (line1[0][1] - line1[1][1])
    a2 = -1 * (line2[0][1] - line2[1][1])
    c1 = a1 * line1[0][0] + b1 * line1[0][1]
    c2 = a2 * line2[0][0] + b2 * line2[0][1]
    D = det((a1, a2), (b1, b2))
    if D == 0:
        return None
    Dx = det((c1, c2), (b1, b2))
    Dy = det((a1, a2), (c1, c2))
    x = int(Dx / D)
    y = int(Dy / D)
    if x < 0 or y < 0:
        return None
    if x > imgShape[0] + imgShape[0] / 5 or y > imgShape[1] + + imgShape[1] / 5:
        return None
    return x, y

def pol2car(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return x1, y1, x2, y2


def get_four_lines(canny):
    img = canny.copy()
    lines = cv2.HoughLines(img, 0.95, np.pi / 90, 70, None, 0, 0)
    lineCar = []
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    print(len(lines))
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            x1, y1, x2, y2 = pol2car(rho, theta)
            lineCar.append(((x1, y1), (x2, y2)))
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    return lineCar,img

def find_corners(linelis):
    corners = []
    for i in range(0, len(linelis)):
        for j in range(i + 1, len(linelis)):
            intersection = line_intersection_cramer(linelis[j], linelis[i], img.shape)
            if intersection != None:
                corners.append(intersection)
    corners = sorted(corners, key=lambda x: x[0])
    corners = sorted(corners, key=lambda x: x[1])
    print(corners)
    return corners

def transform(image, corners):
    img = image.copy()
    #find prespective transform
    src = np.array(corners, np.float32)
    cols, rows, channels = image.shape
    dst = [(0, 0),(rows, 0), (0, cols),(rows, cols)]
    transformMatrix = cv2.getPerspectiveTransform(src, np.array(dst, np.float32))
    #     transform the image
    img = cv2.warpPerspective(img, transformMatrix, (rows, cols))
    return img


img_path = './test_images/sample.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 43)
canny = cv2.Canny(blur, 10, 100)
linelis,lines = get_four_lines(canny)
corners = find_corners(linelis)
for corner in corners:
    cv2.circle(lines, corner, 3,(0,0,255),-1)
rotated = transform(img, corners)


output = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
output = cv2.adaptiveThreshold(output,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,4)
# _,output = cv2.threshold(output,210,255,cv2.THRESH_BINARY)
cv2.imwrite('./steps/blur.png',blur)
cv2.imwrite('./steps/canny.png',canny)
cv2.imwrite('./steps/lines.png',lines)
cv2.imwrite('./steps/rotated.png',rotated)
cv2.imwrite('./output.png',output)