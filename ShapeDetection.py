import cv2

img = cv2.imread("shapes.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)

    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        ratio = w/float(h)
        shape = "Square" if 0.95 <= ratio <= 1.05 else "Rectangle"
    elif len(approx)>4:
        shape = "Circle"
    else:
        shape = "Unknown"

    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
    cv2.putText(img, shape, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    



