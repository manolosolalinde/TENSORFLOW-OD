import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, image_np = cap.read()
    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
