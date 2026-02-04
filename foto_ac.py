import cv2

img = cv2.imread("bina.jpg")

if img is None:
    print("Fotoğraf bulunamadı!")
else:
    img_resized = cv2.resize(img, (224, 224))
    cv2.imshow("224x224 Bina", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
