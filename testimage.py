import cv2

image_path = "C:/Users/tlili/OneDrive/Bureau/facial emotion recognition/data/fstest.png"
frame = cv2.imread(image_path)

if frame is None:
    print("❌ OpenCV cannot read the image. Try using another format (JPG/PNG).")
else:
    print("✅ OpenCV successfully loaded the image.")
    cv2.imshow("Test Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
