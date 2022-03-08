import cv2
def process_img(image):
    image= cv2.resize(image, (84, 84))
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV)
    return image[1]
