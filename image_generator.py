import numpy as np
import cv2


def generate_DIP():
    img_dip = np.zeros((128, 128, 3), np.uint8)
    cv2.putText(img_dip, 'D', (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 180, 80), 4)
    cv2.putText(img_dip, 'I', (55, 95),
                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (240, 10, 60), 4)
    cv2.putText(img_dip, 'P', (85, 75),
                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (60, 60, 225), 4)
    return img_dip
