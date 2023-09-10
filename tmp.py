import cv2
import ECGImageGenerator
import ECG as ECG
import PaperSimulator

row_height = 6
columns = 2
ecg = ECG.ECG()
ECGImageGenerator.plot(ecg.get_signals(),row_height=6, columns=[4,2,2,1,1],sample_rate = 500)
ECGImageGenerator.save_as_png("attempt")

image = cv2.imread('attempt.png')
image_new =  PaperSimulator.add_padding(image)
cv2.imshow('Padded Image', image_new)
cv2.waitKey(0)
cv2.destroyAllWindows()