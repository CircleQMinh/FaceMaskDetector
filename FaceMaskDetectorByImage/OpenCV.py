# import các thư viện cần thiết
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from tkinter import *
from tkinter import filedialog

def detect_image():
	# load model nhận diện khuôn mặt vào chương trình
	print("Tai Model nhan dien khuon mat...")
	prototxtPath ="face_detector/config.prototxt"  #đường dẫn file config
	weightsPath ="face_detector/res10_300x300_ssd_iter_140000.caffemodel"  #đường dẫn file model
	net = cv2.dnn.readNet(prototxtPath, weightsPath) # model nhận diện mặt
	print("Tai Model nhan dien khuon mat thanh cong !")
	# load model nhận diện đeo khẩu tran
	print("Tai Model nhan dien deo khau trang...")
	model = load_model("mask_detector.model")
	print("Tai Model nhan dien deo khau trang thanh cong")
	# tải hình ảnh đầu vào, sao chép nó và lấy không gian hình ảnh
	# thứ nguyên
	print("Tai hinh anh")
	
	filename = filedialog.askopenfilename(title = "Chon mot hinh anh",filetype=(("jpg file","*.jpg"),("all files","*.*")))

	File_path = filename

	image = cv2.imread(File_path)
	orig = image.copy()
	(h, w) = image.shape[:2]

	# tạo một đốm màu từ hình ảnh
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Đưa đốm màu qua mạng và nhận diện các khuôn mặt
	print("Dang nhan dien khuon mat")
	net.setInput(blob)
	detections = net.forward()

	# lặp lại các phát hiện
	for i in range(0, detections.shape[2]):
		# Trích xuất dộ tin cậy (xác suất) liên kết với phát hiện.
		confidence = detections[0, 0, i, 2]

		# Lọc các tin cậy và đảm bảo độ tin cậy
		# lớn hơn 50%
		if confidence > 0.5:
			# tính toán khung (x,y) chứa đối tượng
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Đảm bảo khung giới hạn (x,y) nằm trong khung hình
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# trích xuất ROI của khuôn mặt, chuyển đổi nó từ BGR sang kênh RGB
			# thay đổi kích thước khuôn mặt thành 224x224 và xử lí trước
			face = image[startY:endY, startX:endX] #lấy khuôn mặt dò được
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# Đưa khuôn mặt qua model để xác định có đeo khẩu trang hay không
			(mask, withoutMask) = model.predict(face)[0]

			# Xác định nhãn và màu sắc của nhãn
			label = "Co deo khau trang" if mask > withoutMask else "Khong deo khau trang"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# Thêm xác suất tìm được ở model vào nhãn
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# Hiển thị nhãn và khung chứa khuôn mặt
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# Hiển thị hình ảnh đã sửa lí
	cv2.imshow("Anh da xu li", image)
	cv2.waitKey(0)

detect_image()