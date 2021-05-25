
# import các thư viện cần thiết
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pyautogui
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from datetime import datetime
import ctypes
import tkinter as tk
from tkinter import filedialog


#hàm dự đoán của chương trình
#nhận vào 1 frame (ảnh) , model nhận diện mặt , model nhận diện đeo khẩu trang
def detect_and_predict_mask(frame, faceNet, maskNet): 
	# lấy độ dài và độ rộng của frame
	(h, w) = frame.shape[:2]
	#xử lý hình ảnh trước khi cho vào model để dự đoán (mean subtraction)
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	#cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
	# sử dụng mean để giúp chống lại những thay đổi về độ sáng trong hình ảnh đầu vào

	#bỏ ảnh đã xử lý vào model nhận diện mặt
	faceNet.setInput(blob)
	#lấy kết quả dự đoán là 1 numpy arr

	#confidence = detections[0, 0, i, 2] --> lấy tỉ lệ % 
	#boxPoints = inference_results[0, 0, i, 3:7] --> lấy vị trí khuôn mặt

	detections = faceNet.forward()

	# tạo danh sách mặt , vị trí , dự đoán
	faces = []
	locs = []
	preds = []

	# lặp qua tất cả các vật thể bắt được 
	for i in range(0, detections.shape[2]):
		#lấy % tỉ lệ là mặt người
		confidence = detections[0, 0, i, 2]
		#lọc ra những vật thể có tỉ lệ thấp
		if confidence > 0.5:
			#lấy khuôn mặt dò được
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#rằng buộc để đảm bảo khuôn mặt nằm trong frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			#lấy khuôn mặt , chuyển nó từ dạng BGR thành RGB , resize thành 224x224 và xử lý
			#Nếu sử dụng imshow của matplotlib nhưng đọc hình ảnh bằng OpenCV cần chuyển đổi từ BGR sang RGB.
			try:
				face = frame[startY:endY, startX:endX] #lấy khuôn mặt dò được
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				#lưu lại khuôn mặt và vị trí của nó
				faces.append(face)
				locs.append((startX, startY, endX, endY))
			except:
				# nếu ko nhận dược frame
				print("Khong nhan duoc frame")

	# chỉ dự doán nếu số khuôn mặt tìm được là lớn hơn 0 
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	#trả về vị trí và dự đoán
	return (locs, preds)

def detect_video():
	vs = VideoStream(src=0, resolution = resolution).start() #src=0 -- webcam
	
	while True:
		frame = vs.read()#lấy frame từ cam 
		#thực hiện kiểm tra , ghi lại vị trí và kết quả dự đoán
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		#sẽ tạo box(khung) và pred(dự đoán) cho mỗi bộ(zip) trong vị trí và dự đoán (locs,preds)
		for (box, pred) in zip(locs, preds):
			# lấy vị trí của mặt 
			(startX, startY, endX, endY) = box
			#model dự đoán sẽ trả về 2 biến cho biết % có khẩu trang và % không có khẩu trang
			(mask, withoutMask) = pred

			# xác định màu và dòng chữ dự đoán 	


			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# ghi kết quả dự đoán dưới dạng
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# hiện label lên frame , cao hơn 10px so với khuôn mặt

			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			#cv2.putText(image,lable,vị trí,font,fontscale,color,thickness)

			#vẽ khung hình chữ nhật xung quanh khuôn mặt
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			#cv2.rectangle(image,vị trí,màu,thickness)


		# hiện frame lên màn hình
		cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		#cv2.moveWindow("Camera", 40,30)
		cv2.imshow("Camera", frame)

		#chờ nhấn phím 
		key = cv2.waitKey(1) & 0xFF

		# nếu nhấn 'q' dừng vòng lặp
		if key == ord("q"):
			break
		elif key == ord("p"):
			now = datetime.now()
			dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
			filename = 'image-'+ dt_string +'.jpg'

			(h, w) = frame.shape[:2]
			#pyautogui.screenshot(filename, region=(40,30,w,h))
			pyautogui.screenshot(filename)

			image = cv2.imread(filename)
			cv2.imwrite(filename, image)  
	# tắt cửa sổ frame  
	cv2.destroyAllWindows()
	vs.stop()
def detect_image():
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
	faceNet.setInput(blob)
	detections = faceNet.forward()

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
			(mask, withoutMask) = maskNet.predict(face)[0]

			# Xác định nhãn và màu sắc của nhãn
			label = "Mask" if mask > withoutMask else "No Mask"
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



# load model nhận diện khuôn mặt vào chương trình
print("Tai Model nhan dien khuon mat...")

prototxtPath ="face_detector/config.prototxt"  #đường dẫn file config
weightsPath ="face_detector/res10_300x300_ssd_iter_140000.caffemodel"  #đường dẫn file model
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath) # model nhận diện mặt

print("Tai Model nhan dien khuon mat thanh cong !")
# load model nhận diện đeo khẩu trang 
print("Tai Model nhan dien deo khau trang...")

maskNet = load_model("mask_detector.model") # model nhận diện đeo khẩu trang

print("Tai Model nhan dien deo khau trang thanh cong")
# lấy resolution
user32 = ctypes.windll.user32
resolution = (user32.GetSystemMetrics(1), user32.GetSystemMetrics(0))


#khởi tạo gui
window = tk.Tk()
window.title('Facemask Detector') 
window.geometry("400x280")
window.resizable(False, False)
# Thêm giao diện
label = tk.Label(text="Facemask Detector", height=5,width=35,bg="black",fg="white")
label.pack();
button1 = tk.Button(window, text='Start Camera', width=35, command=detect_video) 
button2 = tk.Button(window, text='Choose Image', width=35, command=detect_image) 
button3 = tk.Button(window, text='Stop', width=35, command=window.destroy) 
button1.pack() 
button2.pack() 
button3.pack() 
window.mainloop()


