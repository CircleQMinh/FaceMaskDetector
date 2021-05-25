

# import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4  #tốc độ học trung bình 
EPOCHS = 5 #số lần train
BS = 32 # 1 lần train có 32 ảnh

#tạo danh sách chứa đường dẫn ảnh
print("Bat dau tai anh....")
imagePaths = list(paths.list_images("./dataset1")) #1-dataset nhỏ #2-dataset lớn #3-dateset vừa
#khởi tạo array chứa ảnh và loại của ảnh
data = []
labels = []

#lập qua các đường dẫn ảnh để bỏ ảnh và label vào arr
troni=1
for imagePath in imagePaths:
	# lấy label từ đường dẫn
	label = imagePath.split(os.path.sep)[-2]

	# load ảnh từ đường dẫn (224x224)
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# thêm ảnh và label vào đúng array của nó
	data.append(image)
	labels.append(label)
	print(troni)
	troni+=1

# chuyển thành NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# thử hiện one-hot encoding để chuyển label thành binary array
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# chia data thành bộ 80% để train và 20% để test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
#random_state = 42 để đảm bảo mỗi lần tách có kết quả giống nhau
#x_train: Phần train của data (x)
#x_test: Phần test của data(x)
#y_train: Phần train của label (y)
#y_test: Phần test của label (y)

# trình tạo hình ảnh để train 
aug = ImageDataGenerator(
	rotation_range=20,				#Phạm vi độ để quay(rotate) hình ảnh
	zoom_range=0.15,				#Phạm vi thu-phóng ảnh 
	width_shift_range=0.2,			#độ lệch trái-phải 
	height_shift_range=0.2,			#độ lệch trên-dưới
	shear_range=0.15,				#độ cắt ảnh
	horizontal_flip=True,			#cho phép lật ảnh theo chiều ngang
	fill_mode="nearest")			#kiểu fill ảnh

# Tạo base model từ model được đào tạo trước MobileNet V2 
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Tạo head model dựa trên base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# Tạo model dựa trên base model và sẽ trở thành model mà chương trình sẽ train
model = Model(inputs=baseModel.input, outputs=headModel)

# loop qua layers trong base model và ngắt(freeze) nó để tránh phá hủy bất kỳ thông tin nào 
# mà chúng có trong các đợt train sau này.
for layer in baseModel.layers:
	layer.trainable = False

# Compile model để train
print("Dang compile model de train...")
# Tạo optimizer dể cải thiện tốc độ và hiệu suất
opt = Adam(lr=INIT_LR,              #learning rate : thay đổi model bao nhiêu khi model update 
		  decay=INIT_LR / EPOCHS)	#decay : leaning rate giảm dần


model.compile(loss="binary_crossentropy", #hàm được sử dụng để tìm lỗi hoặc sai lệch trong quá trình học.
			  optimizer=opt,				
			metrics=["accuracy"])        #được sử dụng để đánh giá hiệu suất của model

# bắt đầu train
print("Bat dau train...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS), #lấy data,label của phần train để train
	steps_per_epoch=len(trainX) // BS,		 #số bước mỗi lần train
	validation_data=(testX, testY),			#lấy data,label của phần test để test		
	validation_steps=len(testX) // BS,		 #số bước mỗi lần test
	epochs=EPOCHS)							#số lần train

# thực hiện dự đoán và kết thúc model
print("Chuan bi ket thuc train...")
# predict trả về có dạng [[%x,%y],[%x,%y],[],....]
predIdxs = model.predict(testX, batch_size=BS)

#chuyển thành [0,1,0,1,...]
predIdxs = np.argmax(predIdxs, axis=1)

# in ra report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save model
print("Dang save model...")
model.save('mask_detector.model', save_format="h5")

