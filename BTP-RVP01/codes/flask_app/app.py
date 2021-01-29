import os
from flask import Flask,flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
# from detector import Detector
from yolo_v4 import Detector
from integrated import Mask
import cv2
from PIL import Image
from yolo import YOLO,detect_video
from social_distance_detection import SocialDistance
from keras_yolov4.test_video import DetectorVideo
# from yolo_v4 import 
	

UPLOAD_FOLDER = '\\static\\Files\\'
ALLOWED_EXTENSIONS = {"mp4","mov","jpg","jpeg","png"}
# ALLOWED_EXTENSIONS_IMAGES = {"jpg","jpeg","png"}

app = Flask(__name__, static_folder='static')
# app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS 


@app.route("/")
def hello():
	return render_template('index.html')
@app.route("/team")
def team():
	return render_template('team.html')
@app.route("/output")
def output():
	return render_template('output.html')

@app.route("/upload",methods = ["GET","POST"])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		keys=['file1','file2','file3','file4','file5','file6','file7','file8']
		flag="file1"
		print(request.files)
		list1=list(request.files)
		# print(list1)
		# if(request.files['file1'] in  keys):
		# 	print("hi")
		if(list1[0] in keys):
			flag=list1[0]
		if flag not in request.files:
			# flag="file2"
			return render_template('upload.html', msg='No file selected')
		# if 'file2' not in request.files:
		# 	flag="file1"
		# 	return render_template('upload.html', msg='No file selected')
		file = request.files[flag]

		if file.filename == '':
			# flash('No Selected file')
			return render_template('upload.html', msg = 'No file Selected')

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			ext=filename.rsplit('.',1)[1].lower()
			# print(ext)
			file.save(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename))
			dec=None
			output_path=""
			if(flag=="file1"):
				if(ext=="mp4"):
					decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.avi'
					output_path="./static/{}".format(x)
					decvideo.video_detect(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename),output_path)
					return render_template('upload.html', msg=["video",output_path])

				else:

					x='detections'+file.filename+'.jpg'
					output_path="./static/{}".format(x)
					dec = Detector('trash_weights.h5','trash_classes.txt')
					image=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
					img,class_ids,dict1= dec.detect_object(image)
					cv2.imwrite(output_path,img)
					print(class_ids)


			elif(flag=="file2"):
				if(ext=='mp4'):
					decvideo = DetectorVideo('gun_weights.h5','gun_classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.avi'
					output_path="./static/{}".format(x)
					decvideo.video_detect(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename),output_path)

					# decvideo.video_detect(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename))
					return render_template('upload.html', msg=["video",output_path])

				else:
					x='detections'+file.filename+'.jpg'
					output_path="./static/{}".format(x)
					dec=Detector('algae_weights.h5','algae_classes.txt')
					image=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
					img,class_ids,dict1= dec.detect_object(image)
					cv2.imwrite(output_path,img)
					print(class_ids)


					
			elif(flag=="file3"):
				if(ext=='mp4'):
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.avi'
					output_path="./static/{}".format(x)
					decvideo = DetectorVideo('knife_weights.h5','knife_classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					decvideo.video_detect(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename),output_path)
					return render_template('upload.html', msg=["video",output_path])

				else:
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.jpg'
					output_path="./static/{}".format(x)
					dec = Detector('multiclass_btp_weights.h5','multiclasses_btp.txt')
					image=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
					img,class_ids,dict1= dec.detect_object(image)
					cv2.imwrite(output_path,img)
					print(class_ids)
			elif(flag=="file4"):
				if(ext=='mp4'):
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.avi'
					output_path="./static/{}".format(x)
					decvideo = DetectorVideo('handbag_weights.h5','handbag_classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					decvideo.video_detect(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename),output_path)
					return render_template('upload.html', msg=["video",output_path])

				else:
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.jpg'
					output_path="./static/{}".format(x)
					dec=Detector('handbag_weights.h5','handbag_classes.txt')
					image=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
					img,class_ids,dict1= dec.detect_object(image)
					cv2.imwrite(output_path,img)
					print(class_ids)
			elif(flag=="file5"):
				if(ext=='mp4'):
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.avi'
					output_path="./static/{}".format(x)
					decvideo = DetectorVideo('Backpack_weights.h5','backpack_classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					decvideo.video_detect(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename),output_path)
					return render_template('upload.html', msg=["video",output_path])


				else:
					x='detections'+file.filename+'.jpg'
					output_path="./static/{}".format(x)
					dec=Detector('Backpack_weights.h5','backpack_classes.txt')
					image=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
					img,class_ids,dict1= dec.detect_object(image)
					cv2.imwrite(output_path,img)
					print(class_ids)
			elif(flag=="file6"):
				# print("mask")
				dec=Mask()
				# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
				x='detections'+file.filename+'.jpg'
				output_path="./static/{}".format(x)
				image=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
				img,class_ids,dict1= dec.detect_object(image)
				cv2.imwrite(output_path,img)
				print(class_ids)
			elif(flag=="file7"):
				if(ext=="jpg"):
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.jpg'
					output_path="./static/{}".format(x)
					dec1=YOLO()  #license plate
					image=Image.open(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename))
					img1,class_ids1= dec1.detect_image(image)
					img1.save(output_path)
					return render_template('upload.html', msg=["image",output_path])
				if(ext=="mp4"):
					dec1=YOLO()
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.avi'
					output_path="./static/{}".format(x)
					video=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
					status= detect_video(dec1,video,output_path)
					# print(status)
					return render_template('upload.html', msg=["video",output_path])

					# detect_video(dec1,video,'/static')
				# print(class_ids1)
			elif(flag=="file8"):
				if(ext=="mp4"):
					dec=SocialDistance()
					# decvideo = DetectorVideo('yolo4_weight.h5','classes.txt','./keras_yolov4/model_data/yolo4_anchors.txt')
					x='detections'+file.filename+'.avi'
					output_path="./static/{}".format(x)
					dec.detect_video(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename),output_path)
					# print(type(writer)) 
					return render_template('upload.html', msg=["video",output_path])


			# 	img,class_ids= dec.detect_image(image)

			# img=None
			# class_ids=None
			
				
				# image=Image.open(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename))
				# img,class_ids= dec.detect_image(image)

			# else:
			# 	# print("VIDEO FILE NAME:",file.filename)
			# 	image=os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename)
			# 	img,class_ids,dict1= dec.detect_object(image)
			# 	cv2.imwrite(os.path.join(os.getcwd()+'/static/','detection.jpg'),img)
			# 	print(class_ids)
			# print(UPLOAD_FOLDER)

			#cv2.imshow('image',img)
			
				
			# file.save(os.path.join(os.getcwd()+UPLOAD_FOLDER,'detection.jpg'))
			return render_template('upload.html', msg=["image",output_path])
		else:
			# flash('Check the File Extension')
			return render_template('upload.html',msg= "Check File Extension")
	elif request.method == 'GET':
		return render_template('upload.html') 



if __name__ == '__main__':
	app.run(debug = True)