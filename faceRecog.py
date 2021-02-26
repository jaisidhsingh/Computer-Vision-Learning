import face_recognition
import numpy as np
import os
import cv2

tolerance = 0.5

known_tags = []
known_faces = []
frame_thickness = 2
font_thickness = 2
model = "cnn"

for filename in os.listdir(r"assets\known_faces"):
	image = face_recognition.load_image_file(r'assets/known_faces/'+ filename)
	encoded_img = face_recognition.face_encodings(image)[0]
	known_faces.append(encoded_img)
	known_tags.append("obama")

for name in os.listdir(r"assets\unknown_faces"):
	image = face_recognition.load_image_file(r'assets/unknown_faces/'+ name)
	loc = face_recognition.face_locations(image, model=model) 
	encoded_img2 = face_recognition.face_encodings(image, loc)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	for face_encoding, face_location in zip(encoded_img2, loc):
		res = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
		if True in res:
			match = res.index(True)
			print(f"match found : obama : {match+1}")

			tl = (face_location[3], face_location[0])
			br = (face_location[1], face_location[2])
			color = [0, 255, 0]
			cv2.rectangle(image, tl, br, color, frame_thickness)


			tl = (face_location[3], face_location[2])
			br = (face_location[1], face_location[2]+20)
			cv2.rectangle(image, tl, br, color, cv2.FILLED)
			cv2.putText(image, f"obama {match+1}", (face_location[3]+10, face_location[2]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), font_thickness)
	cv2.imshow(name, image)
	cv2.waitKey(0)
