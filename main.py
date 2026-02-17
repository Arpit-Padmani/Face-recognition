import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file("ImagesBasic/elon_musk.jpg")
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file("ImagesBasic/test.jpg")
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

imgarpit = face_recognition.load_image_file("ImagesBasic/arpit.jpg")
imgarpit = cv2.cvtColor(imgarpit,cv2.COLOR_BGR2RGB)

imgarpit2 = face_recognition.load_image_file("ImagesBasic/arpit-2.jpg")
imgarpit2 = cv2.cvtColor(imgarpit2,cv2.COLOR_BGR2RGB)

faceLocElon = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]

faceLocTest = face_recognition.face_locations(imgtest)[0]
encodeTest = face_recognition.face_encodings(imgtest)[0]

faceLocArpit = face_recognition.face_locations(imgarpit)[0]
encodeArpit = face_recognition.face_encodings(imgarpit)[0]
cv2.rectangle(imgarpit, (faceLocArpit[3],faceLocArpit[0]),(faceLocArpit[1],faceLocArpit[2]),(255,0,255),2)

faceLocArpit2 = face_recognition.face_locations(imgarpit2)[0]
encodeArpit2 = face_recognition.face_encodings(imgarpit2)[0]


resultElon = face_recognition.compare_faces([encodeElon],encodeArpit2)
resultArpit = face_recognition.compare_faces([encodeArpit],encodeTest)

faceDis = face_recognition.face_distance([encodeElon],encodeArpit2)
faceDisArpit = face_recognition.face_distance([encodeArpit],encodeTest)

print(resultElon , faceDis)
print(resultArpit , faceDisArpit)


cv2.imshow("Arpit ", imgarpit)
# cv2.imshow("Elon Musk", imgElon)
# cv2.imshow("Elon Musk Test", imgtest)
cv2.waitKey(0)
