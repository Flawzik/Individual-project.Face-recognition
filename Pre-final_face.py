import cv2
import pickle
from csv import writer
from csv import DictWriter
import datetime
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'cascades/data/haarcascade_eye.yml')
#smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
labels = {"имя_ученика" : 1}
person={}


cur_date = datetime.datetime.now()

current_date = cur_date.strftime("%d-%m-%Y %H:%M")

with open('attendance,scv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Имя', 'Время']
    )


with open('labels.pickle', 'rb') as f:
    prev_labels = pickle.load(f)
    labels = {v:k for k, v in prev_labels.items()}

cap_cam = cv2.VideoCapture(0)
while True:
    # цикл считывающий кадры видео ряда
   ret, frame =cap_cam.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
   for(x, y, w, h) in faces:
      # print(x, y, w, h)



       roi_gray = gray[y:y + h, x:x + w]#region of interest location of frame
       roi_color=frame[y:y+h, x:x+w]
       id_, conf = recognizer.predict(roi_gray)
       if conf >= 45 and conf <= 85:
#print(person.writerow(person))
           #print (id_)
           #print(labels[id_])
           #date_time = datetime.fromtimestamp(1887639468)
           person[labels[id_]] = current_date
           print(person)
           with open('attendance,scv', 'w') as file:
               writer = csv.writer(file)
               writer.writerow(
                   [person, current_date]
               )

#print(DictWriter.writerow(person))
          # print(person)
           font = cv2.FONT_HERSHEY_SIMPLEX
           name =labels[id_]
           color = (255, 255, 255)
           stroke = 2
           cv2.putText(frame, name,(x, y), font, 1, color, stroke, cv2.LINE_AA)
           #person = person.append(name)

       img_item='1.png'
       cv2.imwrite(img_item,roi_gray) #сохраняем изображение
       cv2.rectangle(frame, (x, y), (x + w, y + h),  (0, 200, 0),2) # прямоугольник вокруг лица ||| цвет BGR (можно поменять цвет)


    #   subitems = smile_cascade_db.detectMultiScale(roi_gray)
      # for (ex,ey,ew,eh) in subitems:
       #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
       # '''''
       cv2.imshow('Face', frame)#[:, ::-1])  # отзеркалить выводимое видео
       cv2.waitKey(1)
cv2.imshow('Face',gray)


#print(DictWriter.writerow(person))

    #if person[labels[id_]]:
     #   print(person)


#if cv2.waitKey(1) &0xff == ord('c'): #окончить цикл нажав на кнопку
 #    break
#else:
  #   continue
#print(person)
"""""
person={}
while True:
    # цикл считывающий кадры видео ряда
   ret, frame =cap_cam.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5,minNeighbors=5)
   person[labels[_ids]] = None
   print(person)
"""


##for(x, y, w, h) in faces:
#print(x, y, w, h)
#roi_gray = gray[y:y + h, x:x + w]#region of interest location of frame  roi_color=frame[y:y+h, x:x+w]
#id_, conf = recognizer.predict(roi_gray)
  #     if conf >= 45 and conf <= 85:
           #print (id_)
    #       print(labels[id_])
      #     font = cv2.FONT_HERSHEY_SIMPLEX
      #    name =labels[id_]
       #    color = (255, 255, 255)
       #    stroke = 2
         #  cv2.putText(frame, name,(x, y), font, 1, color, stroke, cv2.LINE_AA) `1
#'''''''''


#writer.writerow()

print(DictWriter.writerow(person))



cap_cam.release()
cv2.destroyAllWindows()
