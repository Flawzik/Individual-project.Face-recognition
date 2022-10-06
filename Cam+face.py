import cv2

cap_cam = cv2.VideoCapture(0)# ввод веб камеры
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")# заранее прописанный алгоритм cv2 c github

while True: # цикл считывающий кадры видео ряда
    success, img = cap_cam.read() # пока success считывать изображение |проверяет работоспособность

    faces = face_cascade_db.detectMultiScale(img, 1.1, 19) #сюда добовляем список координат распозннаных лица
    for (x, y, w, h) in faces: # координаты лиц
        cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 200), 2) # прямоугольник вокруг лица ||| цвет BGR (можно поменять цвет)
    #'''''


    cv2.imshow('Face', img[:, ::-1]) #отзеркалить выводимое видео
    cv2.waitKey(1)
    #if cv2.waitKey(1) &0xff == ord('f'): || можно окончить  цикл нажав на кнопку
     #  break

cap_cam.release()
cv2.destroyAllWindows()
