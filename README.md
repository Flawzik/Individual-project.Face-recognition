Project.Face-recognition



***

# 🛡️ Smart Access: Система контроля доступа по лицу

Проект реализует систему умных пропусков, использующую компьютерное зрение для идентификации личности. При совпадении лица с базой данных система отправляет сигнал на микроконтроллер Arduino, который имитирует открытие замка (включает индикацию).

## ⚡ Как это работает

1.  **Камера** захватывает видеопоток в реальном времени.
2.  **Алгоритм** обнаруживает лицо и сравнивает его с эталонами в базе.
3.  **Логика:**
    *   ✅ **Доступ разрешен:** Если найдено совпадение > Python отправляет команду через UART -> Arduino включает зеленый светодиод.
    *   ❌ **Доступ запрещен:** Если лицо не распознано -> Система игнорирует запрос или сигнализирует об ошибке.

## 🛠 Стек технологий

*   **Backend & CV:** Python, OpenCV, `face_recognition` (или `deepface`)
*   **Hardware:** Arduino Uno/Nano, LED-модуль
*   **Связь:** PySerial (UART)



## 📄 License

MIT License.

***



## 📬 Contact

bogdahka2006@gmail.com-email


---



<img width="349" height="274" alt="image" src="https://github.com/user-attachments/assets/9accd8ff-6c23-4c1c-b571-7e70f514d0a5" />


<img width="349" height="279" alt="image" src="https://github.com/user-attachments/assets/d0d7119a-4eb2-4077-b6d6-cdabcc76d763" />



