import cv2
import numpy as np

def create_5d_array(frame, x, y):
    # Form: (Höhe, Breite, Kanäle, 1, 1)
    h, w, c = frame.shape
    array = np.zeros((h, w, c, 1, 1), dtype=np.uint8)
    array[:, :, :, 0, 0] = frame  # Bilddaten einfügen
    print(f"5D Array erzeugt mit Klick bei ({x}, {y})")
    return array

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param.copy()
        array_5d = create_5d_array(frame, x, y)
        # Optional: z.B. speichern oder weiterverarbeiten
        print("Array-Shape:", array_5d.shape)

# Kamera starten
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Aktuelle Auflösung: {int(width)}x{int(height)}")
cv2.namedWindow("Kamera")
cv2.setMouseCallback("Kamera", mouse_callback, param=None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Callback-Parameter aktualisieren
    cv2.setMouseCallback("Kamera", mouse_callback, param=frame)

    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
