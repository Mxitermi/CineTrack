import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_center(image_path):
    # Bild laden (nur Schwarz-Weiß)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binärbild erzeugen: Weiß = 255, Schwarz = 0
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Konturen finden
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Rechteck um das größte weiße Objekt zeichnen
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 5)  # Roter Rahmen
        cv2.circle(preview, (x + w // 2, y + h // 2), 5, (0, 255, 0), 8)  # Grüner Mittelpunkt
        plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        plt.title('Rahmen um weißes Objekt')
        plt.axis('off')
        plt.show()
        return [x, y, w, h], [x + w // 2, y + h // 2]  # Rechteck und Mittelpunkt
    

rect, center = find_center(f'../Samples/Masks/{input("Enter the image name: ")}')
# Vorschau anzeigen

