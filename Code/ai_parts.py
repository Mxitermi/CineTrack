import cv2
import numpy as np

def find_mask_center(pred_mask, frame=None, threshold=0.3):
    """
    Findet die Mitte der größten Maske in pred_mask (numpy array, Werte 0..1).
    Optional: Zeichnet Rechteck und Mittelpunkt in das frame (OpenCV Bild).

    Args:
        pred_mask (np.ndarray): predicted Maske als float numpy array (H,W), Werte 0..1
        frame (np.ndarray): optional, BGR Bild, um Zeichnungen vorzunehmen
        threshold (float): Schwellenwert für Binarisierung
        draw (bool): ob Rechteck und Punkt gezeichnet werden sollen

    Returns:
        center (tuple or None): Mittelpunkt (x,y) oder None, wenn keine Maske gefunden
        bbox (tuple or None): bounding box (x,y,w,h) oder None
    """
    pred_mask_bin = (pred_mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(pred_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center = (x + w // 2, y + h // 2)
    return center, (x, y, w, h)
