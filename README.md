CineTrack  
Tracking the Cinema

CineTrack ist ein KI-gestütztes System zur Objektverfolgung im Videostream – entwickelt für Anwendungen im Bereich Filmproduktion, Live-Bildverarbeitung und automatisiertem Kamera-Tracking.

---

## Projektübersicht

Das Projekt besteht aus zwei zentralen Python-Dateien:

### 1. `predicting_ai_with_focus.py`
Diese Datei enthält die ausführbare KI-Anwendung, die mithilfe eines UNet-Modells ein Objekt im Kamerabild erkennt und automatisch den Fokus (Crop-Fenster) auf den vorhergesagten Mittelpunkt der Maske richtet.

Funktionen:
- Live-Kameraeingang (Webcam)
- Interaktiver Startpunkt via Mausklick
- UNet-Modellvorhersage der Objektmaske
- Automatischer Zuschnitt (Crop) auf festen Bereich (`CROP_WIDTH` x `CROP_HEIGHT`), zentriert auf dem vorhergesagten Mittelpunkt
- Darstellung des Ausschnitts in Echtzeit

---

### 2. `main_training.py`
Dieses Skript dient dem Training des UNet-Modells auf einem annotierten Datensatz mit Heatmaps.

Nach dem Training wird das Modell gespeichert und kann mit `predicting_ai_with_focus.py` geladen werden.

---

## Modell

Das verwendete UNet-Modell hat eine Größe von circa 125 MB und ist damit zu groß für einen Upload auf GitHub oder die BW-KI-Abgabeseite.

## Download des Modells:  

GitHub-Repo:
https://github.com/Mxitermi/CineTrack

Google Drive (nur Modell)
https://drive.google.com/file/d/1vKMGtvg0zM8HsO5BcHBJTYJHDb-G0B_R/view?usp=drive_link

Pfad im Code anpassen:
MODEL_PATH = "C:/data/python/CineTrack_Extras/Models/unet_model_03.pth"
