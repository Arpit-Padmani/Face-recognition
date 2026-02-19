import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageTk
import tkinter as tk

path = 'photos'
images = []
classNames = []

# Load images from the folder
if not os.path.exists(path): os.makedirs(path)
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(images, names):
    encodeList = []
    valid_images, valid_names = [], []
    for i, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img_rgb)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
            valid_images.append(images[i])
            valid_names.append(names[i])
    return encodeList, valid_images, valid_names

print('Encoding started...')
encodeListKnown, images, classNames = findEncodings(images, classNames)
print(f'Encoding complete. Loaded {len(encodeListKnown)} faces.')

cap = cv2.VideoCapture(0)

def update_webcam():
    """Continuously updates the Tkinter label with the live webcam feed."""
    success, img = cap.read()
    if success:
        # Resize for the UI display
        img_display = cv2.resize(img, (400, 300))
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        webcam_label.imgtk = img_tk
        webcam_label.configure(image=img_tk)
    
    # Repeat after 10ms
    webcam_label.after(10, update_webcam)

def take_and_match():
    success, img = cap.read()
    if not success: return

    # Scale down for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    match_results = []

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # --- YOUR SPECIFIC MATCHING LOGIC ---
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            
            # Drawing logic for visual feedback (even if just captured)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Prepare data for Grid Display
        for i in range(len(faceDis)):
            match_results.append({
                "name": classNames[i],
                "dist": faceDis[i],
                "img": images[i]
            })

    # Sort results by distance (accuracy)
    match_results.sort(key=lambda x: x["dist"])
    display_results(match_results)

def display_results(sorted_data):
    for widget in grid_frame.winfo_children():
        widget.destroy()

    cols = 3
    for index, data in enumerate(sorted_data[:6]): # Limit to top 6
        r, c = index // cols, index % cols
        img_small = cv2.resize(data["img"], (120, 120))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

        cell = tk.Frame(grid_frame, bd=1, relief=tk.SOLID)
        cell.grid(row=r, column=c, padx=5, pady=5)

        img_lbl = tk.Label(cell, image=img_tk)
        img_lbl.image = img_tk
        img_lbl.pack()

        acc = round((1 - data["dist"]) * 100, 1)
        tk.Label(cell, text=f"{data['name']}\n{acc}%", font=("Arial", 8)).pack()

# --- TKINTER UI ---
root = tk.Tk()
root.title("AI Face Recognition System")
root.geometry("800x600")

# Left Side: Webcam
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

webcam_label = tk.Label(left_frame, text="Loading Camera...")
webcam_label.pack()

btn_capture = tk.Button(left_frame, text="TAKE PHOTO", command=take_and_match, 
                        bg="#2196F3", fg="white", font=("Arial", 12, "bold"), pady=10)
btn_capture.pack(fill=tk.X, pady=10)

# Right Side: Results Grid
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(right_frame, text="Search Results (Accuracy)", font=("Arial", 12, "bold")).pack()
grid_frame = tk.Frame(right_frame)
grid_frame.pack()

# Start the webcam loop
update_webcam()

root.mainloop()
cap.release()