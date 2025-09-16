# datagen.py
# This script extracts animal and background data of set sizes (images) for model training and stores them in a folder defined at the start of the code

print(">>> datagen module imported")

import cv2
import os
import time
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def run_extraction(filename, video_path, output_directory,
                   window_resize_factor, frame_rate,
                   image_width, image_height, grabsize):

    video_file = os.path.join(video_path, filename + ".mp4")
    cap = cv2.VideoCapture(video_file)
    os.makedirs(output_directory, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Variables for mouse event
    # Variables to track mouse events
    drawing = False
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    def click_event(event, x, y, flags, param):
        nonlocal start_x, start_y, end_x, end_y, drawing, frame
    
        if event == cv2.EVENT_LBUTTONDOWN:
            roi = frame_org[int((y-grabsize)*window_resize_factor):int((y+grabsize)*window_resize_factor),int((x-grabsize)*window_resize_factor):int((x+grabsize)*window_resize_factor)]
            cv2.rectangle(frame, (x-grabsize,y-grabsize), (x+grabsize,y+grabsize), (0,255,0),1)
            cv2.imshow('image', frame)
            image_name = f'{filename}_{str(round(elapsed_time))}_{len(os.listdir(output_directory))}.png'
            if roi.size > 0:
                cv2.imwrite(os.path.join(output_directory, image_name), roi)

    # Video playback
    playing = True
    current_frame = 0
    tot_time = time.strftime("%M:%S", time.gmtime(total_frames/frame_rate))

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                break
            elapsed_time = current_frame/frame_rate
            current_time = time.strftime("%M:%S", time.gmtime(elapsed_time))

            # Time overlay
            cv2.putText(frame, current_time, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(frame, "|"+tot_time, (290, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

            frame_org = frame
            frame = cv2.resize(frame, (int(image_width/window_resize_factor),
                                       int(image_height/window_resize_factor)))
            cv2.imshow("video", frame)
            current_frame += 1

        key = cv2.waitKey(1)

        if key == 27:  # esc = quit
            sys.exit(0)
            break

        elif key == 32:  # space = pause/play + crop
            playing = not playing
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', int(image_width/window_resize_factor),
                             int(image_height/window_resize_factor))
            cv2.imshow('image', frame)
            cv2.setMouseCallback('image', click_event)
            k1 = cv2.waitKey(0)
            if k1 != -1:
                cv2.destroyWindow('image')
                playing = not playing

        elif playing:
            if key == ord('z'):  # rewind 5s
                current_frame = max(0, current_frame - 5 * frame_rate)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == ord('x'):  # forward 10s
                current_frame = min(total_frames, current_frame + 10 * frame_rate)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    cap.release()
    cv2.destroyAllWindows()


def gui_entry():
    """Single Tkinter form for entering all variables before running extraction"""
    root = tk.Tk()
    root.title("FastGeoRef - Extract Training Data")

    inputs = {}
    
    def select_video():
        path = filedialog.askopenfilename(
            title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")]
        )
        if path:
            video_entry.delete(0, tk.END)
            video_entry.insert(0, os.path.dirname(path))

            # Auto-fill the filename (without extension)
            filename_entry.delete(0, tk.END)
            filename_entry.insert(0, os.path.splitext(os.path.basename(path))[0])
    
    def select_output_dir():
        path = filedialog.askdirectory(title="Select Tracks Output Folder")
        if path:
            output_entry.delete(0, tk.END)
            output_entry.insert(0, path)
    
    form = tk.Frame(root, padx=10, pady=10)
    form.pack()

    # --- Video file ---
    tk.Label(form, text="Video Path:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    video_entry = tk.Entry(form, width=50)
    video_entry.grid(row=0, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_video).grid(row=0, column=2, padx=5, pady=5)
    inputs["video_path"] = video_entry

    # --- Video filename (auto-filled) ---
    tk.Label(form, text="Video Filename (auto-added):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    filename_entry = tk.Entry(form, width=50)
    filename_entry.grid(row=1, column=1, padx=5, pady=5)
    inputs["video_filename"] = filename_entry
    
    # --- Tracks output folder ---
    tk.Label(form, text="Tracks Output Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    output_entry = tk.Entry(form, width=50)
    output_entry.grid(row=2, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_output_dir).grid(row=2, column=2, padx=5, pady=5)
    inputs["tracks_path"] = output_entry

    # --- Default parameter values ---
    params = {
        "window_resize_factor": 2.5, # This helps adjust the size of the video display. A 4K video (2160p) will need to be resized to a lower resolution (only for viewing) to be displayed in a HD (1080p) screen
        "frame_rate": 30,
        "image_width": 3840,
        "image_height": 2160,
        "grabsize": 15 # This will lead to a box of resolution grabsize x 2 pixels (in the resized frame). Set a grab size appropriate to the size of the object/animal in the video
        }
    
    # --- Parameters section ---
    row = 3
    for key, default in params.items():
        tk.Label(form, text=f"{key}:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(form)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=5, pady=5)
        inputs[key] = entry
        row += 1

    # --- Run button ---
    def run_clicked():
        try:
            for key, entry in inputs.items():
                # Convert numbers where appropriate
                val = entry.get()
                try:
                    inputs[key] = float(val) if "." in val else int(val)
                except ValueError:
                    inputs[key] = val  # keep as string if not numeric
            root.destroy()
            
            filename = inputs["Filename"].get()
            video_path = inputs["Video path"].get()
            output_directory = inputs["Output directory"].get()
            window_resize_factor = float(inputs["Window resize factor"].get())
            frame_rate = int(inputs["Frame rate"].get())
            image_width = int(inputs["Image width (px)"].get())
            image_height = int(inputs["Image height (px)"].get())
            grabsize = int(inputs["Grabsize (px)"].get())
            
            run_extraction(filename, video_path, output_directory,
                           window_resize_factor, frame_rate,
                           image_width, image_height, grabsize)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    tk.Button(root, text="Run Extraction", command=run_clicked, bg="green", fg="white").pack(pady=10)
    root.mainloop()

    return inputs

#if __name__ == "__main__":
def main():
    gui_entry()