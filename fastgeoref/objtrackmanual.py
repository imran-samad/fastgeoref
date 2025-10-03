# objtrackmanual.py
# This script allows users to manually click on objects in the video and obtain their pixel coordinates and geo-referenced coordinates

print(">>> objtrackmanual module imported")

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import time
import pandas as pd
from .tracking_functions import pix_to_pos, get_flight_details


def run_manual_tracking(params):
    """Main function to run manual object tracking with user inputs from GUI"""

    # --- Extract inputs ---
    filename = params["video_filename"]
    video_path = params["video_path"] + "/"
    logs_path = params["log_path"] + "/"
    tracks_path = params["tracks_path"] + "/"
    log_filename = params["log_filename"] + ".csv"

    # Open video
    cap = cv2.VideoCapture(f"{video_path}/{filename}.mp4")

    # Initialize variables
    pt = np.empty((0, 2), dtype=int)  # Store pixel coords
    time_elap = []  # Store time of frames clicked
    window_resize_factor = inputs['window_resize_factor'] #This helps adjust the size of the video display. A 4K video (2160p) will need to be resized (only for viewing) to be displayed in a HD (1080p) screen
    image_width = inputs['h_pix']
    image_height = inputs['v_pix']
    frame_rate = inputs['frame_rate']
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Click event callback ---
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal pt, time_elap, elapsed_time, frame
            pt = np.concatenate((pt, [(round(x * window_resize_factor),
                                       round(y * window_resize_factor))]))
            time_elap.append(round(elapsed_time, 2))

            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame,
                        f"{round(x*window_resize_factor)},{round(y*window_resize_factor)}",
                        (x, y),
                        font, 1.3, (0, 0, 0), 1)
            cv2.imshow("image", frame)

    # --- Video playback with pause/click ---
    playing = True
    current_frame = 0
    tot_time = time.strftime("%M:%S", time.gmtime(total_frames / frame_rate))

    while True:
        try:
            if playing:
                ret, frame = cap.read()
                elapsed_time = current_frame / frame_rate
                current_time = time.strftime("%M:%S", time.gmtime(elapsed_time))
                if not ret:
                    break
    
                # Display time overlay
                cv2.putText(frame, current_time, (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                cv2.putText(frame, "|" + tot_time, (290, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                frame = cv2.resize(frame,
                                   (int(image_width / window_resize_factor),
                                    int(image_height / window_resize_factor)))
                cv2.imshow("video", frame)
                current_frame += 1
    
            key = cv2.waitKey(1)
    
            if key == 27:  # esc = quit
                break
    
            elif key == 32:  # spacebar = pause and click
                playing = not playing
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image",
                                 int(image_width / window_resize_factor),
                                 int(image_height / window_resize_factor))
                cv2.imshow("image", frame)
                cv2.setMouseCallback("image", click_event)
                k1 = cv2.waitKey(0)
                if k1 != -1:
                    cv2.destroyWindow("image")
                    playing = not playing
    
            elif playing:
                if key == ord("z"):  # rewind 5s
                    current_frame = max(0, current_frame - 5 * frame_rate)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                elif key == ord("x"):  # forward 5s
                    current_frame = min(total_frames,
                                        current_frame + 5 * frame_rate)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        except KeyboardInterrupt:
            print("⛔ Interrupted by user, shutting down...")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # --- Save raw tracks ---
    tracked_dtns = pd.DataFrame({
        "x_image": pt[:, 0],
        "y_image": pt[:, 1],
        "video_time": time_elap
    })
    tracked_dtns.to_csv(f"{tracks_path}/{filename}_tracks_raw.csv",
                        index=False)

    # --- Geo-referencing ---
    df_video, df_flight, srt_dt = get_flight_details(
        filename, video_path, logs_path, log_filename
    )
    data_pos = pix_to_pos(
        tracked_dtns, df_flight, srt_dt,
        time_off=inputs['time_off'], height_ft_off=inputs['height_ft_off'], log_freq = inputs['log_freq'], h_pix=inputs['h_pix'], v_pix=inputs['v_pix'], h_fov=inputs['h_fov'], v_fov=inputs['v_fov']
    )

    tracked_dtns1 = pd.concat(
        [tracked_dtns.reset_index(drop=True),
         data_pos.reset_index(drop=True)], axis=1
    )
    tracked_dtns1.to_csv(
        f"{tracks_path}/{filename}_tracks_geo-referenced.csv", index=False
    )

    messagebox.showinfo("Success", "Manual tracking and geo-referencing complete!")


inputs = {}

def ask_user_inputs():
    """Tkinter form for manual object tracking with file/folder browsing"""

    import os
    root = tk.Tk()
    root.title("Manual Object Tracking")

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

    def select_log():
        path = filedialog.askopenfilename(
            title="Select Flight Log CSV", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            log_entry.delete(0, tk.END)
            log_entry.insert(0, os.path.dirname(path))

            # Auto-fill the filename (without extension)
            filename_log_entry.delete(0, tk.END)
            filename_log_entry.insert(0, os.path.splitext(os.path.basename(path))[0])

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
    tk.Label(form, text="Video Filename:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    filename_entry = tk.Entry(form, width=50)
    filename_entry.grid(row=1, column=1, padx=5, pady=5)
    inputs["video_filename"] = filename_entry

    # --- Log file ---
    tk.Label(form, text="Log Path:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    log_entry = tk.Entry(form, width=50)
    log_entry.grid(row=2, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_log).grid(row=2, column=2, padx=5, pady=5)
    inputs["log_path"] = log_entry
    
    # --- Log filename (auto-filled) ---
    tk.Label(form, text="Log Filename:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    filename_log_entry = tk.Entry(form, width=50)
    filename_log_entry.grid(row=3, column=1, padx=5, pady=5)
    inputs["log_filename"] = filename_log_entry

    # --- Tracks output folder ---
    tk.Label(form, text="Tracks Output Folder:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
    output_entry = tk.Entry(form, width=50)
    output_entry.grid(row=4, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_output_dir).grid(row=4, column=2, padx=5, pady=5)
    inputs["tracks_path"] = output_entry

    # --- Default parameter values ---
    params = {
        "window_resize_factor": 2.5,
        "time_off": 0,
        "height_ft_off": 0,
        "log_freq": 5, 
        "h_pix": 3840,
        "v_pix": 2160,
        "frame_rate": 30,
        "h_fov": 70,
        "v_fov": 40
    }
    
    # --- Parameters section ---
    row = 7
    for key, default in params.items():
        tk.Label(form, text=f"{key}:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(form)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=5, pady=5)
        inputs[key] = entry
        row += 1

    # --- Run button ---
    def submit():
        for key, entry in inputs.items():
            # Convert numbers where appropriate
            val = entry.get()
            try:
                inputs[key] = float(val) if "." in val else int(val)
            except ValueError:
                inputs[key] = val  # keep as string if not numeric
        root.destroy()
        run_manual_tracking(inputs)

    tk.Button(form, text="Run Manual Tracking", command=submit,
              bg="green", fg="white", padx=10, pady=5).grid(row=row, column=1, pady=10)

    root.mainloop()
    return inputs

#if __name__ == "__main__":
def main():
    ask_user_inputs()