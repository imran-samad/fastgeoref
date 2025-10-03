# georef.py

# This script geo-references each detection saved during the previous step i.e., it uses output from the 'objtrackauto' script.

print(">>> georef module imported")

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from .tracking_functions import pix_to_pos, get_flight_details


def run_georef(params):
    """
    Geo-references each detection saved during object tracking.
    """
    # --- Extract inputs ---
    filename = params["video_filename"]
    video_path = params["video_path"] + "/"
    log_filename = params["log_filename"] + ".csv"
    log_path = params["log_path"] + "/"
    track_path = params["track_path"] + "/"
    track_filename = params["track_filename"] + ".csv"

    # Load raw tracking results
    tracked_dtns = pd.read_csv(f"{track_path}{track_filename}")

    # Obtain flight information
    df_video, df_flight, srt_dt = get_flight_details(
        filename, video_path, log_path, log_filename
    )

    # Subsetting detections for geo-referencing
    counts = tracked_dtns.iloc[:, 9].value_counts()  #frequency of each detection
    tracked_dtns_filtered = tracked_dtns[
        tracked_dtns.iloc[:, 9].isin(counts[counts > 60].index)
    ] #removing sporadic FPositives i.e., with <2 seconds (60 frames) of detections
    tracked_dtns_filtered.loc[:, "time_s"] = (
        (tracked_dtns_filtered.iloc[:, 12]) / 30
    ).apply(lambda x: np.floor(x)) # rounding off time to seconds for grouping (below)

    tracked_dtns_filtered1 = (
        tracked_dtns_filtered.groupby(
            [tracked_dtns_filtered.iloc[:, 9], "time_s"], group_keys=False
        )
        .apply(lambda x: x.head(1))
        .reset_index(drop=True)
    ) # Filter to keep one detection per second for each ID (column 9)

    # Set up pixel coordinates
    grabsize=params['grabsize']
    data_pix = pd.DataFrame(
        {
            "x_image": grabsize + tracked_dtns_filtered1.iloc[:, 0],
            "y_image": grabsize + tracked_dtns_filtered1.iloc[:, 5],
            "video_time": tracked_dtns_filtered1.iloc[:, 12] / 30,
        }
    )

    # Convert pixel positions to geo-coordinates
    data_pos = pix_to_pos(
        data_pix,
        #df_video,
        df_flight,
        srt_dt,
        time_off=params['time_off'],
        height_ft_off=params['height_ft_off'],
        log_freq = params['log_freq'],
        h_pix=params['h_pix'],
        v_pix=params['v_pix'],
        h_fov=params['h_fov'],
        v_fov=params['v_fov']
    )
    #pix and fov refer to the pixel and field-of-view resolution of the video/drone; h and v are horizontal and vertical components respectively.

    # Merge and save
    tracked_dtns1 = pd.concat(
        [tracked_dtns_filtered1.reset_index(drop=True), data_pos.reset_index(drop=True)],
        axis=1,
    )
    
    tracked_dtns1 = tracked_dtns1.rename(columns={
    tracked_dtns1.columns[0]: 'pix_coord_tl_x',
    tracked_dtns1.columns[1]: 'pix_coord_tl_y',
    tracked_dtns1.columns[2]: 'pix_coord_bl_x',
    tracked_dtns1.columns[3]: 'pix_coord_bl_y',
    tracked_dtns1.columns[4]: 'pix_coord_br_x',
    tracked_dtns1.columns[5]: 'pix_coord_br_y',
    tracked_dtns1.columns[6]: 'pix_coord_tr_x',
    tracked_dtns1.columns[7]: 'pix_coord_tr_y',
    tracked_dtns1.columns[8]: 'ID_prob',
    tracked_dtns1.columns[9]: 'ID',
    tracked_dtns1.columns[10]: 'Tracking_YN',
    tracked_dtns1.columns[11]: 'time_since_update',
    tracked_dtns1.columns[12]: 'frame_no'
    })
    
    output_file = f"{track_path}/{filename}_tracks_geo-referenced.csv"
    tracked_dtns1.to_csv(output_file, index=False)

    return output_file

inputs = {}

def ask_georef_inputs():
    """Tkinter form for geo-referencing with file/folder browsing"""
    import os
    root = tk.Tk()
    root.title("Geo-referencing Setup")

    def select_video_dir():
        path = filedialog.askopenfilename(
            title="Select video file", filetypes=[("Video files", "*.mp4;*.avi")]
        )
        if path:
            video_entry.delete(0, tk.END)
            video_entry.insert(0, os.path.dirname(path))
            
            # Auto-fill the filename (without extension)
            filename_video_entry.delete(0, tk.END)
            filename_video_entry.insert(0, os.path.splitext(os.path.basename(path))[0])

    def select_tracks_dir():
        path = filedialog.askopenfilename(
            title="Select tracks file", filetypes=[("CSV", "*.csv")]
        )
        if path:
            tracks_entry.delete(0, tk.END)
            tracks_entry.insert(0, os.path.dirname(path))
            # Auto-fill the filename (without extension)
            filename_tracks_entry.delete(0, tk.END)
            filename_tracks_entry.insert(0, os.path.splitext(os.path.basename(path))[0])

    def select_logs_dir():
        path = filedialog.askopenfilename(
            title="Select log file", filetypes=[("CSV", "*.csv")]
        )
        if path:
            log_entry.delete(0, tk.END)
            log_entry.insert(0, os.path.dirname(path))
            # Auto-fill the filename (without extension)
            filename_log_entry.delete(0, tk.END)
            filename_log_entry.insert(0, os.path.splitext(os.path.basename(path))[0])

    form = tk.Frame(root, padx=10, pady=10)
    form.pack()
    
    # --- Video folder ---
    tk.Label(form, text="Video Path:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    video_entry = tk.Entry(form, width=50)
    video_entry.grid(row=0, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_video_dir).grid(row=0, column=2, padx=5, pady=5)
    inputs["video_path"] = video_entry
    
    # --- Video filename (auto-filled) ---
    tk.Label(form, text="Video Filename:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    filename_video_entry = tk.Entry(form, width=50)
    filename_video_entry.grid(row=1, column=1, padx=5, pady=5)
    inputs["video_filename"] = filename_video_entry
    
    # --- Tracks folder ---
    tk.Label(form, text="Tracks Path:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    tracks_entry = tk.Entry(form, width=50)
    tracks_entry.grid(row=2, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_tracks_dir).grid(row=2, column=2, padx=5, pady=5)
    inputs["track_path"] = tracks_entry
    
    # --- Track filename (auto-filled) ---
    tk.Label(form, text="Track Filename:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    filename_tracks_entry = tk.Entry(form, width=50)
    filename_tracks_entry.grid(row=3, column=1, padx=5, pady=5)
    inputs["track_filename"] = filename_tracks_entry
    
    # --- Logs folder ---
    tk.Label(form, text="Logs Path:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
    log_entry = tk.Entry(form, width=50)
    log_entry.grid(row=4, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_logs_dir).grid(row=4, column=2, padx=5, pady=5)
    inputs["log_path"] = log_entry
    
    # --- Log filename (auto-filled) ---
    tk.Label(form, text="Log Filename:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
    filename_log_entry = tk.Entry(form, width=50)
    filename_log_entry.grid(row=5, column=1, padx=5, pady=5)
    inputs["log_filename"] = filename_log_entry
    
    # --- Default parameter values ---
    parms = {
        "grabsize": 15,
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
    row = 6
    for key, default in parms.items():
        tk.Label(form, text=f"{key}:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(form)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=5, pady=5)
        inputs[key] = entry
        row += 1
    
    # --- Run button ---
    def submit():
        for key, entry in inputs.items():
            val = entry.get()
            try:
                inputs[key] = float(val) if "." in val else int(val)
            except ValueError:
                inputs[key] = val  # keep as string if not numeric
        root.destroy()
        run_georef(inputs)
    
    tk.Button(form, text="Run Geo-referencing", command=submit,
              bg="green", fg="white", padx=10, pady=5).grid(row=row, column=1, pady=10)
    
    root.mainloop()

#if __name__ == "__main__":
def main():
    ask_georef_inputs()