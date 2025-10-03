# objtrackauto.py
# This script performs object detection on each frame of a video using the model built using the 'modelbuild.py' script. Potential areas where an object may be are identified as blobs or keypoints where model inference is performed. Detections with high confidence scores are marked with dots on a new frame which is stored as a video in an output directory

print(">>> objtrackauto module imported")

import pandas as pd
import cv2
import numpy as np
import math as m
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from .tracking_functions import yolo
import threading

# ---------------------------
# Tkinter GUI for user inputs
# ---------------------------
def get_inputs():
    root = tk.Tk()
    root.title("ObjTrackAuto Settings")

    final_inputs = {}

    def select_video():
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if path:
            video_entry.delete(0, tk.END)
            video_entry.insert(0, path)
            
    # def select_model():
    #     path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.h5;*h5py")])
    #     if path:
    #         model_entry.delete(0, tk.END)
    #         model_entry.insert(0, path)
            
    def select_model():        
        path = filedialog.askdirectory()
        if path:
            model_entry.delete(0, tk.END)
            model_entry.insert(0, path)

    def select_output_dir():
        path = filedialog.askdirectory()
        if path:
            output_entry.delete(0, tk.END)
            output_entry.insert(0, path)

    tk.Label(root, text="Video File:").grid(row=0, column=0, sticky="e")
    video_entry = tk.Entry(root, width=50)
    video_entry.grid(row=0, column=1)
    tk.Button(root, text="Browse", command=select_video).grid(row=0, column=2)

    tk.Label(root, text="Model Folder (don't browse inside the folder):").grid(row=1, column=0, sticky="e")
    model_entry = tk.Entry(root, width=50)
    model_entry.grid(row=1, column=1)
    tk.Button(root, text="Browse", command=select_model).grid(row=1, column=2)

    tk.Label(root, text="Output Directory:").grid(row=2, column=0, sticky="e")
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=2, column=1)
    tk.Button(root, text="Browse", command=select_output_dir).grid(row=2, column=2)

    # Parameters
    params = {
        "window_resize_factor": 2.5, # This helps adjust the size of the video display. A 4K video (2160p) will need to be resized to a lower resolution (only for viewing) to be displayed in a HD (1080p) screen
        "grabsize": 15, # must be same as the grabsize in datagen script
        # min and max value for thresholding the grayscaled frame (see Rathore et al., 2023); if not known set vales as 0
        "threshold_value1": 0,
        "threshold_value2": 255,
        
        "threshold_step": 5, # the value by which threshold_value1 is increased until it reaches threshold_value2 during blob detection
        
        "minAreaThresh": 200, # approx. min size of the area (in pixel) occupied by each animal (this minimizes multiple detections of a single animal; e.g., different body parts identified as different individuals)
        "maxAreaThresh": 5000, # approx. max size of the area (in pixel) occupied by each animal (this minimizes detection of 2 or more animals as one)
        "detection_start_time": 0, # the time in seconds from where detections need to start
        "frame_rate": 30 #fps
    }
    
    param_entries = {}
    row = 3
    for key, default in params.items():
        tk.Label(root, text=f"{key}:").grid(row=row, column=0, sticky="e")
        entry = tk.Entry(root)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1)
        param_entries[key] = entry
        row += 1

    def submit():
        # Collect all values into a clean dictionary
        final_inputs["video_file"] = video_entry.get()
        final_inputs["model_folder"] = model_entry.get()
        final_inputs["output_dir"] = output_entry.get()
        for key, entry in param_entries.items():
            val = entry.get()
            try:
                final_inputs[key] = float(val) if "." in val else int(val)
            except ValueError:
                final_inputs[key] = val
        root.destroy()

    tk.Button(root, text="Start", command=submit).grid(row=row, column=1, pady=10)
    root.mainloop()

    return final_inputs

# ---------------------------
# Main processing script
# ---------------------------

def process_video(settings):
    #settings = get_inputs()

    movie_name = settings["video_file"]
    model_name = settings["model_folder"]
    output_dir = settings["output_dir"]

    filename = movie_name.split("/")[-1].split(".")[0]  # derive from video filename

    # Initialise variables
    window_resize_factor = settings["window_resize_factor"]
    grabsize = settings["grabsize"]
    threshold_value1 = settings["threshold_value1"]
    threshold_value2 = settings["threshold_value2"]
    threshold_step = settings["threshold_step"]
    minAreaThresh = settings["minAreaThresh"]
    maxAreaThresh = settings["maxAreaThresh"]
    detection_start_time = settings["detection_start_time"]
    frame_rate = settings["frame_rate"]

    cap = cv2.VideoCapture(movie_name)

    nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Nx, Ny = int(nx / 2), int(ny / 2) #Set resolution for the final video to be saved. Here its set to be half the size of the original video. 

    df = pd.DataFrame(columns=['c_id','x_px','y_px','frame']) # dataframe to store the number of detections and their coordinates in each frame

    #Define a tracker to initiate and continue tracks; max_age is the maximum duration after which an undetected ID is dropped; track_threshold: the minimum average confidence of a track; init_threshold: the minimum confidence in a new detection required to start a track; init_nms: threshold to initialise a track based on a detection’s overlap with another; link_iou: threshold to link detections based on proportion overlap
    tracker = yolo.yoloTracker(max_age=60, track_threshold=0.6, init_threshold=0.6, init_nms=0.0, link_iou=0.01)

    bb_model = load_model(model_name) # if this does not work, use 'from tensorflow.keras.models import load_model'

    out = cv2.VideoWriter(
        f"{output_dir}/{filename}_det.avi",
        cv2.VideoWriter_fourcc('M','J','P','G'),
        30,
        (nx, ny)
    ) # save the output video
    
    i = detection_start_time*frame_rate # starting frame number of the video from where predictions are to be made
    row = 0 # counter for total number of confident detections
    data_all = pd.DataFrame([]) #dataframe to store all detections and their coordinates throughout the video
    tracked_dtns = np.empty((0, 13), dtype=float)
    while cap.isOpened() & (i < nframe):
        print(f"[UPDATING.....]{i}/{nframe}th frame detected and stored", flush=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
    
        # Convert frame to grayscale
        grayF = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #remove noise
        gray = cv2.medianBlur(grayF,5)
        #Invert image
        gray = cv2.bitwise_not(gray)
        
        # Blob detection in grayscaled image
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
      
        # Change thresholds
        params.minThreshold = threshold_value1
        params.maxThreshold = threshold_value2
        params.thresholdStep = threshold_step
      
        # Filter by Area
        params.filterByArea = True
        params.minArea = minAreaThresh
        params.maxArea = maxAreaThresh
        
        # Filter by Circularity
        params.filterByCircularity = False
          
        # Filter by Convexity
        params.filterByConvexity = False
          
        # Filter by Inertia
        params.filterByInertia = False
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
      
        # Detect blobs on the grayscaled image
        keypoints = detector.detect(gray)
        
        #storing locations of detected objects
        data = pd.DataFrame([]) # to store all detections in a frame
        if keypoints:
            testX = np.ndarray(
                shape=(len(keypoints), int(grabsize*window_resize_factor*2), int(grabsize*window_resize_factor*2), 3),
                dtype='uint8'
            )
            j = 0 # counter for number of keypoints detected
            for keyPoint in keypoints:
                ix, iy = keyPoint.pt # coordinates of the jth blob
                tmpImg = frame[max(0,int(iy-grabsize*window_resize_factor)):min(ny,int(iy+grabsize*window_resize_factor)),
                               max(0,int(ix-grabsize*window_resize_factor)):min(nx,int(ix+grabsize*window_resize_factor))].copy() #extract image in the blob area
                tmpImg1 = cv2.resize(tmpImg, (int(grabsize*window_resize_factor*2), int(grabsize*window_resize_factor*2))) # rescale extracted image to original resolution
                testX[j] = tmpImg1
                j += 1
    
            testX = testX.reshape(-1, 75, 75, 3).astype('float32') / 255. #resize image to the resolution at which the model was built
            pred = bb_model.predict(testX) #predict the presence/absence of animals
    
            #Identify confident detections
            j=0 # reinitialising j; counter for number of confident detections
            FKP = [] # to store confident detections
            indx=[] # to store the index of confident detections
            for pr in pred:
                if pr[1] > 0.7: #only use detections with confidence score > 0.7 to minimise false positives
                    row += 1 # counter for total number of confident detections
                    df.loc[row] = [j, keypoints[j].pt[0], keypoints[j].pt[1], i]
                    FKP.append(keypoints[j])
                    indx.append(j)
                j += 1
    
            pts = [(m.floor(ii.pt[0]), m.floor(ii.pt[1]), round(ii.size)) for ii in FKP] #size is the approximate diameter in pixels
            for k, item in enumerate(pts):
                dt = pd.DataFrame({
                    'obj': [k],
                    'frame': [i],
                    'x_image': [round(item[0]-grabsize), round(item[0]-grabsize), round(item[0]+grabsize), round(item[0]+grabsize)],
                    'y_image': [round(item[1]+grabsize), round(item[1]-grabsize), round(item[1]-grabsize), round(item[1]+grabsize)],
                    'prb': [pred[indx[k],1]],
                    'video_time': [round(i/30,2)]
                }, index=[0,1,2,3]) # dt will have four rows representing the bottom-left, top-left, top-right, bottom-right of the detection box
                data = pd.concat([data, dt], ignore_index=True)
                data_all = pd.concat([data_all, data], ignore_index=True)
    
        dtn = np.empty((0, 9), dtype=float)  # to store the detections as an array below
        if len(data) != 0: #if there is at least one confident detection
            for ii in range(0,len(data),4): # each detection takes up for rows, so the loop is increased by 4 values at every iteration
                dtn = np.concatenate((dtn, np.array([
                    data['x_image'][ii], data['y_image'][ii],
                    data['x_image'][ii+1], data['y_image'][ii+1],
                    data['x_image'][ii+2], data['y_image'][ii+2],
                    data['x_image'][ii+3], data['y_image'][ii+3],
                    data['prb'][ii]
                ]).reshape(1,-1)))
            tracks, tracked_dtns = tracker.update(np.asarray(dtn), tracked_dtns, i)
    
            if len(tracked_dtns) > 0:
                tracks_disp = tracked_dtns[np.ix_(np.where(tracked_dtns[:,12]==i)[0],[0,1,4,5,9])] # copying detections in current frame (4 pixel coordinates (0,5,4,1) + ID (8))
                for ids in tracks_disp:
                    np.random.seed(int(ids[4])) # show each track as its own colour
                    r,g,b = np.random.randint(256), np.random.randint(256), np.random.randint(256)
                    cv2.circle(frame, (int((ids[0]+ids[2])/2), int((ids[1]+ids[3])/2)), 0, (b,g,r), 15) # place a small dot on the object detected
                    cv2.putText(frame, str(int(ids[4])),
                                (int((ids[0]+ids[2])/2+10), int((ids[1]+ids[3])/2)),
                                0, 5e-3 * 200, (b, g, r), 2) #write the unique ID of the object
    
        new_frame = cv2.resize(frame, (Nx, Ny))
        out.write(new_frame)
        i += 1
    
    cap.release()
    out.release()

    tracked_dtns = pd.DataFrame(tracked_dtns)
    tracked_dtns.to_csv(f"{output_dir}/{filename}_tracks_raw.csv", index=False)

#if __name__ == "__main__":
def main():
    settings = get_inputs()
    t = threading.Thread(target=process_video, args=(settings,))
    t.start()
