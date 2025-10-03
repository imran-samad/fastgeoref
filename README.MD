# fastgeoref: Animal Detection, Tracking, and Geo-referencing from Drone Video  
**Version:** 1.0.0  
**Authors:** Imran Samad (imransamad7@gmail.com)  

---

## Package Overview
FastGeoRef is a Python package designed to streamline the process of detecting, tracking, and geo-referencing objects (specifically animals) from drone video footage. It offers both automated and manual workflows, allowing flexibility based on your specific tracking needs. The package is built to facilitate the development of detection models, apply them for tracking, and convert pixel-level detections into precise geographic coordinates. The core functionalities are implemented as five distinct Python scripts, each accessible via a Graphical User Interface (GUI) once the package is installed. Details can be found in [1], with a worked out case study in [2].

**References**  
[1] Samad, I., Sutaria, D., Farine D. R., Shanker, K. & Cantor M. (2025). Integrated deep learning and geo-referencing for flexible drone-based animal tracking. *bioRxiv*. https://doi.org/10.1101/2025.02.05.636599  

[2] Samad, I., Patil, H., Cantor M., Farine D. R., Sutaria, D. & Shanker, K. (2025). Human activities drive novel behaviours and transitions in dolphins. *bioRxiv*. https://doi.org/10.1101/2025.07.10.663819  

---

## Installation

The package is designed to work with python v3.9. Other versions may cause incompatibility issues and therefore creating a virtual environment through conda with python v3.9 is recommended

Step 1: To install (mini)conda visit https://www.anaconda.com/docs/getting-started/miniconda/install

Step 2: Run the anaconda prompt

Step 3 create and activate a virtual environment using the command

```bash
conda create -n <env_name> python==3.9
conda activate <env_name>
```

Step 4: install the FastGeoRef package

```bash
pip install git+https://github.com/imran-samad/fastgeoref.git
```

---

## Key Features

- Automated and Manual Tracking: Choose between developing a machine learning model for automatic detection or manually clicking on objects of interest.  
- CNN-based Detection: Build and utilise Convolutional Neural Networks for robust object identification.  
- Geo-referencing: Accurately convert pixel coordinates from video frames to real-world geographic coordinates using drone flight log data.  
- Modular Workflow: A clear, step-by-step process allows users to generate training data, build models, perform tracking, and geo-reference detections.  
- User-Friendly GUIs: Each core function is launched with an intuitive GUI for inputting parameters and controlling the process.  

---

## General Workflow

The overall workflow for animal tracking typically follows these paths:

### 1. Automated Tracking
1. Data Generation (**fastgeoref-datagen**): Extract image data from videos to create a training dataset.
2. Model Building (**fastgeoref-modelbuild**): Train a CNN model for object detection.
3. Automated Object Tracking (**fastgeoref-objtrackauto**): Use the trained model to automatically detect and track animals in videos.  
4. Geo-referencing (**fastgeoref-georef**): Convert the tracked pixel coordinates into geographic coordinates.

### 2. Manual Tracking
5. Manual Object Tracking & Geo-referencing (**fastgeoref-objtrackmanual**): Manually click on animals in the video to track them, and simultaneously geo-reference these manual detections.

An example dataset (`DJI_0001.MP4`), script (`example_script.py`), and output files are included to demonstrate functionality. A pre-trained model for humpback dolphin detection is also provided in the `model` folder.  

---

## Dependencies

The package requires Python 3.9+ and:

- numpy==1.26.0  
- scikit-learn==1.3.2  
- pandas==2.1.1  
- filterpy==1.4.5  
- pysrt  
- shapely==2.0.6  
- scipy==1.11.3  
- tensorflow==2.14.0  
- opencv-python

---

## Core Functions of FastGeoRef

The package's five core functions can be accessed via command-line entry points, which launch GUIs. Details of all function arguments are provided in FastGeoRef_functions.docx file. FAQs and solutions to common issues are provided in the FastGeoRef_FAQ.docx file.

### 1. Data Generation
- **GUI Command:** `fastgeoref.run_datagen()`  
- **Summary:** Extract image data from videos to create a training dataset.  
- **Inputs (via GUI):** video file, paths, frame rate, grabsize, etc.  
- **Outputs:** cropped image patches (.png files).  
- **Controls:** Spacebar (pause), Mouse Click (save), z/x (rewind/forward), Esc (exit).  

### 2. Model Building
- **GUI Command:** `fastgeoref.run_modelbuild()`  
- **Summary:** Train a CNN model for object detection.  
- **Inputs:** training data path, model save path, image size, batch size, epochs.  
- **Outputs:** trained CNN model (`.h5py`).  

### 3. Object Detection & Tracking (Automated)
- **GUI Command:** `fastgeoref.run_auto_tracking()`  
- **Summary:** Automated detection + YOLO-based tracking.  
- **Inputs:** video, model file, output directory, thresholds, YOLO params.  
- **Outputs:** processed video (`*_det.avi`), raw tracks (`*_tracks_raw.csv`).  

### 4. Geo-referencing
- **GUI Command:** `fastgeoref.run_georef()`  
- **Summary:** Convert tracked pixels to latitude/longitude using logs.  
- **Inputs:** video info, tracks file, logs, offsets, FOV, resolutions.  
- **Outputs:** geo-referenced detections (`*_tracks_geo-referenced.csv`).  

### 5. Manual Object Tracking & Geo-referencing
- **GUI Command:** `fastgeoref.run_manual_tracking()`  
- **Summary:** Manual clicks to track & geo-reference animals.  
- **Inputs:** video info, logs, output folder, offsets, resolutions, FOV.  
- **Outputs:** raw CSV, geo-referenced CSV.  
- **Controls:** Spacebar (pause), Mouse Click (add point), z/x (rewind/forward), Esc (exit).  

---

## Helper Functions

Located in `tracking_functions.py`:

- **get_flight_details()**: Extracts flight data from video subtitles + logs.  
- **pix_to_pos()**: Converts pixel positions into geo-coordinates.  
- **YOLO class**: Kalman filter-based object tracker.  

---

## Working Example

Run the example pipeline:

```bash
python example script/example_script.py
```

Outputs for `DJI_0001.MP4` are included.  
Full dataset + outputs: [http://e.pc.cd/VBMy6alK](http://e.pc.cd/VBMy6alK)  

---
