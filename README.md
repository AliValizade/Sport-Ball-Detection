# ⚽ Sports Ball Detection using Image Processing

A computer vision project for detecting sports balls in noisy images using **Python, OpenCV, and classical image processing techniques**.

The project explores how traditional image processing methods can be combined to preprocess images, extract useful visual information, and detect circular objects such as sports balls.

---

## 📌 Overview

Detecting objects in real-world images can be challenging because of:

* Image noise
* Poor contrast
* Different lighting conditions
* Complex backgrounds
* Variations in object color and size

This project investigates a classical computer vision pipeline for sports ball detection without relying on machine learning or deep learning models.

The approach combines several image processing techniques and uses the **Hough Circle Transform** for detecting circular objects.

---

## 🎯 Project Goals

The main objectives of this project are to:

* Remove noise from input images
* Improve image quality and contrast
* Analyze dominant colors
* Extract useful image features
* Detect circular objects
* Explore the effectiveness of classical image processing techniques for object detection

---

## 🧠 Image Processing Pipeline

The general processing pipeline is:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Noise Reduction
     │
     ├── Gaussian Filter
     ├── Median Filter
     └── Adaptive Median Filter
     │
     ▼
Contrast Enhancement
     │
     └── Histogram Equalization
     │
     ▼
Color Analysis
     │
     └── Dominant Color Extraction
     │
     ▼
Circle Detection
     │
     └── Hough Circle Transform
     │
     ▼
Detected Sports Ball
```

---

## ✨ Features

* Noise reduction using:

  * Gaussian filtering
  * Median filtering
  * Adaptive median filtering
* Image contrast enhancement
* Histogram equalization
* Dominant color analysis
* Image restoration
* Circle detection using the Hough Circle Transform
* Modular image-processing implementation
* Python and OpenCV based implementation

---

## 🛠️ Technologies

* **Python 3**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* Classical Image Processing
* Hough Circle Transform

---

## 📂 Project Structure

```text
Sport-Ball-Detection/
│
├── img/
│   └── Input images
│
├── modules/
│   ├── median.py
│   ├── gaussian.py
│   ├── hist.py
│   ├── restore.py
│   └── hough_circle.py
│
├── output/
│   └── Generated results
│
├── main.py
├── .gitignore
└── README.md
```

### Main Components

| File / Directory          | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `img/`                    | Input images                                       |
| `modules/median.py`       | Median and adaptive median filtering               |
| `modules/gaussian.py`     | Gaussian filtering                                 |
| `modules/hist.py`         | Histogram equalization and dominant color analysis |
| `modules/restore.py`      | Image restoration operations                       |
| `modules/hough_circle.py` | Circle detection using Hough Transform             |
| `output/`                 | Generated processing results                       |
| `main.py`                 | Main application entry point                       |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/AliValizade/Sport-Ball-Detection.git
cd Sport-Ball-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python numpy matplotlib
```

---

## ▶️ Usage

Run the main script from the repository root:

```bash
python main.py
```

The application processes the configured input image and applies the image-processing pipeline for sports ball detection.

---

## 🔬 Methodology

### 1. Image Preprocessing

The input image is prepared for further processing.

Typical preprocessing operations include resizing and conversion to grayscale where required.

### 2. Noise Reduction

Different filtering techniques are explored to reduce image noise:

* Gaussian filter
* Median filter
* Adaptive median filter

The goal is to improve the quality of the image before object detection.

### 3. Contrast Enhancement

Histogram equalization is used to improve image contrast and make important visual structures easier to identify.

### 4. Dominant Color Analysis

Color information is analyzed to identify dominant visual characteristics of the image.

This can provide additional information that may help distinguish the target object from its surroundings.

### 5. Circle Detection

The **Hough Circle Transform** is used to detect circular structures in the processed image.

Because many sports balls have approximately circular shapes, circle detection provides a useful classical approach for this experiment.

---

## 📊 Results

The experiments demonstrate that combining multiple classical image processing techniques can improve the quality of the input image and help identify circular objects.

The project particularly explores the effects of:

* Noise reduction
* Contrast enhancement
* Color analysis
* Hough-based circle detection

The original experiments showed that filtering improved image quality and that the Hough Circle Transform could successfully identify circular structures in suitable images.

---

## ⚠️ Limitations

This approach has several limitations.

Performance can be affected by:

* Complex backgrounds
* Occluded objects
* Poor lighting
* Significant image noise
* Non-circular object appearance
* Similar colors between the object and background
* Incorrect Hough Transform parameters

The method is therefore primarily intended as an educational exploration of classical computer vision techniques.

For robust real-world object detection, modern machine learning and deep learning approaches such as **YOLO-based object detectors** would generally be more suitable.

---

## 🎓 Learning Outcomes

This project provides practical experience with:

* Image preprocessing
* Noise reduction
* Image restoration
* Histogram equalization
* Color analysis
* Feature extraction
* Hough Transform
* Classical computer vision
* OpenCV programming
* Designing a modular image-processing pipeline

---

## 🚀 Possible Future Improvements

Potential improvements include:

* Add quantitative evaluation metrics
* Compare different filtering methods
* Automatically tune Hough Circle parameters
* Add support for multiple input images
* Improve visualization of intermediate processing stages
* Add automated tests
* Add command-line arguments
* Compare classical detection with a deep learning detector
* Add a reproducible experiment configuration

---

## 👨‍💻 Author

**Ali Valizadeh**

Python Developer · Django · AI, NLP & Automation · University Instructor

GitHub:

https://github.com/AliValizade

---

## 👨‍🏫 Academic Context

This project was developed as an exploration of image processing and classical computer vision techniques.

**Supervisor:** Dr. Boshra Rajaee

---

## 📄 License

No explicit open-source license has been specified for this repository.
