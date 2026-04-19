# 🌍 Feature Extraction from Satellite Images

## 📌 Overview

This project focuses on extracting meaningful features from satellite images and using them for classification tasks. It is designed for land-use and land-cover analysis using machine learning techniques.

The workflow includes image preprocessing, feature extraction, and classification based on extracted features.

---

## 🚀 Features

* 📷 Satellite image preprocessing
* 🧠 Feature extraction from images
* 📊 Classification using extracted features
* 🗂 Organized dataset handling

---

## 🗂 Project Structure

```
Feature_extraction_from_satellite_images/
│── classification.py
│── feature_extraction.py
│── sort_eurosat_rgb.py
│── data/              # Dataset (not included)
│── README.md
│── .gitignore
```

---

## 📥 Dataset Setup

This repository does **not include the dataset** to keep it lightweight and avoid large file uploads.

https://drive.google.com/file/d/12UC7x4vi2Va4rFAUimUxKKMXKsEtF9gg/view?usp=drive_link
```

⚠️ Note

* The `data/` folder is ignored using `.gitignore`
* Large files such as `.npy` are also excluded
* Make sure the dataset is correctly placed before running the code

---

 ⚙️ Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

---

 ▶️ Usage

### 1. Run Feature Extraction

```
python feature_extraction.py
```

### 2. Run Classification

```
python classification.py
```

---

 📌 Notes

* Ensure the dataset is available in the correct directory
* The project is best run inside a virtual environment (`.venv`)
* Modify file paths if your dataset structure differs

---

🎯 Applications

* Land-use classification
* Environmental monitoring
* Remote sensing analysis
* Satellite image processing

---

🛠 Tech Stack

* Python
* NumPy
* Machine Learning libraries

---

