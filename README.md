# Image Segmentation Using Clustering

## Project Setup and Workflow

### 1. Clone the Repository
```bash
git clone https://github.com/SHOCKWAVE07/ImageSegmentationUsingClustering
cd ImageSegmentationUsingClustering
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Dataset
Download the dataset from [https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation]

### 6. Run Analysis Scripts
```bash
# Find optimal K value for K-means clustering
python optimal_k.py

# Determine optimal threshold for converting segmented images to black and white
python optimal_threshold.py

# Run segmentation
python Segmentation.py
```