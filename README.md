# Image Segmentation Using Clustering

Flooded zone image segmentation plays a key role in rapid disaster management function, thereby allowing efficient identification of the flooded zones. This is highly critical for timely response, resource allocation, and infrastructure planning to mitigate the dangers of loss of life, destruction of property, and disruption of essential services. There is a major scope for the use of advanced algorithms to minimize the impacts of flooding, ultimately saving lives and protecting communities from future disasters.


### Requied packages are mentioned in requirements.txt file


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

### 5. Run Analysis Scripts
```bash
# Find optimal K value for K-means clustering
python optimal_k.py

# Determine optimal threshold for converting segmented images to black and white
python optimal_threshold.py

# Run segmentation
python Segmentation.py
```