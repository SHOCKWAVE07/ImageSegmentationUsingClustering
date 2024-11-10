import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load_images(image_dir, num_images=5):
    images = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if i >= num_images:
            break
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is not None:
            images.append(img)
    return images

def segment_image(image, k):
    if image is None:
        print("Error: Invalid image")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)
    return segmented_image

def calculate_pixel_accuracy(pred, target):
    return np.mean(pred == target)

def compare_with_mask(segmented_image, mask, accuracies):
    ground_truth_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    segmented_image_resized = cv2.resize(segmented_image, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    segmented_gray = cv2.cvtColor(segmented_image_resized, cv2.COLOR_BGR2GRAY)

    _, ground_truth_binary = cv2.threshold(ground_truth_mask, 127, 1, cv2.THRESH_BINARY)
    _, segmented_binary = cv2.threshold(segmented_gray, 127, 1, cv2.THRESH_BINARY)

    accuracy = calculate_pixel_accuracy(segmented_binary, ground_truth_binary)
   
    accuracies.append(accuracy)

def plot_results(results):
    # Create a single plot for accuracy scores of all methods
    fig = go.Figure()

    # Add accuracy scores as a line plot with markers
    fig.add_trace(go.Scatter(
        x=list(results.keys()), 
        y=[np.mean(v['accuracy']) for v in results.values()], 
        mode='lines+markers', 
        name='Accuracy',
        marker=dict(size=8, color='blue')
    ))

    # Update layout
    fig.update_layout(
        title="Segmentation Accuracy for Different Methods",
        xaxis_title="Methods",
        yaxis_title="Accuracy Score",
        height=400,
        width=1000
    )
    
    # Save the plot as an image
    if not os.path.exists('Image'):
        os.makedirs('Image')
    fig.write_image("Image/accuracy_plot.png")
    
    print("Plot saved as Image/accuracy_plot.png")

def main():
    k_values = range(2, 11)
    results = {k: {'accuracy': []} for k in k_values}

    image_dir = 'Data/Image'
    mask_dir = 'Data/Mask'
    
    images = load_images(image_dir)
    masks = load_images(mask_dir)
    for k in k_values:
        for img, mask in zip(images, masks):
            segmented_image = segment_image(img, k)
            if segmented_image is not None:
                compare_with_mask(segmented_image, mask, results[k]['accuracy'])
        
    plot_results(results)

if __name__ == "__main__":
    main()
