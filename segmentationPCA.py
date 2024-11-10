import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
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

def segment_image_with_pca(image, n_components=2, k=3):
    if image is None:
        print("Error: Invalid image")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    pixel_values_reduced = pca.fit_transform(pixel_values)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixel_values_reduced)
    centers = kmeans.cluster_centers_
    
    # Map labels back to the original color space
    centers = pca.inverse_transform(centers)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)
    return segmented_image

def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    return np.sum(intersection) / np.sum(union)

def calculate_dice(pred, target):
    intersection = np.logical_and(pred, target)
    return 2. * np.sum(intersection) / (np.sum(pred) + np.sum(target))

def calculate_pixel_accuracy(pred, target):
    return np.mean(pred == target)

def compare_with_mask(segmented_image, mask, iou_scores, dice_scores, accuracies):
    ground_truth_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    segmented_image_resized = cv2.resize(segmented_image, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    segmented_gray = cv2.cvtColor(segmented_image_resized, cv2.COLOR_BGR2GRAY)

    _, ground_truth_binary = cv2.threshold(ground_truth_mask, 127, 1, cv2.THRESH_BINARY)
    _, segmented_binary = cv2.threshold(segmented_gray, 127, 1, cv2.THRESH_BINARY)

    iou = calculate_iou(segmented_binary, ground_truth_binary)
    dice = calculate_dice(segmented_binary, ground_truth_binary)
    accuracy = calculate_pixel_accuracy(segmented_binary, ground_truth_binary)
    
    iou_scores.append(iou)
    dice_scores.append(dice)
    accuracies.append(accuracy)

def plot_results(results):
    # Calculate mean values for each metric
    mean_accuracy = np.mean(results['accuracy'])
    mean_iou = np.mean(results['iou'])
    mean_dice = np.mean(results['dice'])

    # Create histogram
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=['Metrics'], y=[mean_accuracy], marker_color='blue'),
        go.Bar(name='IoU Score', x=['Metrics'], y=[mean_iou], marker_color='green'),
        go.Bar(name='Dice Score', x=['Metrics'], y=[mean_dice], marker_color='red')
    ])

    fig.update_layout(
        title="Segmentation Metrics for K=3 with PCA",
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode='group',
        height=400,
        width=600
    )
    
    # Save the plot
    if not os.path.exists('Image'):
        os.makedirs('Image')
    fig.write_image("Image/metrics_histogram_with_pca_k3.png")
    
    print("Histogram saved as Image/metrics_histogram_with_pca_k3.png")

def main():
    k = 3  # Fix k=3 for clustering
    results = {'accuracy': [], 'iou': [], 'dice': []}

    image_dir = 'Data/Image'
    mask_dir = 'Data/Mask'
    
    images = load_images(image_dir)
    masks = load_images(mask_dir)
    for img, mask in zip(images, masks):
        segmented_image = segment_image_with_pca(img, k=k)
        if segmented_image is not None:
            compare_with_mask(segmented_image, mask, results['iou'], results['dice'], results['accuracy'])
        
    plot_results(results)

if __name__ == "__main__":
    main()
