import os
import cv2
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_images(image_dir, num_images=5):
    images = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if i >= num_images:
            break
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is not None:
            images.append(img)
    return images

def segment_image_with_kmeans(image, k=3, reduction=None, n_components=2):
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3)).astype(np.float32)

    if reduction == 'PCA':
        reducer = PCA(n_components=n_components)
        pixel_values_reduced = reducer.fit_transform(pixel_values)
        pixel_values_inverse = reducer.inverse_transform
    elif reduction == 'ICA':
        reducer = FastICA(n_components=n_components, random_state=42)
        pixel_values_reduced = reducer.fit_transform(pixel_values)
        pixel_values_inverse = reducer.inverse_transform
    else:
        pixel_values_reduced = pixel_values

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixel_values_reduced)
    centers = kmeans.cluster_centers_
    if reduction:
        centers = pixel_values_inverse(centers)

    segmented_image = np.uint8(centers[labels.flatten()]).reshape(image_rgb.shape)
    return segmented_image

def segment_image_with_meanshift(image, reduction=None, n_components=2):
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3)).astype(np.float32)

    if reduction == 'PCA':
        reducer = PCA(n_components=n_components)
        pixel_values_reduced = reducer.fit_transform(pixel_values)
        pixel_values_inverse = reducer.inverse_transform
    elif reduction == 'ICA':
        reducer = FastICA(n_components=n_components, random_state=42)
        pixel_values_reduced = reducer.fit_transform(pixel_values)
        pixel_values_inverse = reducer.inverse_transform
    else:
        pixel_values_reduced = pixel_values

    bandwidth = estimate_bandwidth(pixel_values_reduced, quantile=0.2, n_samples=500)
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = meanshift.fit_predict(pixel_values_reduced)
    centers = meanshift.cluster_centers_
    if reduction:
        centers = pixel_values_inverse(centers)

    segmented_image = np.uint8(centers[labels.flatten()]).reshape(image_rgb.shape)
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
    # Separate methods based on clustering technique
    kmeans_methods = {method: results[method]['accuracy'] for method in results if 'KMeans' in method}
    meanshift_methods = {method: results[method]['accuracy'] for method in results if 'MeanShift' in method}
    
    # Define colors for each type of reduction
    color_map = {
        'None': 'blue',
        'PCA': 'green',
        'ICA': 'red'
    }

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("KMeans Accuracy Comparison", "MeanShift Accuracy Comparison")
    )
    
    # Helper function to determine color based on reduction type
    def get_color(method):
        if 'PCA' in method:
            return color_map['PCA']
        elif 'ICA' in method:
            return color_map['ICA']
        else:
            return color_map['None']
    
    # Plot KMeans methods in the first subplot
    for method, accuracies in kmeans_methods.items():
        mean_accuracy = np.mean(accuracies)
        fig.add_trace(
            go.Bar(name=method, x=['Accuracy'], y=[mean_accuracy], marker_color=get_color(method)),
            row=1, col=1
        )
    
    # Plot MeanShift methods in the second subplot
    for method, accuracies in meanshift_methods.items():
        mean_accuracy = np.mean(accuracies)
        fig.add_trace(
            go.Bar(name=method, x=['Accuracy'], y=[mean_accuracy], marker_color=get_color(method)),
            row=1, col=2
        )

    # Update layout and add legend
    fig.update_layout(
        title="Segmentation Accuracy Comparison for KMeans and MeanShift Methods",
        xaxis_title="Metrics",
        yaxis_title="Accuracy Score",
        height=500,
        width=1000,
        legend=dict(
            title="Dimensionality Reduction",
            itemsizing='constant',
            orientation="h",
            x=0.5,
            y=-0.15,
            xanchor='center'
        )
    )
    
    # Add legend entries for dimensionality reduction types
    for reduction, color in color_map.items():
        fig.add_trace(
            go.Bar(name=reduction, x=[None], y=[None], marker_color=color, showlegend=True)
        )

    # Save the plot as a single image
    if not os.path.exists('Image'):
        os.makedirs('Image')
    fig.write_image("Image/metrics_histogram_comparison_accuracy.png")
    
    print("Histogram saved as Image/metrics_histogram_comparison.png")


def main():
    k = 3
    methods = {
        "KMeans (No Reduction)": lambda img: segment_image_with_kmeans(img, k=k),
        "KMeans + PCA": lambda img: segment_image_with_kmeans(img, k=k, reduction='PCA'),
        "KMeans + ICA": lambda img: segment_image_with_kmeans(img, k=k, reduction='ICA'),
        "MeanShift (No Reduction)": lambda img: segment_image_with_meanshift(img),
        "MeanShift + PCA": lambda img: segment_image_with_meanshift(img, reduction='PCA'),
        "MeanShift + ICA": lambda img: segment_image_with_meanshift(img, reduction='ICA'),
    }
    
    results = {method: {'accuracy': []} for method in methods}

    image_dir = 'Data/Image'
    mask_dir = 'Data/Mask'
    
    images = load_images(image_dir)
    masks = load_images(mask_dir)

    for method, segment_fn in methods.items():
        for img, mask in zip(images, masks):
            segmented_image = segment_fn(img)
            if segmented_image is not None:
                compare_with_mask(segmented_image, mask, results[method]['accuracy'])
        
        print(f"Finished processing {method}")

    plot_results(results)

if __name__ == "__main__":
    main()
