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
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Accuracy', 'IoU Score', 'Dice Score'))

    # Plot accuracy
    fig.add_trace(go.Scatter(x=list(results.keys()), 
                             y=[np.mean(v['accuracy']) for v in results.values()], 
                             mode='lines+markers', 
                             name='Accuracy'), row=1, col=1)
    
    # Plot IoU
    fig.add_trace(go.Scatter(x=list(results.keys()), 
                             y=[np.mean(v['iou']) for v in results.values()], 
                             mode='lines+markers', 
                             name='IoU'), row=1, col=2)
    
    # Plot Dice Score
    fig.add_trace(go.Scatter(x=list(results.keys()), 
                             y=[np.mean(v['dice']) for v in results.values()], 
                             mode='lines+markers', 
                             name='Dice'), row=1, col=3)

    fig.update_layout(height=400, width=1200, title_text="Segmentation Metrics for Different K Values")
    fig.show()

def main():
    k_values = range(2, 11)
    results = {k: {'accuracy': [], 'iou': [], 'dice': []} for k in k_values}

    image_dir = 'Data/Image'
    mask_dir = 'Data/Mask'
    
    images = load_images(image_dir)
    masks = load_images(mask_dir)
    for k in k_values:
        for img, mask in zip(images, masks):
            print(img,mask)
            segmented_image = segment_image(img, k)
            if segmented_image is not None:
                compare_with_mask(segmented_image, mask, results[k]['iou'], results[k]['dice'], results[k]['accuracy'])
        
    plot_results(results)

if __name__ == "__main__":
    main()
