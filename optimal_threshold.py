import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go  

def plot_accuracy_vs_threshold(results):
 
    avg_accuracy = np.mean(results['accuracy'])
    print(f"Average accuracy: {avg_accuracy}")
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=results['threshold'], y=results['accuracy'], mode='lines+markers', name='Accuracy'))

    fig.add_trace(go.Scatter(x=results['threshold'], y=[avg_accuracy] * len(results['threshold']),
                             mode='lines', name='Average Accuracy', line=dict(dash='dash', color='red')))

    fig.update_layout(title='Threshold vs Accuracy',
                      xaxis_title='Threshold',
                      yaxis_title='Accuracy')

    if not os.path.exists('Image'):
        os.makedirs('Image')
    fig.write_image("Image/optimal_threshold.png")
    
    print("Line graph saved as Image/optimal_threshold.png")


def resize_images_to_same_dimension(*images):
    min_height = min(image.shape[0] for image in images)
    min_width = min(image.shape[1] for image in images)
    
    resized_images = [cv2.resize(image, (min_width, min_height)) for image in images]
    return resized_images

def convert_to_black_and_white(image, threshold_value=0, use_otsu=False):
    if len(image.shape) == 3: 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if use_otsu:
        _, bw_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    return bw_image

def convert_to_three_channels(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def calculate_pixel_accuracy(pred, target):
    if pred.shape != target.shape:
        target = cv2.resize(target, (pred.shape[1], pred.shape[0])) 
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    accuracy = np.mean(pred_flat == target_flat)
    return accuracy

def segment_with_threshold(image, threshold_value):
    return convert_to_black_and_white(image, threshold_value)

def compare_with_mask(segmented_image, mask, img, accuracies):
    ground_truth_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert mask to grayscale
    segmented_image_resized = cv2.resize(segmented_image, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    image_resized, mask_resized, segmented_image_resized = resize_images_to_same_dimension(img, mask, segmented_image_resized)

    segmented_image_bw = convert_to_black_and_white(segmented_image_resized)
    mask_resized_bw = convert_to_black_and_white(mask_resized)

    mask_resized_bw_3d = convert_to_three_channels(mask_resized_bw)
    segmented_image_bw_3d = convert_to_three_channels(segmented_image_bw)

    accuracy = calculate_pixel_accuracy(segmented_image_bw, mask_resized_bw)
    accuracies.append(accuracy)
    

# Main processing loop
def main():
    image_dir = 'Data/Image'  
    mask_dir = 'Data/Mask'  

    images = load_images(image_dir) 
    masks = load_images(mask_dir) 

    images = images[:25]
    masks = masks[:25]

    results = {'threshold': [], 'accuracy': []}  

    best_threshold = 0
    best_accuracy = 0

    for threshold in range(0, 256,20):
        print(f"Processing for Threshold: {threshold}")
        
        temp={'accuracy':[]}
        for img, mask in zip(images, masks):
            segmented_image_bw = segment_with_threshold(img, threshold)
           
            _, mask_bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            compare_with_mask(segmented_image_bw, mask, img, temp['accuracy'])

        results['threshold'].append(threshold)
        results['accuracy'].append(np.mean(temp['accuracy']))

    best_threshold, best_accuracy = get_best_threshold(results)

    print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy}")

    plot_accuracy_vs_threshold(results)

def get_best_threshold(results):
    if 'threshold' in results and 'accuracy' in results:
        best_accuracy_idx = np.argmax(results['accuracy'])
        
        best_threshold = results['threshold'][best_accuracy_idx]
        best_accuracy = results['accuracy'][best_accuracy_idx]
        
        return best_threshold, best_accuracy
    else:
        print("Invalid results data")
        return None, None

def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is not None:
            images.append(img)
    return images

if __name__ == "__main__":
    main()
