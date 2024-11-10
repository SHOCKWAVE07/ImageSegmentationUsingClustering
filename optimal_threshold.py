import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Import Plotly for interactive plotting

def plot_accuracy_vs_threshold(results):
 
    # Calculate average accuracy across all thresholds
    avg_accuracy = np.mean(results['accuracy'])
    print(f"Average accuracy: {avg_accuracy}")
    
    # Create the plotly figure
    fig = go.Figure()

    # Line plot for Threshold vs Accuracy
    fig.add_trace(go.Scatter(x=results['threshold'], y=results['accuracy'], mode='lines+markers', name='Accuracy'))

    # Add a horizontal line for Average Accuracy
    fig.add_trace(go.Scatter(x=results['threshold'], y=[avg_accuracy] * len(results['threshold']),
                             mode='lines', name='Average Accuracy', line=dict(dash='dash', color='red')))

    # Update layout with titles
    fig.update_layout(title='Threshold vs Accuracy',
                      xaxis_title='Threshold',
                      yaxis_title='Accuracy')

    # Save the plot as a single image
    if not os.path.exists('Image'):
        os.makedirs('Image')
    fig.write_image("Image/optimal_threshold.png")
    
    print("Line graph saved as Image/optimal_threshold.png")


# Function to resize the images to the same size
def resize_images_to_same_dimension(*images):
    # Get the minimum height and width
    min_height = min(image.shape[0] for image in images)
    min_width = min(image.shape[1] for image in images)
    
    resized_images = [cv2.resize(image, (min_width, min_height)) for image in images]
    return resized_images

# Convert the segmented image to black and white (binary)
def convert_to_black_and_white(image, threshold_value=0, use_otsu=False):
    # Check if the image is already grayscale (single channel)
    if len(image.shape) == 3:  # If the image has 3 channels (color image)
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # If already grayscale, use it as it is
        gray_image = image

    # Apply thresholding to make it binary (black and white)
    if use_otsu:
        _, bw_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    return bw_image


# Ensure all images are 3D (with 3 channels) for concatenation
def convert_to_three_channels(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Calculate pixel accuracy by comparing prediction and target
def calculate_pixel_accuracy(pred, target):
    # Ensure both images are of the same shape
    if pred.shape != target.shape:
        target = cv2.resize(target, (pred.shape[1], pred.shape[0]))  # Resize target mask to match segmented image
    
    # Flatten the arrays to 1D to avoid issues with elementwise comparison in 2D
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Calculate accuracy by comparing pixel values
    accuracy = np.mean(pred_flat == target_flat)
    return accuracy

# Segment using specified threshold value
def segment_with_threshold(image, threshold_value):
    return convert_to_black_and_white(image, threshold_value)

# Function to display the images and calculate accuracy
def compare_with_mask(segmented_image, mask, img, accuracies):
    # Convert images to grayscale
    ground_truth_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert mask to grayscale
    segmented_image_resized = cv2.resize(segmented_image, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Resize images to the same dimension
    image_resized, mask_resized, segmented_image_resized = resize_images_to_same_dimension(img, mask, segmented_image_resized)

    # Convert the segmented image to black and white
    segmented_image_bw = convert_to_black_and_white(segmented_image_resized)
    mask_resized_bw = convert_to_black_and_white(mask_resized)

    # Convert black and white images back to 3 channels
    mask_resized_bw_3d = convert_to_three_channels(mask_resized_bw)
    segmented_image_bw_3d = convert_to_three_channels(segmented_image_bw)

    # Calculate accuracy (binary comparison)
    accuracy = calculate_pixel_accuracy(segmented_image_bw, mask_resized_bw)
    accuracies.append(accuracy)
    

# Main processing loop
def main():
    image_dir = 'Data/Image'  # Directory for input images
    mask_dir = 'Data/Mask'  # Directory for ground truth masks

    images = load_images(image_dir)  # Load images
    masks = load_images(mask_dir)  # Load masks

    images = images[:25]
    masks = masks[:25]

    results = {'threshold': [], 'accuracy': []}  # Store accuracy for each threshold

    best_threshold = 0
    best_accuracy = 0

    # Loop through all thresholds from 0 to 255 with step 50
    for threshold in range(0, 256,20):
        print(f"Processing for Threshold: {threshold}")
        
        # Loop through each image and corresponding mask
        temp={'accuracy':[]}
        for img, mask in zip(images, masks):
            # Segment the image using the current threshold
            segmented_image_bw = segment_with_threshold(img, threshold)
            
            # Ensure the mask is binary (0 and 255)
            _, mask_bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            compare_with_mask(segmented_image_bw, mask, img, temp['accuracy'])

        results['threshold'].append(threshold)
        results['accuracy'].append(np.mean(temp['accuracy']))
            
    best_threshold, best_accuracy = get_best_threshold(results)

    print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy}")

    plot_accuracy_vs_threshold(results)

# Find the best threshold based on maximum accuracy
def get_best_threshold(results):
    # Ensure 'threshold' and 'accuracy' lists exist in results
    if 'threshold' in results and 'accuracy' in results:
        # Find the index of the highest accuracy
        best_accuracy_idx = np.argmax(results['accuracy'])
        
        # Get the corresponding threshold for the best accuracy
        best_threshold = results['threshold'][best_accuracy_idx]
        best_accuracy = results['accuracy'][best_accuracy_idx]
        
        return best_threshold, best_accuracy
    else:
        # If there is an issue with the data, return None
        print("Invalid results data")
        return None, None

# Example usage



# Load images function
def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is not None:
            images.append(img)
    return images

# Call the main function to run the segmentation and accuracy calculation
if __name__ == "__main__":
    main()
