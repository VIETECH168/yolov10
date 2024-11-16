from ultralytics import YOLOv10
import cv2
import matplotlib.pyplot as plt

# Load YOLOv10-n model
model = YOLOv10.from_pretrained('jameslahm/yolov10n')

# Perform inference on an image
results = model(source='dog.jpeg', conf=0.25)

# Define custom class names
class_names = {0: 'human', 16: 'dog'}  # Mapping class IDs to class names

# Define colors for each class (BGR format)
colors = {
    0: (0, 255, 0),  # Green for 'human'
    16: (255, 0, 0)  # Blue for 'dog'
}

# Print the bounding box coordinates and classes
print("Bounding boxes (xyxy format) and classes:")
for box in results[0].boxes:
    print(f"Coordinates: {box.xyxy}, Class: {box.cls}")

# Visualize the results
# Convert the image from RGB to BGR for OpenCV
image = cv2.imread('dog.jpeg')

for box in results[0].boxes:
    # Get coordinates and class index
    xyxy = box.xyxy[0].cpu().numpy()  # Convert to numpy array
    xmin, ymin, xmax, ymax = map(int, xyxy)

    class_id = int(box.cls[0].cpu().numpy())  # Get class index as an integer

    # Draw the bounding box with the specified color
    color = colors.get(class_id, (255, 255, 255))  # Default to white if class_id is unknown
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    # Convert confidence tensor to a standard float
    confidence = box.conf[0].item()  # Convert tensor to float

    # Prepare the label with the class name using the custom mapping
    label = f"{class_names[class_id]}: {confidence:.2f}"  # Class name and confidence

    # Draw the label on the image
    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with bounding boxes and class labels
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axis
plt.show()
