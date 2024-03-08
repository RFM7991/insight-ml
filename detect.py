import cv2
import torch
from torchvision import transforms
import pupil_net

# Load the trained model
model = pupil_net.PupilNet()
model.load_state_dict(torch.load('pupil_net2.pth'))
model.eval()

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_tensor = preprocess(frame)
    frame_tensor = frame_tensor.unsqueeze(0)  # Add a batch dimension

    # Get the predicted pupil coordinates
    with torch.no_grad():
        output = model(frame_tensor)
        x, y = output[0].tolist()  # Assuming the model outputs (x, y) coordinates

    # Postprocess the coordinates (if necessary)
    # Convert back to original image scale, etc.

    # Draw a red bounding box on the frame
    # Adjust the box size and position as needed
    box_size = 20
    top_left = (int(x - box_size / 2), int(y - box_size / 2))
    bottom_right = (int(x + box_size / 2), int(y + box_size / 2))
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)  # Red bounding box

    # Display the frame
    cv2.imshow('Pupil Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
