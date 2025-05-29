import cv2
import matplotlib.pyplot as plt

def load_models():
    # Model files (make sure these files are in your working directory)
    face_model_config = "opencv_face_detector.pbtxt"
    face_model_weights = "opencv_face_detector_uint8.pb"
    age_model_config = "age_deploy.prototxt"
    age_model_weights = "age_net.caffemodel"
    gender_model_config = "gender_deploy.prototxt"
    gender_model_weights = "gender_net.caffemodel"

    # Load DNN models from files
    face_net = cv2.dnn.readNet(face_model_weights, face_model_config)
    age_net = cv2.dnn.readNet(age_model_weights, age_model_config)
    gender_net = cv2.dnn.readNet(gender_model_weights, gender_model_config)

    return face_net, age_net, gender_net

def detect_faces(face_net, frame):
    frame_h, frame_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=True, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # confidence threshold
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)

            # Clip coordinates to image size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w - 1, x2)
            y2 = min(frame_h - 1, y2)

            faces.append((x1, y1, x2, y2))
    return faces

def predict_age_gender(face_img, age_net, gender_net):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']

    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    return gender, age

def draw_predictions(frame, faces, age_net, gender_net):
    for (x1, y1, x2, y2) in faces:
        face_img = frame[y1:y2, x1:x2]

        # Skip if face ROI is empty or invalid
        if face_img.size == 0:
            continue

        gender, age = predict_age_gender(face_img, age_net, gender_net)
        label = f"{gender}, {age}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    return frame

def main():
    image_path = 'C:\\Users\\kdine\\OneDrive\\Documents\\Pictures\\balareddy.jpg'  # change this to your image path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    image = cv2.resize(image, (720, 640))

    face_net, age_net, gender_net = load_models()

    faces = detect_faces(face_net, image)
    result_img = draw_predictions(image.copy(), faces, age_net, gender_net)

    # Convert BGR to RGB for matplotlib display
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(result_rgb)
    plt.axis('off')
    plt.title("Age and Gender Detection")
    plt.show()

if __name__ == "__main__":
    main()
