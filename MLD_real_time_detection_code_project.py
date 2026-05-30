import cv2
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2  
)

state_dict = torch.load("/Users/hetvi/Downloads/best_vit_model (2).pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_eye_tree_eyeglasses.xml')


score = 0
CLOSED_LIMIT = 8
font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    open_eyes = 0
    closed_eyes = 0

    for (x, y, w, h) in eyes:
        eye_img = gray[y:y+h, x:x+w]
        pil_img = Image.fromarray(eye_img).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()

        if pred == 1:
            eye_state = "OPEN"
            open_eyes += 1
        else:
            eye_state = "CLOSED"
            closed_eyes += 1

     
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, eye_state, (x, y-10), font, 0.7, (0, 255, 255), 2)

 
    if closed_eyes > 0:
        score += closed_eyes
    else:
        score -= 1
        if score < 0:
            score = 0

 
    if score > CLOSED_LIMIT:
        cv2.putText(frame, "DROWSY !!!", (50, 100), font, 1.5, (0, 0, 255), 3)

    cv2.putText(frame, f"Score: {score}", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.imshow("Drowsiness Detection - ViT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
