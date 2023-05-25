from ultralytics import YOLO
import cv2

def load_model(model_path):
    model = YOLO(model_path)
    return model


    
def transform_output(results,img,model):
    #img = cv2.imread(img_path)
    for r in results[0].boxes:
        # rectangle
        x_min, y_min, x_max, y_max = map(int, r.xyxy[0])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # class name & confidence
        class_name = model.names[int(r.cls[0])]
        class_conf = round(float(r.conf[0]),2)
        text = f"{class_name} {class_conf}"

        # Determine text size
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        # Set text background rectangle coordinates
        text_bg_coords = ((x_min, y_min - text_size[1] - 10), (x_min + text_size[0], y_min - 10))
        # Draw text background rectangle with red color
        cv2.rectangle(img, text_bg_coords[0], text_bg_coords[1], (0, 0, 255), cv2.FILLED)
        # Put white text on the red background
        cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img

def predict(img_path,model):
    results = model.predict(img_path, save=False, conf=0.5)
    return transform_output(results,img_path,model)

if __name__ == '__main__':
    model_path = 'weights/best.pt'
    img_path = 'images/Hyundai-Grand-i10-Nios-200120231541.jpg'


    model = YOLO(model_path)

    results = model.predict(img_path, save=False, conf=0.5)

    cv2.imshow('ImageWindow',transform_output(results,img_path,model))
    cv2.waitKey(0)
    cv2.destroyAllWindows
        