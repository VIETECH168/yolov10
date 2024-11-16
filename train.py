from ultralytics import YOLOv10

def main():
    # Load a pre-trained YOLOv10 model
    model = YOLOv10.from_pretrained('jameslahm/yolov10n')

    # Specify the path to the data.yaml file
    data_path = r'C:\Users\phand\OneDrive - Nanyang Technological University\Online_Class\Code\yolov10\datasets\football\data.yaml'

    # Start training the model
    try:
        results = model.train(data=data_path, epochs=10, imgsz=640)

        # Print training results
        print("Training results:", results)
    except Exception as e:
        print("An error occurred during training:", str(e))

if __name__ == '__main__':
    main()
