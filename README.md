# Emotion Detection from Webcam

This project is a real-time emotion detection application that uses a webcam to capture video frames and predicts the emotion of the detected face using a pre-trained deep learning model. The detected emotions are displayed on the web interface along with the live video feed.

## Requirements

- Python > 3.8
- OpenCV
- Flask
- Keras
- Numpy
- Pandas

## Getting Started

1. Clone the repository:

```
git clone https://github.com/ticuss/emotion-detection.git
```

2. Run the application:
   Be sure to be in the app folder in the terminal

```
python camera_flask_app.py
```

3. Open your web browser and go to `http://localhost:5000` to access the application.

## Possible problems

- If the app is not running close completely google chrome or what browser do you use and re-run the application

## Usage

- The web interface displays the live video feed from your webcam along with the predicted emotions for the detected faces.
- Click the "Capture" button to take a snapshot of the current frame and save it as an image in the "shots" directory.
- Click the "Negative" button to toggle the negative effect on the video feed.
- Click the "Face Only" button to display only the detected faces in the video feed.
- Click the "Stop/Start" button to stop or start the webcam feed.
- Click the "Start/Stop Recording" button to start or stop recording a video. The recorded videos are saved in the project directory.

## Project Structure

- `app.py`: The main Flask application that handles the web interface and video streaming.
- `templates/`: Directory containing HTML templates for the web interface.
- `saved_model/`: Directory containing the pre-trained model and required model files.
- `shots/`: Directory where captured snapshots and recorded videos are saved.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

- The pre-trained model used in this project is based on the [FER2013 dataset](https://www.kaggle.com/deadskull7/fer2013) available on Kaggle.

## Disclaimer

This project is for educational and demonstration purposes only. The accuracy of emotion detection may vary, and it should not be used for any critical applications without proper evaluation and testing.
