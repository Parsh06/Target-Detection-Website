# sihOfcourse

## Human Action Detection

This Python application leverages the Flask framework to create a web-based interface for real-time human action detection. The application uses the Mediapipe library for pose estimation and OpenCV for video capturing and processing.

### Requirements

- Python 3.x
- Flask
- OpenCV
- Mediapipe
- numpy

You can install the required libraries using the following command:

```bash
pip install Flask 
pip install opencv-python 
pip install mediapipe
pip install numpy
```
## Implementation without API

## Code Structure For Individual Test
1. For Jump testing  use `jumping.py`
```bash
python jumping .py
```
2. For crawling testing use `crawling.py`
```bash
python crawling.py
```

3. For Comlete Testing without timer  use `trial.py`
```bash
python trial.py
```

4. For Complete Testng With Timer use `main.py`
```bash
python main.py 
```


## API Implementation 
Under Development 
### Usage For API

1. Run the application by executing the `app.py` file:

```bash
python app.py
```

2. Use Go Live Server Extension on `index.html` to use the frontend on web browser

3. The application will use your computer's camera to capture video. Detected actions (e.g., jumping, running, crawling) will be displayed on the video feed in real-time.

### Code Structure For API

- `app.py`: This is the main Python script that contains the Flask application and handles the routes.
- `templates/index.html`: The HTML file for the web interface.

### API Endpoint

The application also provides an API endpoint at `/api` that you can use to integrate this action detection functionality into other applications. Simply make a GET request to this endpoint, and it will return a JSON response with a message.

### Acknowledgements

- This application utilizes the [Flask](https://flask.palletsprojects.com/en/2.1.x/) web framework for creating the web interface.
- It also leverages the [Mediapipe](https://mediapipe.dev/) library for pose estimation and hand tracking.
- Video capturing and processing is done using the [OpenCV](https://opencv.org/) library.


