# Hand Gesture Detection API and Inference

This project provides a FastAPI endpoint for hand gesture classification (Digits and English alphabet) and real-time inference scripts for dataset creation and testing.

## Prerequisites

1.  **Python Version**: Ensure you are using **Python 3.9.6**.
2.  **Virtual Environment (Optional but Recommended)**:
    ```bash
    python3.9 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## 1. Running the FastAPI Endpoint

The `endpoint.py` serves a REST API to classify hand gestures from an image. 

To start the server, run:
```bash
python -m uvicorn endpoint:app --port 8080 --reload
```
The API will be available at `http://localhost:8080`.

## 2. Running Simple Inference (Real-time Webcam)

There are several inference scripts for real-time testing via your webcam (e.g., `inference_english.py`, `inference_digit.py`, `inference_urdu.py`). 

To run the English alphabet inference:
```bash
python inference_english.py
```
*   A window will pop up showing your webcam feed.
*   It will draw the hand landmarks and display the predicted gesture.
*   Press `ESC` to exit the window.

## 3. Creating and Expanding the Dataset

You can use the inference scripts to collect new data and expand your custom datasets. 

Here is how data collection works (using `inference_english.py` as an example):

1.  Run the inference script: `python inference_english.py`.
2.  **Enter Logging Mode**: Press the `k` key on your keyboard. The mode text on the screen will change to `MODE: Logging Key Point`.
3.  **Log Data**: Make a hand gesture in front of the camera. While making the gesture, press a number key `0-9` that corresponds to the class label you want to assign to that gesture.
4.  **Saved Data**: The script will instantly extract the normalized hand landmark coordinates and append them, along with the label number, to the corresponding CSV file (e.g., `model/New_english_dataset/keypoint.csv`).
5.  **Return to Normal**: To stop logging, press `n`.

*(Note: If you are logging Point History gestures, press `h` to enter "Logging Point History" mode).*

## 4. Frontend Integration

To use the API from a frontend application (like React or Next.js), you need to send a `POST` request containing the frame image via `multipart/form-data`.

### Available Endpoints
*   **Digits Prediction**: `POST /digit`
*   **English Alphabet Prediction**: `POST /english_alphabet`

### Example Frontend Code (Live Camera in Flutter & React Native)

When integrating with mobile apps (Flutter or React Native) for real-time inference, you need to capture frames from the live camera and send them to the API using a multipart `POST` request. 

#### 1. Flutter Example (using `camera` and `http` packages)

```dart
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

bool isProcessing = false;

// Call this inside your CameraController's startImageStream callback
Future<void> processCameraFrame(CameraImage image, String type) async {
  if (isProcessing) return; // Prevent overlapping requests
  isProcessing = true;

  try {
    // 1. Convert CameraImage (YUV420/BGRA8888) to JPEG bytes. 
    // Note: You will need a helper function to convert 'image' to JPEG bytes.
    // For example, using the 'image' package to encode to Jpg.
    List<int> jpegBytes = await convertCameraImageToJpeg(image);

    var uri = Uri.parse("http://<YOUR_LOCAL_IP>:8080/$type");
    var request = http.MultipartRequest('POST', uri);
    
    request.files.add(http.MultipartFile.fromBytes(
      'image', 
      jpegBytes,
      filename: 'frame.jpg'
    ));

    var response = await request.send();
    if (response.statusCode == 200) {
      var responseData = await response.stream.bytesToString();
      var result = json.decode(responseData);
      print("Predicted: \${result['prediction_class']} (Confidence: \${result['confidence']})");
    }
  } catch (e) {
    print("Error processing frame: $e");
  } finally {
    isProcessing = false;
  }
}
```

#### 2. React Native Example (using `react-native-vision-camera` and `fetch`)

```javascript
import { useFrameProcessor } from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';

let isProcessing = false;

const sendFrameToAPI = async (base64Image, type = 'english_alphabet') => {
  if (isProcessing) return;
  isProcessing = true;

  try {
    const endpoint = `http://<YOUR_LOCAL_IP>:8080/${type}`;
    const formData = new FormData();
    
    // Convert base64 back to a file object for multipart/form-data
    formData.append('image', {
      uri: `data:image/jpeg;base64,\${base64Image}`,
      type: 'image/jpeg',
      name: 'frame.jpg',
    });

    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    const result = await response.json();
    console.log(`Predicted: \${result.prediction_class} (\${result.confidence})`);
  } catch (error) {
    console.error("Frame processing error:", error);
  } finally {
    isProcessing = false;
  }
};

// Inside your component
const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  // Note: react-native-vision-camera requires a frame processor plugin 
  // to convert the frame to base64. 
  // Assuming 'frameToBase64' is your custom C++/Objective-C/Java plugin:
  const base64Image = frameToBase64(frame); 
  
  runOnJS(sendFrameToAPI)(base64Image);
}, []);
```
