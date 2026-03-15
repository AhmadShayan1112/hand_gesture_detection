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

### Example Frontend Code (JavaScript/React)

Here is a simple `fetch` request showing how to call the endpoint using a standard Image File or Blob:

```javascript
// Function to upload a hand gesture image to the API
async function predictGesture(imageBlob, type = 'english_alphabet') {
    // Determine the endpoint URL based on type
    const endpoint = type === 'digit' 
        ? 'http://localhost:8080/digit' 
        : 'http://localhost:8080/english_alphabet';

    // Prepare form data
    const formData = new FormData();
    // Assuming 'imageBlob' is a Blob, File, or an image captured from a canvas
    formData.append('image', imageBlob, 'capture.jpg'); 

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
            // Note: Don't set 'Content-Type' manually when using FormData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Predicted Class:", result.prediction_class);
        console.log("Confidence:", result.confidence);
        
        return result; // Example: { prediction_class: "A", confidence: 0.98 }
    } catch (error) {
        console.error("Error predicting gesture:", error);
    }
}
```

### Capturing Frames from a Webcam (Frontend)
If you're using a webcam in the browser, a common approach is:
1.  Draw the ` <video> ` frame to a ` <canvas> ` element.
2.  Convert the canvas to a Blob using `canvas.toBlob(...)`.
3.  Pass that Blob into the `predictGesture` function shown above.
