# UNO Cards Game using Machine Learning

This project uses **Machine Learning** to recognize **UNO cards** based on images captured from a webcam. It uses a Flask web application for the interface, where users can capture card images and view predictions of the card's color and type.

## Project Structure

```bash
├── app.py                # Main Flask application
├── templates/
│   └── index.html        # Front-end for capturing images
│   └──images/            # Sample images to test in Postman
├── database/             # Training dataset of UNO card images
├── requirements.txt      # Python dependencies
├── encode_image.py       # Python file to convert image into base64 string
└── README.md             # Project README
```
## Key Components

**app.py:** Contains the Flask app, ML model training, and card prediction logic.
**/database:** Folder with labeled UNO card images used to train the model.
**index.html:** Web interface for capturing card images using a webcam.

## Installation
Clone the repository:
```bash
pip install -r requirements.txt
```

## Run the Flask app:

```bash
python app.py
```

The server will run on http://127.0.0.1:5000/.

## Flask Routes
**/ (GET):** Serves the index.html page for capturing images.
**/predict (POST):** Receives the image (base64 format), processes it, and returns the predicted card color and type.

## Usage
Open the web interface at http://127.0.0.1:5000/.
Capture an UNO card image.
The predicted card's color and type will be displayed.


## Testing in Postman
Start the Flask app.
In Postman, send a POST request to http://127.0.0.1:5000/predict with JSON:
Convert Image into base64 string using encode_image.py
```json
{
    "image_data": "data:image/jpeg;base64,<base64_string>"
}
```
Receive the predicted card color and type in the response.