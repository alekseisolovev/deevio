# Deevio Data Scientist Hiring Challenge

## Documentation

-   Clone this repository
-   Open model.ipynb to see data preprocessing and model training pipeline  

### Set up
```bash
docker pull tensorflow/serving
docker run -t --rm -p 8501:8501 -v "deevio/tmp/efficientnet:/models/efficientnet" -e MODEL_NAME=efficientnet tensorflow/serving
```
### Use
```bash
import json, requests
import tensorflow as tf

def load_image(x):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.cast(x, tf.float32) / 255.0    
    return tf.expand_dims(x, axis=0)
    
image = 'data/1522073513_good.jpeg'
tensor = load_image(image)

input_data_json = json.dumps(
    {
        "signature_name": "serving_default", 
        "instances": tensor.numpy().tolist(),
    }
)

SERVER_URL = 'http://localhost:8501/v1/models/efficientnet:predict'

response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status() # raise an exception in case of error
response = response.json()

y_pred = response["predictions"]
```
