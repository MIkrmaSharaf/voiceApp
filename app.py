from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
import tempfile
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize SpeechBrain model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    try:
        # Check if the request has a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        # Ensure the file has been uploaded
        if file.filename == '':
            return jsonify({'error': 'No file selected for upload'}), 400

        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(file.read())
            temp_audio_path = temp_audio_file.name

        # Load the audio file using its path
        signal = model.load_audio(temp_audio_path)

        # Normalize the signal and generate the embedding
        normalized_signal = model.audio_normalizer(signal, sample_rate=16000)
        embedding = model.encode_batch(normalized_signal).detach().numpy().flatten().tolist()

        # Clean up the temporary file
        os.remove(temp_audio_path)

        # Return the embedding
        return jsonify({'embedding': embedding})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
