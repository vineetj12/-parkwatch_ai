import os
import subprocess
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes to allow Vercel frontend to fetch data
CORS(app)

DATA_FILE = "parkwatch_data.json"
PIPELINE_FILE = "pipeline.py"
DATASET_FILE = "dataset.csv"

@app.route("/api/data", methods=["GET"])
def get_data():
    # If the processed data file doesn't exist, try to run the pipeline to generate it
    if not os.path.exists(DATA_FILE):
        if os.path.exists(PIPELINE_FILE) and os.path.exists(DATASET_FILE):
            print("parkwatch_data.json not found. Running pipeline to generate it...")
            try:
                # Run the pipeline script
                result = subprocess.run(["python", PIPELINE_FILE], capture_output=True, text=True, check=True)
                print("Pipeline execution stdout:\n", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Pipeline execution failed:\n", e.stderr)
                return jsonify({"error": f"Failed to run pipeline: {e.stderr}"}), 500
            except Exception as e:
                print("Pipeline execution error:", str(e))
                return jsonify({"error": f"Failed to run pipeline: {str(e)}"}), 500
        else:
            return jsonify({
                "error": "Processed data file not found, and cannot run pipeline because pipeline.py or dataset.csv is missing."
            }), 404

    # Double check if file exists now and serve it
    if os.path.exists(DATA_FILE):
        return send_from_directory(os.getcwd(), DATA_FILE)
    else:
        return jsonify({"error": "Data file not found"}), 404

@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline():
    if os.path.exists(PIPELINE_FILE):
        try:
            print("Triggered manual pipeline regeneration...")
            result = subprocess.run(["python", PIPELINE_FILE], capture_output=True, text=True, check=True)
            return jsonify({
                "status": "success", 
                "message": "Pipeline completed successfully",
                "output": result.stdout
            })
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Pipeline execution failed: {e.stderr}"}), 500
        except Exception as e:
            return jsonify({"error": f"Pipeline execution failed: {str(e)}"}), 500
    return jsonify({"error": "pipeline.py not found"}), 404

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
