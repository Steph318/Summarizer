# Text Summarizer

## Project Overview
This project implements an end-to-end NLP (Natural Language Processing) solution for text summarization. The primary objective is to condense lengthy texts (e.g., articles or documents) into shorter, more concise summaries while preserving the original meaning and key information.

### Project Phases
The project includes the following phases:
1. **Data Ingestion and Validation**: Loading and validating the dataset.
2. **Data Transformation and Preprocessing**: Preparing data for model training.
3. **Model Training**: Using the `google/pegasus-cnn_dailymail` model from Hugging Face for text summarization.
4. **Model Evaluation**: Evaluating the model's performance.
5. **Prediction Pipeline and FastAPI Endpoint Development**: Creating a user-friendly API for predictions.
6. **CI/CD Deployment**: Automating deployment using GitHub Actions.

---

## Dataset
The project utilizes the [SAMSum dataset](https://huggingface.co/datasets/samsum) for training the text summarization model. The dataset contains dialogues and their corresponding summaries, making it suitable for conversational summarization tasks.

---

## Installation and Setup

### Clone the Repository
To begin, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/text-summarizer.git
cd text-summarizer
```

### Install Dependencies
Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Model
To train the summarization model, run the following command:
```bash
python main.py
```

### Launch the FastAPI Endpoint
To serve the model using FastAPI, follow these steps:
1. Run the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
2. Open your browser or an API client (e.g., Postman) and navigate to:
   ```
   http://localhost:8000/docs
   ```
   This will display the API documentation with interactive endpoints.

---

## Project Structure
Below is an overview of the project directory:
```
text-summarizer/
│
├── src/
│   ├── textSummarizer/
│   │   ├── components/        # Core components (e.g., data ingestion, transformation)
│   │   ├── pipeline/          # ML pipeline scripts
│   │   ├── utils/             # Utility functions
│   │   └── entity/            # Configuration and schema definitions
│
├── artifacts/                 # Intermediate artifacts (e.g., processed datasets)
├── app.py                     # FastAPI application
├── main.py                    # Entry point for training
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Contributing
Contributions to this project are welcome! To contribute, follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git add .
   git commit -m "Add your message here"
   git push origin feature/your-feature-name
   ```
4. Open a pull request and describe the changes you made.

---

## License
This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgements
This project was developed using:
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- [FastAPI](https://fastapi.tiangolo.com/)

For questions or feedback, feel free to open an issue or contact the repository owner.
