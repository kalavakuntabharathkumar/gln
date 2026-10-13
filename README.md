# GradeLens — Transformer-Based Answer Evaluation and Learning Analytics

A starter project combining DistilBERT fine-tuning (PyTorch + Hugging Face), evaluation metrics, MLflow experiment logging, SQLite storage, and a Streamlit/Plotly dashboard.

## Dashboard quick start
```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/create_demo_data.py
streamlit run dashboard.py
```
Open http://localhost:8501.

## Fine-tuning
Prepare `data/answers.csv` with `answer_text` and integer `label` columns. Then run:
```bash
python train.py --data data/answers.csv --output models/gradelens-distilbert --epochs 2
```
First run downloads pretrained DistilBERT weights. `evaluate.py` evaluates a saved checkpoint. Synthetic dashboard data is for demonstration only, not a reproduction of claimed resume metrics. Use teacher review for all high-stakes grading.
