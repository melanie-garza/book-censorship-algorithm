# 📖 The Censorship Algorithm

An interactive Machine Learning dashboard that opens the "black box" of U.S. book censorship. Trained on **16,439 books**, this project compares two interpretable models each matched to the data type it handles best—to expose both real world censorship patterns and algorithmic blind spots.

## 🧠 Models & Performance

| Model | Input Feature | Approach | Test Accuracy |
| :--- | :--- | :--- | :---: |
| **Logistic Regression** | Book Synopsis (`description`) | TF-IDF (unigrams & bigrams, 3,000 features) | **77.2%** |
| **Decision Tree** | Genre Metadata (`genre`) | Binary Genre Flags (`max_depth=3`) | **73.9%** |

## ✨ Key Features
* **Live Synopsis Editor & Word-by-Word X-Ray:** Edit any book's synopsis in real time to see how individual words push the linear score ($z$) and shift the prediction along the Sigmoid curve.
* **Global Feature Weights:** Visualizes the top words driving predictions toward *Banned* vs. *Not Banned* across 13,000+ training synopses.
* **Step-by-Step Decision Tree Path:** Displays the exact 3-level If/Else rules triggered by a book's genre tags alongside the full Graphviz tree diagram.

## 🚀 Run Locally

1. Clone the repository:
bash
git clone [https://github.com/melanie-garza/book-censorship-algorithm.git](https://github.com/melanie-garza/book-censorship-algorithm.git)
cd book-censorship-algorithm

2. Create a virtual environment and install dependencies:
bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3. Launch the Streamlit app:
bash
streamlit run app.py

## 🛠️ Tech Stack
Python · Streamlit · scikit-learn · Plotly · Pandas · NumPy · Graphviz
