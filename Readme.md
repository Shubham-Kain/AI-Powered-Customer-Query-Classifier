# AI-Powered Customer Query Classifier

## Project Overview
This project builds a customer query classification system that predicts three labels for incoming support requests:
- **Category**
- **Intent**
- **Priority**
- **Reply**

The solution is based on text preprocessing, TF-IDF vectorization, and supervised classification models.
The Streamlit app also uses LangChain with an OpenRouter model to generate a reply based on the user query.
It includes a training notebook (`train.ipynb`) for model development and a Streamlit app (`app.py`) for live prediction.

## Dataset
- File: `customer_query_dataset_humanized_12924.csv`
- Approximate rows: **12,924**
- Key columns:
  - `Customer_query`: customer support query text
  - `Category`: business or ticket category label
  - `Intent`: user intent label
  - `Priority`: urgency label

The dataset is mainly related to the **Customer Support / Customer Service domain**, specifically for a **software/SaaS business**.

This dataset is used to fit TF-IDF vectorizers, train classification models, and evaluate performance on held-out test data.

## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- NLTK
- Streamlit
- LangChain / OpenAI / OpenRouter
- pickle
- python-dotenv

## Model Training
Model training and evaluation are performed inside `train.ipynb`.
- Text preprocessing: lowercasing, punctuation removal, stopword filtering, stemming, and tokenization
- Feature extraction: `TfidfVectorizer`
- Classifier: `MultinomialNB` pipelines
- Hyperparameter tuning: `GridSearchCV`
- Output artifacts: TF-IDF vectorizer pickles and classifier pickles for production use

## Evaluation Metrics
### Category Model
- Train accuracy: **0.9879**
- Test accuracy: **0.9876**
- precision: **0.99**
- recall: **0.99**


### Intent Model
- Train accuracy: **0.9951**
- Test accuracy: **0.9880**
- precision: **0.94**
- recall: **0.99**


### Priority Model
- Train accuracy: **0.7421**
- Test accuracy: **0.7377**
- precision: **0.83**
- recall: **0.70**


## Usage
1. Ensure all dependencies are installed in the virtual environment.
2. Use `train.ipynb` to train or retrain models and generate updated pickles.
3. Run the Streamlit app with:
```bash
streamlit run app.py
```
4. Enter a customer query in the app and receive predicted Category, Intent, Priority, and a generated reply.

## Notes
- The dataset is humanized for realistic customer support text.
- The Streamlit app is designed for quick testing and can be extended into a production interface.
- `app.py` is an alternate prediction script and can be used as a leaner model-serving entrypoint.
