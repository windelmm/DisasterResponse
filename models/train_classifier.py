import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pickle

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    X = df['message']
    y = df.iloc[:, 4:]
    
    catNames = y.columns
    
    return X, y, catNames

def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])

    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)}

    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters, verbose=3, cv=3)    
    return model

def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)

    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()