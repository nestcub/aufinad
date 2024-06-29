## AUFINAD (Automated Financial Advisory)

### Project Overview
AUFINAD is a web application built using Flask that utilizes a deep learning model (LSTM) to predict stock prices. It includes features like live stock price widgets, financial calculators, and a chatbot for financial questions.

### Features
- Stock Price Prediction using LSTM
- Live Stock Price Widget
- Financial Calculators
- Financial Questions Chatbot
- News Section (Web Scraping)
- Database Storage for Previous Predictions

### Prerequisites
- Python 3.7 or higher
- pip

### Setup Instructions

#### Step 1: Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### Step 2: Download the Project
Download the project as a zip file and extract it.

#### Step 3: Install Required Packages
```sh
pip install -r requirements.txt
```

#### Step 4: Configure the Database
Make sure you have SQLite installed. Configure the database by running the following commands in the project directory:

```sh
export FLASK_APP=application.py
flask db init
flask db migrate -m "Initial migration."
flask db upgrade
```

#### Step 5: Create a .env File
Create a `.env` file in the project root directory to store your google API key and a secret key[secret key can be anything, i just randomly clicked through my keyboard which included letters and numbers].
```
SECRET_KEY=your_secret_key
GOOGLE_API_KEY=your_google_api_key
```

#### Step 6: Create a .gitignore File
Before you commit your project to a Git repository, create a `.gitignore` file in the project root directory. This file tells Git which files and directories to ignore. 

Copy the contents of the `.gitignore` file from [Python.gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore) or add the following custom entries:
- Remember, when adding custom entries to gitignore file, add the correct names of your virtual environment, environment variables file, etc.

```
# Environment variables
.env

# Virtual environment
venv/

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/
```

### Running the Project
```sh
python application.py
```
This will run the project on Flask's local server.

### Usage
1. Navigate to the home page.
2. In the navbar, find services such as the chatbot, news, and calculators.
3. Scroll down to the Stock Price Predictor section, select the stock, enter the start and end dates, and click on predict.
4. The ML algorithm will run in the backend and provide graphs of actual vs. predicted prices.

### Additional Code Snippets
The `bestprediction.py` file includes the original algorithm adjusted to run in Flask. 

### Import the Required Libraries
The main algorithm involves importing the following libraries:
```python
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
```

### Example Stock Prediction Function
Here is an example function to predict stock prices:
```python
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, sequence_length):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def predict_stock_price(stock_symbol, start_date, end_date, sequence_length):
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    scaled_data, scaler = preprocess_data(stock_data)
    X, y = create_sequences(scaled_data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_and_train_model(X_train, y_train, sequence_length)
    mse = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error on Test Data for {stock_symbol}: {mse:.6f}")
    last 10 days = scaled_data[-sequence_length:]
    next_day_scaled_price = model.predict(last_10_days.reshape(1, sequence_length, 1))
    next_day_price = scaler.inverse_transform(next_day_scaled_price)
    print(f"Predicted price for {stock_symbol} on the next trading day: ₹{next_day_price[0][0]:.2f}")
    y_pred = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred)
    y_actual = scaler.inverse_transform(y)
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index[-len(y):], y_actual, label='Actual Price', color='b')
    plt.plot(stock_data.index[-len(y_pred):], y_pred, label='Predicted Price', color='r')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (₹)')
    plt.title(f'Actual vs. Predicted Stock Prices for {stock_symbol}')
    plt.legend()
    plt.show()
```

### Notes
- The application uses SQLAlchemy for database interactions. Ensure that your database URI is correctly set in the `.env` file.
- The application uses the Gemini API for the chatbot functionality. Ensure the API key is set correctly in the `.env` file.
- For good practice, security, and ease of database configuration add (SQLALCHEMY_DATABASE_URI = 'sqlite:///past_predictions.db') in the `.env` file and avoid adding (application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///past_predictions.db') in application.py.

### Contribution
Feel free to fork this project, make improvements, and submit pull requests.
---