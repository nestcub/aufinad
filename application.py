from flask import Flask, render_template, request, jsonify,redirect,url_for
import bestprediction,requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend for non-interactive image generation
from io import BytesIO
import base64
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
from dotenv import load_dotenv
import os
load_dotenv()

application = Flask(__name__, static_url_path='/static')
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///past_predictions.db'
db = SQLAlchemy(application)
application.secret_key = os.environ.get('SECRET_KEY')

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stock_symbol = db.Column(db.String(10), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    next_day_price = db.Column(db.Integer, nullable = False)
    plot_data = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f"Prediction(stock_symbol='{self.stock_symbol}', start_date='{self.start_date}', end_date='{self.end_date}')"
    

# List of Nifty Fifty stocks
nifty_fifty_stocks = [
    "ASIANPAINT.NS", "EICHERMOT.NS", "NESTLEIND.NS", "GRASIM.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "M&M.NS",
    "RELIANCE.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "WIPRO.NS", "DRREDDY.NS",
    "TITAN.NS", "SBIN.NS", "KOTAKBANK.NS", "UPL.NS", "INFY.NS",
    "BAJFINANCE.NS", "SUNPHARMA.NS", "JSWSTEEL.NS", "HDFCBANK.NS", "TCS.NS",
    "ICICIBANK.NS", "POWERGRID.NS", "MARUTI.NS", "INDUSINDBK.NS", "AXISBANK.NS",
    "HCLTECH.NS", "ONGC.NS", "NTPC.NS", "BHARTIARTL.NS", "TECHM.NS",
    "ZEEL.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "BSE.NS", "GAIL.NS", "BRITANNIA.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS"
]

@application.route('/')
@application.route('/home')
def home():
    return render_template('index.html',nifty_fifty_stocks = nifty_fifty_stocks)

# @application.route('/')
# @application.route('/home')
# def home():
#     # Get real-time stock prices
#     stock_prices = real_time_stock_prices(nifty_fifty_stocks)
#     return render_template('index.html', stock_prices = stock_prices)

# def real_time_stock_prices(stock_symbols):
#     stock_prices = {}
#     for symbol in stock_symbols:
#         stock = yf.Ticker(symbol)
#         data = stock.history(period='1d')
#         if not data.empty:
#             latest_price = data['Close'].iloc[-1]
#             stock_prices[symbol] = round(latest_price, 2)
#     return stock_prices    
############################################## STOCK PREDICTOR ###########################################################v

@application.route('/stock')
def stockdetails():
    predictions = Prediction.query.all()
    return render_template('chatbots/stockdetails.html', predictions = predictions)


@application.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form.get('stock_symbol')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    sequence_length = 10

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    result= bestprediction.predict_stock_price(stock_symbol, start_date, end_date, sequence_length)

    # Create a date range from start_date to end_date
    date_range = [start_date + timedelta(days=i) for i in range(len(result['actual_prices']))]
    date_range = [date.strftime("%Y-%m-%d") for date in date_range]

    actual_prices = result['actual_prices']
    predicted_prices = result['predicted_prices']
    print(actual_prices)

    plt.figure(figsize=(10, 6))
    plt.plot(date_range, actual_prices, label='Actual Price', color='red')
    plt.plot(date_range, predicted_prices, label='Predicted Price', color='blue')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.legend()

    # Adjust layout to prevent xlabel from being cut off
    plt.tight_layout()
    
    # Save the plot as an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    lstm_data = base64.b64encode(img.getvalue()).decode()
    lstm_data_for_database = base64.b64encode(img.getvalue())
    img.close()

    actual_prices = get_stock_prices(stock_symbol, start_date, end_date)['Close']

    # Calculate moving average
    moving_avg = calculate_moving_average(actual_prices, sequence_length)

    # Calculate weighted moving average
    weights = [1 / (sequence_length - i) for i in range(sequence_length)]
    weighted_avg = calculate_weighted_moving_average(actual_prices, sequence_length, weights)

    # Calculate exponential smoothing
    alpha = 0.3  # Smoothing factor
    exp_smoothing = calculate_exponential_smoothing(actual_prices, alpha)

    next_trading_day = end_date + timedelta(days=1)

    ma_pp = moving_avg.iloc[-1]  
    wma_pp = weighted_avg.iloc[-1]  
    es_pp = exp_smoothing.iloc[-1]  

    # Create a date range from start_date to end_date
    date_range = [start_date + timedelta(days=i) for i in range(len(actual_prices))]
    date_range = [date.strftime("%Y-%m-%d") for date in date_range]

    # Create separate plots
    plot_data = []

    # Actual vs. Moving Average plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_range, actual_prices, label='Actual Price', color='red')
    ax.plot(date_range, moving_avg, label='Moving Average', color='green')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs. Moving Average')
    ax.legend()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data.append(base64.b64encode(img.getvalue()).decode())
    img.close()
    plt.close()

    # Actual vs. Weighted Moving Average plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_range, actual_prices, label='Actual Price', color='red')
    ax.plot(date_range, weighted_avg, label='Weighted Moving Average', color='purple')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs. Weighted Moving Average')
    ax.legend()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data.append(base64.b64encode(img.getvalue()).decode())
    img.close()
    plt.close()

    # Actual vs. Exponential Smoothing plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_range, actual_prices, label='Actual Price', color='red')
    ax.plot(date_range, exp_smoothing, label='Exponential Smoothing', color='orange')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs. Exponential Smoothing')
    ax.legend()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data.append(base64.b64encode(img.getvalue()).decode())
    img.close()
    plt.close()

    try:
        # Save the prediction data to the database
        prediction = Prediction(
            stock_symbol=stock_symbol,
            start_date=start_date,
            end_date=end_date,
            next_day_price = result['next_day_price'],
            plot_data=lstm_data_for_database  # Convert plot_data to bytes
        )
        db.session.add(prediction)
        db.session.commit()
        db.session.close()  # Close the database session
    except Exception as e:
        db.session.rollback()  # Rollback changes if an error occurs
        return f"An error occurred while saving prediction data: {str(e)}", 500


    # Pass the plot image data to the HTML template
    return render_template('chatbots/result.html', plot_data=plot_data,result=result, lstm_data=lstm_data,ma_pp = ma_pp,wma_pp = wma_pp, es_pp = es_pp)

def get_stock_prices(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Helper functions for calculating moving average, weighted moving average, and exponential smoothing
def calculate_moving_average(data, window_size):
    moving_avg = data.rolling(window=window_size).mean()
    return moving_avg

def calculate_weighted_moving_average(data, window_size, weights):
    def weighted_mean(values, weights):
        return sum(values * weights) / sum(weights)

    weighted_avg = data.rolling(window=window_size).apply(weighted_mean, args=(weights,), raw=True)
    return weighted_avg

def calculate_exponential_smoothing(data, alpha):
    exp_smoothing = data.ewm(alpha=alpha, adjust=False).mean()
    return exp_smoothing
######################################### ROUTE FOR RECORD  DELETION FROM DATABASE################################################
@application.route('/delete/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    prediction = Prediction.query.get(prediction_id)
    if prediction:
        try:
            db.session.delete(prediction)
            db.session.commit()
            db.session.close()
            return redirect(url_for('stockdetails'))
        except Exception as e:
            db.session.rollback()
            return f"An error occurred while deleting prediction data: {str(e)}", 500
    else:
        return f"Prediction with ID {prediction_id} not found", 404

######################################### ROUTES FOR INFORMATION BUCKETS################################################

@application.route('/emergency_fund', methods=['GET', 'POST'])
def emergency_fund_calc():
    if request.method == 'POST':
        monthly_income = float(request.form['monthly_income'])
        monthly_expenses = float(request.form['monthly_expenses'])
        existing_savings = float(request.form['existing_savings'])
        
        recommended_emergency_fund = max(6 * monthly_expenses, 50000)

        if monthly_income < monthly_expenses:
            result = "Your monthly expenses exceed your income. You should work on reducing expenses. "
        else:
            if existing_savings >= recommended_emergency_fund:
                result = "Great! You already have a sufficient Emergency Fund."
            else:
                additional_savings_required = recommended_emergency_fund - existing_savings
                monthly_savings_required = additional_savings_required / 12
                result = f"Based on the rule of thumb, your recommended Emergency Fund for 6 months is: ₹{recommended_emergency_fund:.2f}"
                if monthly_income > monthly_expenses:
                    result += f" To reach this goal, you need to save an additional ₹{monthly_savings_required:.2f} per month."
                else:
                    additional_earnings_required = monthly_savings_required
                    result += f" To reach this goal, you need to earn an additional ₹{additional_earnings_required:.2f} per month (or reduce expenses)."

        return render_template('services/emergency_fund.html', result=result)
    return render_template('services/emergency_fund.html')



@application.route('/insurance_planning', methods=['GET', 'POST'])
def plan_insurance():
    if request.method == 'POST':
        marital_status = request.form['marital_status'].strip().lower()
        children = int(request.form['children']) if marital_status == 'married' else 0
        
        advice = []
        advice.append("Insurance Planning Advice:")
        advice.append(" Health Insurance: Ensure you have health insurance for yourself and your family.")
        advice.append(" Life Insurance: Consider life insurance to provide financial security to your family in case of an unfortunate event.")
        if children > 0:
            advice.append(" Education Insurance: If you have children, plan for their education expenses by investing in education insurance.")
        if marital_status == 'married':
            advice.append(" Homeowners or Renters Insurance: Protect your home and belongings with appropriate insurance.")
        
        tips = []
        tips.append("Tips for Finding the Best Insurance:")
        tips.append(" Research and compare policies from multiple insurance providers to get the best coverage at a competitive price.")
        tips.append(" Read the policy documents carefully to understand coverage, terms, and conditions.")
        tips.append(" Seek recommendations from trusted sources and read reviews to assess the reputation of insurance companies.")
        tips.append(" Beware of red flags such as exceptionally low premiums, unsolicited calls, and providers with poor customer service.")
        
        return render_template('services/insurance_planning.html', advice=advice, tips=tips)
    return render_template('services/insurance_planning.html')

@application.route('/debt_score', methods=['GET', 'POST'])
def debt_score():
    if request.method == 'POST':
        total_debt = float(request.form['total_debt'])
        annual_income = float(request.form['annual_income'])
        monthly_expenses = float(request.form['monthly_expenses'])
        
        debt_to_income_ratio = total_debt / annual_income
        
        if debt_to_income_ratio > 0.5:
            debt_score = "High"
        elif debt_to_income_ratio > 0.3:
            debt_score = "Moderate"
        else:
            debt_score = "Low"
        
        advice = []
        
        if debt_score == "High":
            advice.append("Advice for Managing High Debt:")
            advice.append("1. Create a detailed budget to control your spending and allocate more towards debt payments.")
            advice.append("2. Focus on paying off high-interest debts first to reduce overall interest costs.")
            advice.append("3. Consider debt consolidation or negotiating with creditors for better terms.")
            advice.append("4. Explore additional sources of income to accelerate debt repayment.")
        elif debt_score == "Moderate":
            advice.append("Advice for Managing Moderate Debt:")
            advice.append("1. Continue to make consistent debt payments and avoid taking on more debt.")
            advice.append("2. Prioritize paying off high-interest debts to reduce interest costs.")
            advice.append("3. Create a financial plan to reduce debt systematically.")
        elif debt_score == "Low":
            advice.append("Advice for Managing Low Debt:")
            advice.append("1. Maintain your responsible financial habits and avoid unnecessary debt.")
            advice.append("2. Build an emergency fund to cover unexpected expenses.")
        
        return render_template('services/debt_score.html', debt_score=debt_score, advice=advice)
    
    return render_template('services/debt_score.html')
    
########################################################CHATBOT####################################################################

@application.route('/renderchat')
def renderchat():
    return render_template('chatbots/chatbot.html')

# @application.route('/chat', methods=['GET','POST'])
# def chat():
#     user_message = request.form.get('user_message')
#     if user_message:
#         response = generate_response(user_message)
#         return jsonify({"bot_response": response})
#     else:
#         return jsonify({"error": "Invalid request"})

# def generate_response(user_message):
#     # You can add your chatbot logic here based on user messages
#     if "predict stock price" in user_message:
#         print("Sure! Please provide the stock symbol, start date, and end date for the prediction.")
#         return render_template('chatbots/stockdetails2.html')
#     else:
#         return "I'm not sure how to respond to that."    

import textwrap
import google.generativeai as genai

def to_markdown(text):
  text = text.replace('•', '  *')  # Use two spaces for better formatting
  return textwrap.indent(text, '> ', predicate=lambda _: True)

# Configure Generative AI model (assuming API key is set in environment variable)
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

from flask import session
from flask import redirect, url_for

@application.route('/chat/clear', methods=['GET'])
def clear_chat():
    if 'messages' in session:
        session.pop('messages')
    return redirect(url_for('chat'))

@application.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'messages' not in session:
        session['messages'] = []

    messages = session['messages']

    if request.method == 'POST':
        user_message = request.form['message']
        messages.append({'role': 'user', 'parts': [user_message]})
        response = model.generate_content(messages)
        model_response = to_markdown(response.text)
        messages.append({'role': 'model', 'parts': [model_response]})
        session['messages'] = messages

    return render_template('chatbots/chatbot.html', messages=messages)

########################################################financial calculators####################################################################

@application.route('/rendercagr')
def rendercagr():
    return render_template('services/cagr.html')


@application.route('/calculate_cagr', methods=['POST'])
def calculate_cagr_route():
    if request.method == 'POST':
        initial_value = float(request.form['initial_value'])
        final_value = float(request.form['final_value'])
        num_of_years = float(request.form['num_of_years'])

        cagr_result = ((final_value / initial_value) ** (1/num_of_years)) - 1
        cagr_result = cagr_result * 100

        return render_template('services/cagr.html', cagr_result=cagr_result)
    return render_template('services/cagr.html')

########################################################NEWS####################################################################
import requests
from bs4 import BeautifulSoup
def scrape_moneycontrol():
    # URL of the webpage
    url = "https://www.moneycontrol.com/news/business/stocks/"

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all <p> tags
    paragraphs = soup.find_all("p")

    # Extracting content from <p> tags
    content = [p.text.strip() for p in paragraphs]

    # Find all <h2> tags containing <a> tags
    headings_with_links = soup.find_all("h2")

    # Extracting titles and URLs from <h2> tags
    headings_links = []
    for h2 in headings_with_links:
        link = h2.find("a")
        if link:
            title = link.text.strip()
            url = link.get("href")
            headings_links.append({"title": title, "url": url})

    return content, headings_links

@application.route("/rendernews")
def rendernews():
    content, headings_links = scrape_moneycontrol()
    return render_template("services/news.html", content=content, headings_links=headings_links)


with application.app_context():
    db.create_all()


if __name__== "__main__":
    application.run(debug=True)