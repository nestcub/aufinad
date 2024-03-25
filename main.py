from flask import Flask, render_template, request, jsonify
import bestprediction,requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend for non-interactive image generation
from io import BytesIO
import base64
import matplotlib.dates as mdates
from datetime import datetime, timedelta


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

############################################## STOCK PREDICTOR ###########################################################v

@app.route('/stock')
def stockdetails():
    return render_template('stockdetails.html')


@app.route('/predict', methods=['POST'])
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
    plot_data = base64.b64encode(img.getvalue()).decode()
    img.close()

    # Pass the plot image data to the HTML template
    return render_template('result.html', result=result, plot_data=plot_data)

######################################### ROUTES FOR INFORMATION BUCKETS################################################

@app.route('/emergency_fund', methods=['GET', 'POST'])
def emergency_fund_calc():
    if request.method == 'POST':
        monthly_income = float(request.form['monthly_income'])
        monthly_expenses = float(request.form['monthly_expenses'])
        existing_savings = float(request.form['existing_savings'])
        income_label = request.form['monthly_income']
        expenses_label = request.form['monthly_expenses']
        savings_label = request.form['existing_savings']
        
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

        # Create a bar chart with user-defined labels
        labels = [income_label, expenses_label, savings_label]
        values = [monthly_income, monthly_expenses, existing_savings]
        colors = ['green', 'red', 'blue']

        plt.bar(labels, values, color=colors)
        plt.title('Financial Overview')
        plt.xlabel('Categories')
        plt.ylabel('Amount (in currency)')

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plt.close()

        # Convert the BytesIO object to base64 for embedding in HTML
        image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')
        
        return render_template('services/emergency_fund.html', result=result, chart=image_base64)
    return render_template('services/emergency_fund.html')



@app.route('/insurance_planning', methods=['GET', 'POST'])
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

@app.route('/debt_score', methods=['GET', 'POST'])
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

@app.route('/renderchat')
def renderchat():
    return render_template('chatbot.html')

@app.route('/chat', methods=['GET','POST'])
def chat():
    user_message = request.form.get('user_message')
    if user_message:
        response = generate_response(user_message)
        return jsonify({"bot_response": response})
    else:
        return jsonify({"error": "Invalid request"})

def generate_response(user_message):
    # You can add your chatbot logic here based on user messages
    if "predict stock price" in user_message:
        print("Sure! Please provide the stock symbol, start date, and end date for the prediction.")
        return render_template('stockdetails2.html')
    else:
        return "I'm not sure how to respond to that."
    


########################################################financial calculators####################################################################

@app.route('/rendercagr')
def rendercagr():
    return render_template('cagr.html')


@app.route('/calculate_cagr', methods=['POST'])
def calculate_cagr_route():
    if request.method == 'POST':
        initial_value = float(request.form['initial_value'])
        final_value = float(request.form['final_value'])
        num_of_years = float(request.form['num_of_years'])

        cagr_result = ((final_value / initial_value) ** (1/num_of_years)) - 1
        cagr_result = cagr_result * 100

        return render_template('services/cagr.html', cagr_result=cagr_result)
    return render_template('services/cagr.html')




if __name__== "__main__":
    app.run(debug=True)