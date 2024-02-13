def calculate_debt_score():
    print("Welcome to the Debt Score Calculator!")
    print("Managing debt and understanding interest rates are essential for your financial well-being.")
    print("Debt can be categorized as high, moderate, or low, depending on factors like the amount owed, interest rates, and your ability to make payments.")
    
    # Gather user's financial information
    total_debt = float(input("Please enter your total debt amount in rupees: ₹"))
    annual_income = float(input("Please enter your annual income in rupees: ₹"))
    monthly_expenses = float(input("Please enter your monthly expenses in rupees: ₹"))
    
    # Calculate the debt-to-income ratio
    debt_to_income_ratio = total_debt / annual_income
    
    # Determine the debt score
    if debt_to_income_ratio > 0.5:
        debt_score = "High"
    elif debt_to_income_ratio > 0.3:
        debt_score = "Moderate"
    else:
        debt_score = "Low"
    
    # Provide debt information and advice
    print("\nYour debt score is: {}".format(debt_score))
    print("\nDebt Score Categories:")
    print("  - High: Your debt is a significant portion of your income, and you should consider reducing it.")
    print("  - Moderate: Your debt is manageable, but be cautious and avoid accumulating more debt.")
    print("  - Low: Your debt is relatively low, which is a healthy financial situation.")
    
    if debt_score == "High":
        print("\nAdvice for Managing High Debt:")
        print("1. Create a detailed budget to control your spending and allocate more towards debt payments.")
        print("2. Focus on paying off high-interest debts first to reduce overall interest costs.")
        print("3. Consider debt consolidation or negotiating with creditors for better terms.")
        print("4. Explore additional sources of income to accelerate debt repayment.")
    
    if debt_score == "Moderate":
        print("\nAdvice for Managing Moderate Debt:")
        print("1. Continue to make consistent debt payments and avoid taking on more debt.")
        print("2. Prioritize paying off high-interest debts to reduce interest costs.")
        print("3. Create a financial plan to reduce debt systematically.")
    
    if debt_score == "Low":
        print("\nAdvice for Managing Low Debt:")
        print("1. Maintain your responsible financial habits and avoid unnecessary debt.")
        print("2. Build an emergency fund to cover unexpected expenses.")
    
    print("\nRemember, reducing debt and paying attention to interest rates can significantly improve your financial stability.")

if __name__ == "__main__":
    calculate_debt_score()
