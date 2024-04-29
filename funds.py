def calculate_emergency_fund():
    print("Welcome to the Emergency Fund Calculator!")
    print("\nAn Emergency Fund is a crucial financial safety net, providing you with the peace of mind and the ability to navigate unexpected expenses such as medical bills, job loss, or sudden repairs without risking your financial stability.\n")
    
    # Gather user's financial information
    monthly_income = float(input("Please enter your monthly income in rupees: ₹"))
    monthly_expenses = float(input("Please enter your monthly expenses in rupees: ₹"))
    
    # Check if the user already has an Emergency Fund
    existing_savings = float(input("Do you already have savings or an Emergency Fund? If so, enter the amount in rupees, otherwise, enter ₹0: "))
    
    # Calculate the recommended Emergency Fund for 6 months with a minimum of ₹50,000
    recommended_emergency_fund = max(6 * monthly_expenses, 50000)  # Saving at least 6 months' worth of expenses or ₹50,000, whichever is higher
    
    if monthly_income < monthly_expenses:
        print("Your monthly expenses exceed your income. You should work on reducing expenses.")
    else:
        if existing_savings >= recommended_emergency_fund:
            print("Great! You already have a sufficient Emergency Fund.")
        else:
            additional_savings_required = recommended_emergency_fund - existing_savings
            monthly_savings_required = additional_savings_required / 12
            print("Based on the rule of thumb, your recommended Emergency Fund for 6 months is: ₹{:.2f}".format(recommended_emergency_fund))
            if monthly_income > monthly_expenses:
                print("To reach this goal, you need to save an additional ₹{:.2f} per month.".format(monthly_savings_required))
            else:
                additional_earnings_required = monthly_savings_required
                print("To reach this goal, you need to earn an additional ₹{:.2f} per month (or reduce expenses).".format(additional_earnings_required))

if __name__ == "__main__":
    calculate_emergency_fund()
