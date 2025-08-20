import pandas as pd

# Load Excel
df = pd.read_excel("data/default_of_credit_card_clients.xls", header=1)

# Rename columns to match ALL_FEATURES
rename_map = {
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "LIMIT_BAL": "limit_bal",
    "AGE": "age",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": "default_payment_next_month"   # ✅ FIX HERE
}


df = df.rename(columns=rename_map)

# Save as CSV for your pipeline
df.to_csv("data/default_of_credit_card_clients.csv", index=False)
print("✅ Converted and saved dataset to data/default_of_credit_card_clients.csv")
