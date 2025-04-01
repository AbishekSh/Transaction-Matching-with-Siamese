import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime

def gen_desc():
    transaction_types = ['Payment', 'Transfer', 'Invoice', 'Withdrawal', 'Deposit', 'Refund', 'ACH Transfer']
    vendors = ['Amazon', 'Walmart', 'Netflix', 'Starbucks', 'Uber', 'Apple', 'Google', 'Costco', 'Shell', 'BestBuy']
    return f"{random.choice(transaction_types)} {random.choice(vendors)} #{random.randint(1000,9999)}"

# Generate realistic ledger and bank transactions
def gen_data(n=1000):
    data_ledger, data_bank = [], []
    base_date = datetime(2024, 1, 1)

    for _ in range(n):
        desc = gen_desc()
        amount = round(random.uniform(5, 2000), 2)
        date = base_date + timedelta(days=random.randint(0, 90))
        
        ledger_entry = [desc, amount, date.strftime('%Y-%m-%d')]
        
        # Simulate slight variations for bank data
        bank_amount = amount + random.uniform(-10, 10)
        bank_date = date + timedelta(days=random.choice([0, 1, -1]))  # slight date shift
        bank_desc = desc.replace('Payment', 'Pmt') if random.random() > 0.7 else desc  # occasional abbreviation
        
        bank_entry = [bank_desc, round(bank_amount, 2), bank_date.strftime('%Y-%m-%d')]
        
        data_ledger.append(ledger_entry)
        data_bank.append(bank_entry)

    ledger_df = pd.DataFrame(data_ledger, columns=['Description', 'Amount', 'Date'])
    bank_df = pd.DataFrame(data_bank, columns=['Description', 'Amount', 'Date'])
    return ledger_df, bank_df

ledger_transactions, bank_transactions = gen_data(1000)

ledger_transactions_path = 'ledger_transactions.csv'
bank_transactions_path = 'bank_transactions.csv'
ledger_transactions.to_csv(ledger_transactions_path, index=False)
bank_transactions.to_csv(bank_transactions_path, index=False)

ledger_transactions_path, bank_transactions_path

