import pandas as pd
import numpy as np
import random

def generate_transaction_data(n=1000):
    descriptions = ['Payment', 'Transfer', 'Deposit', 'Withdrawal', 'Invoice']
    data = []
    for _ in range(n):
        desc = random.choice(descriptions) + " #" + str(random.randint(1000,9999))
        amount = round(random.uniform(10, 1000), 2)
        date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=random.randint(0, 90))
        data.append([desc, amount, date])
    return pd.DataFrame(data, columns=['Description', 'Amount', 'Date'])

df1 = generate_transaction_data()
df2 = df1.sample(frac=0.8).copy()  # Matching set with some records slightly altered
df2['Amount'] += np.random.uniform(-5,5,len(df2))
df2['Description'] = df2['Description'].apply(lambda x: x.replace('Payment','Pymt') if random.random() > 0.5 else x)

df1.to_csv('ledger_transactions.csv', index=False)
df2.to_csv('bank_transactions.csv', index=False)
