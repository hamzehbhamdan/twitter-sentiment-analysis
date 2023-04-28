import pandas as pd

# Read the CSV file
df = pd.read_csv('companyvalues.csv')

# Sort the dataframe by ticker and then by date within each ticker
df = df.sort_values(by=['ticker_symbol', 'day_date'])

# Create a new column that shows the change in close value from the previous row
df['change_in_close'] = df.groupby(['ticker_symbol'])['close_value'].apply(lambda x: (x - x.shift(1)).round(2)).values

# Save the updated dataframe to the original CSV file
df.to_csv('companyvalues.csv', index=False)

# Print a message to confirm that the file has been updated
print("CSV file updated successfully.")
