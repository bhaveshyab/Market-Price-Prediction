
import pandas as pd

df = pd.read_csv("tomato_clean.csv")

print("TOTAL ROWS:", len(df))
print("\nSAMPLE RAW DATES:")
print(df["Arrival_Date"].head(15).to_list())

# Compare the original date-parsing method with mixed-format parsing
date_old = pd.to_datetime(
    df["Arrival_Date"], dayfirst=True, errors="coerce"
)

date_mixed = pd.to_datetime(
    df["Arrival_Date"], format="mixed",
    dayfirst=True, errors="coerce"
)

print("\nDATE CHECK")
print("Dates parsed by original method:", date_old.notna().sum())
print("Dates parsed by mixed method:", date_mixed.notna().sum())

print("\nEXAMPLES OF DATES THE ORIGINAL METHOD REJECTED:")
print(df.loc[date_old.isna(), "Arrival_Date"].head(15).to_list())

# Check the target price
price = pd.to_numeric(df["Modal_Price"], errors="coerce")

print("\nPRICE CHECK")
print("Valid numeric prices:", price.notna().sum())
print("Missing or invalid prices:", price.isna().sum())
print("Zero or negative prices:", (price <= 0).sum())

# Check how many rows remain with mixed-format dates
usable = date_mixed.notna() & price.notna() & (price > 0)

print("\nUSABLE ROWS WITH MIXED DATE PARSING:", usable.sum())

if usable.any():
    print("Corrected date range:",
          date_mixed[usable].min().date(),
          "to",
          date_mixed[usable].max().date())
