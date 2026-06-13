"""
Sales Data Analysis Project using Pandas
=========================================
This project covers:
1. Dataset generation (synthetic sales data)
2. Data cleaning & preprocessing
3. Exploratory Data Analysis (EDA)
4. KPI calculations
5. Visualizations (saved as PNG)
6. Summary report (saved as CSV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── 0. Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

# ── 1. Generate Synthetic Sales Dataset ───────────────────────────────────────
n = 1000

regions    = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Furniture", "Groceries", "Sports"]
channels   = ["Online", "In-Store", "Phone"]

dates = pd.date_range(start="2023-01-01", end="2023-12-31", periods=n)

df = pd.DataFrame({
    "order_id"       : range(1001, 1001 + n),
    "date"           : dates,
    "region"         : np.random.choice(regions, n),
    "category"       : np.random.choice(categories, n),
    "channel"        : np.random.choice(channels, n),
    "units_sold"     : np.random.randint(1, 50, n),
    "unit_price"     : np.round(np.random.uniform(10, 500, n), 2),
    "discount_pct"   : np.random.choice([0, 5, 10, 15, 20], n),
    "customer_age"   : np.random.randint(18, 70, n),
    "customer_rating": np.round(np.random.uniform(1, 5, n), 1),
})

# Introduce some nulls for realism
null_idx = np.random.choice(df.index, size=30, replace=False)
df.loc[null_idx[:15], "customer_rating"] = np.nan
df.loc[null_idx[15:], "discount_pct"]    = np.nan

print("=" * 60)
print("SALES DATA ANALYSIS PROJECT")
print("=" * 60)

# ── 2. Data Overview ───────────────────────────────────────────────────────────
print("\n── Dataset Shape ──")
print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

print("\n── Data Types ──")
print(df.dtypes.to_string())

print("\n── Missing Values ──")
print(df.isnull().sum().to_string())

print("\n── First 5 Rows ──")
print(df.head().to_string(index=False))

# ── 3. Data Cleaning ──────────────────────────────────────────────────────────
# Fill missing discount with median; rating with mean
df["discount_pct"].fillna(df["discount_pct"].median(), inplace=True)
df["customer_rating"].fillna(round(df["customer_rating"].mean(), 1), inplace=True)

# Drop duplicates (none expected, but good practice)
df.drop_duplicates(inplace=True)

print("\n── After Cleaning – Missing Values ──")
print(df.isnull().sum().to_string())

# ── 4. Feature Engineering ────────────────────────────────────────────────────
df["revenue"]        = df["units_sold"] * df["unit_price"]
df["discount_amt"]   = df["revenue"] * df["discount_pct"] / 100
df["net_revenue"]    = df["revenue"] - df["discount_amt"]
df["month"]          = df["date"].dt.month
df["month_name"]     = df["date"].dt.strftime("%b")
df["quarter"]        = df["date"].dt.quarter

# ── 5. KPIs ───────────────────────────────────────────────────────────────────
total_revenue     = df["net_revenue"].sum()
avg_order_value   = df["net_revenue"].mean()
total_units       = df["units_sold"].sum()
avg_rating        = df["customer_rating"].mean()
total_discount    = df["discount_amt"].sum()

print("\n" + "=" * 60)
print("KEY PERFORMANCE INDICATORS (KPIs)")
print("=" * 60)
print(f"  Total Net Revenue   : ${total_revenue:>12,.2f}")
print(f"  Avg Order Value     : ${avg_order_value:>12,.2f}")
print(f"  Total Units Sold    : {total_units:>13,}")
print(f"  Avg Customer Rating : {avg_rating:>13.2f} / 5.0")
print(f"  Total Discounts     : ${total_discount:>12,.2f}")

# ── 6. Analysis Summaries ─────────────────────────────────────────────────────
print("\n── Revenue by Region ──")
region_rev = (df.groupby("region")["net_revenue"]
               .agg(["sum","mean","count"])
               .rename(columns={"sum":"Total Revenue","mean":"Avg Revenue","count":"Orders"})
               .sort_values("Total Revenue", ascending=False))
print(region_rev.to_string())

print("\n── Revenue by Category ──")
cat_rev = (df.groupby("category")["net_revenue"]
             .agg(["sum","mean","count"])
             .rename(columns={"sum":"Total Revenue","mean":"Avg Revenue","count":"Orders"})
             .sort_values("Total Revenue", ascending=False))
print(cat_rev.to_string())

print("\n── Revenue by Channel ──")
chan_rev = (df.groupby("channel")["net_revenue"]
              .agg(["sum","mean"])
              .rename(columns={"sum":"Total Revenue","mean":"Avg Revenue"})
              .sort_values("Total Revenue", ascending=False))
print(chan_rev.to_string())

print("\n── Monthly Revenue ──")
month_rev = (df.groupby(["month","month_name"])["net_revenue"]
               .sum()
               .reset_index()
               .sort_values("month"))
month_rev.drop(columns="month", inplace=True)
print(month_rev.to_string(index=False))

print("\n── Quarterly Revenue ──")
qtr_rev = df.groupby("quarter")["net_revenue"].sum()
print(qtr_rev.to_string())

print("\n── Top 5 Highest Revenue Orders ──")
print(df.nlargest(5, "net_revenue")[
    ["order_id","date","region","category","units_sold","unit_price","net_revenue"]
].to_string(index=False))

# ── 7. Correlation ────────────────────────────────────────────────────────────
numeric_cols = ["units_sold","unit_price","discount_pct","net_revenue","customer_rating"]
corr = df[numeric_cols].corr().round(2)
print("\n── Correlation Matrix ──")
print(corr.to_string())

# ── 8. Visualizations ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20))
fig.suptitle("Sales Data Analysis Dashboard – 2023", fontsize=20, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

PALETTE = ["#2563EB","#7C3AED","#DB2777","#D97706","#059669"]

# 8a. Monthly Revenue (line)
ax1 = fig.add_subplot(gs[0, :2])
mr  = df.groupby("month")["net_revenue"].sum()
ax1.plot(mr.index, mr.values, marker="o", color=PALETTE[0], linewidth=2.5)
ax1.fill_between(mr.index, mr.values, alpha=0.15, color=PALETTE[0])
ax1.set_title("Monthly Net Revenue", fontsize=13, fontweight="bold")
ax1.set_xlabel("Month"); ax1.set_ylabel("Net Revenue ($)")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=30)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax1.grid(True, alpha=0.3)

# 8b. Revenue by Region (bar)
ax2 = fig.add_subplot(gs[0, 2])
region_rev["Total Revenue"].sort_values().plot(kind="barh", ax=ax2, color=PALETTE)
ax2.set_title("Revenue by Region", fontsize=13, fontweight="bold")
ax2.set_xlabel("Net Revenue ($)")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax2.grid(axis="x", alpha=0.3)

# 8c. Revenue by Category (bar)
ax3 = fig.add_subplot(gs[1, 0])
cat_rev["Total Revenue"].sort_values().plot(kind="barh", ax=ax3, color=PALETTE)
ax3.set_title("Revenue by Category", fontsize=13, fontweight="bold")
ax3.set_xlabel("Net Revenue ($)")
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax3.grid(axis="x", alpha=0.3)

# 8d. Revenue by Channel (pie)
ax4 = fig.add_subplot(gs[1, 1])
chan_vals = chan_rev["Total Revenue"]
ax4.pie(chan_vals, labels=chan_vals.index, autopct="%1.1f%%",
        colors=PALETTE[:len(chan_vals)], startangle=140,
        textprops={"fontsize": 10})
ax4.set_title("Revenue by Channel", fontsize=13, fontweight="bold")

# 8e. Quarterly Revenue (bar)
ax5 = fig.add_subplot(gs[1, 2])
qtr_rev.plot(kind="bar", ax=ax5, color=PALETTE[:4], edgecolor="white", width=0.6)
ax5.set_title("Quarterly Revenue", fontsize=13, fontweight="bold")
ax5.set_xlabel("Quarter"); ax5.set_ylabel("Net Revenue ($)")
ax5.set_xticklabels([f"Q{i}" for i in range(1, 5)], rotation=0)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax5.grid(axis="y", alpha=0.3)

# 8f. Units Sold Distribution (hist)
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(df["units_sold"], bins=20, color=PALETTE[2], edgecolor="white", alpha=0.85)
ax6.set_title("Units Sold Distribution", fontsize=13, fontweight="bold")
ax6.set_xlabel("Units Sold"); ax6.set_ylabel("Frequency")
ax6.grid(axis="y", alpha=0.3)

# 8g. Unit Price vs Net Revenue (scatter)
ax7 = fig.add_subplot(gs[2, 1])
sc  = ax7.scatter(df["unit_price"], df["net_revenue"],
                  c=df["units_sold"], cmap="viridis", alpha=0.4, s=18)
plt.colorbar(sc, ax=ax7, label="Units Sold")
ax7.set_title("Unit Price vs Net Revenue", fontsize=13, fontweight="bold")
ax7.set_xlabel("Unit Price ($)"); ax7.set_ylabel("Net Revenue ($)")
ax7.grid(alpha=0.3)

# 8h. Customer Rating Distribution (hist)
ax8 = fig.add_subplot(gs[2, 2])
ax8.hist(df["customer_rating"], bins=20, color=PALETTE[4], edgecolor="white", alpha=0.85)
ax8.set_title("Customer Rating Distribution", fontsize=13, fontweight="bold")
ax8.set_xlabel("Rating (1–5)"); ax8.set_ylabel("Frequency")
ax8.grid(axis="y", alpha=0.3)

plt.savefig("/mnt/user-data/outputs/sales_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  [Saved] sales_dashboard.png")

# ── 9. Export Summary CSVs ────────────────────────────────────────────────────
region_rev.to_csv("/mnt/user-data/outputs/revenue_by_region.csv")
cat_rev.to_csv("/mnt/user-data/outputs/revenue_by_category.csv")
month_rev.to_csv("/mnt/user-data/outputs/monthly_revenue.csv", index=False)
df.to_csv("/mnt/user-data/outputs/cleaned_sales_data.csv", index=False)

print("  [Saved] revenue_by_region.csv")
print("  [Saved] revenue_by_category.csv")
print("  [Saved] monthly_revenue.csv")
print("  [Saved] cleaned_sales_data.csv")

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)
