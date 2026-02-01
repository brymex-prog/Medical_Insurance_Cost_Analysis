#Fundamentals libraries for data processing, numerical computations, and visualizations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing data using PANDAS
medical_df = pd.read_csv('medical_insurance.csv')

#Checking data information
print('The number of rows and colums in the Medical Insurance dataset is', 
      medical_df.shape[0], 'and', medical_df.shape[1], 'respectively')

#Data Overview
medical_df.info()

#Type conversion for the Alcohol Frequency Column
medical_df['alcohol_freq'] = medical_df['alcohol_freq'].astype('str')
medical_df.loc[medical_df["alcohol_freq"] == "nan", "alcohol_freq"] = 'No Alcohol'

#Checking for null after cleaning
print(medical_df.isnull().sum().sum())

#Summary Statistics (using the important numerical variables)

num_variable = [
    'age', 'income','bmi', 'systolic_bp', 'diastolic_bp','ldl','hba1c', 'deductible', 'copay','risk_score',
    'annual_medical_cost', 'annual_premium',
    'monthly_premium', 'claims_count', 'avg_claim_amount', 'total_claims_paid'
]

summary_df = pd.DataFrame({
    'Count': medical_df[num_variable].count(),
    'Mean': medical_df[num_variable].mean(),
    'Median': medical_df[num_variable].median(),
    'Std_Dev': medical_df[num_variable].std(),
    'Min': medical_df[num_variable].min(),
    '25%': medical_df[num_variable].quantile(0.25),
    '75%': medical_df[num_variable].quantile(0.75),
    'Max': medical_df[num_variable].max()
})
summary_df = summary_df.round(2)

#Table as an image

fig, ax = plt.subplots(figsize=(16, 7))
ax.axis('off')

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    rowLabels=summary_df.index,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.6)

plt.title(
    "Summary Statistics of Key Medical and Insurance Variables",
    fontsize=14,
    pad=20
)

plt.savefig(
    "summary_statistics_medical_insurance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


#Data Handling - Transformation
medical_df['annual_medical_cost_log'] = np.log1p(medical_df['annual_medical_cost'])

#Creating BMI Category
medical_df['bmi_category'] = pd.cut(
    medical_df['bmi'],
    bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

#Creating Age Group
medical_df['age_group'] = pd.cut(
    medical_df['age'],
    bins=[17, 30, 45, 60, 100],
    labels=['18–30', '31–45', '46–60', '60+']
)

#Before Transformation
plt.figure(figsize=(8, 5))
plt.hist(medical_df['annual_medical_cost'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Annual Medical Cost (Before Transformation)')
plt.xlabel('Cost ($)')
plt.ylabel('Frequency')

plt.tight_layout()

# Save image
plt.savefig(
    "before_transformation.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

#After Transformation
plt.figure(figsize=(8, 5))
plt.hist(medical_df['annual_medical_cost_log'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Annual Medical Cost (After Transformation)')
plt.xlabel('Cost ($)')
plt.ylabel('Frequency')

plt.tight_layout()

# Save image
plt.savefig(
    "after_transformation.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


#Table of Annual Medical Cost by Age Group

age_group_cost_df = (
    medical_df
    .groupby('age_group', observed=True)
    .agg(
        Count=('annual_medical_cost_log', 'count'),
        Mean_Cost=('annual_medical_cost_log', 'mean'),
        Median_Cost=('annual_medical_cost_log', 'median'),
        Std_Dev=('annual_medical_cost_log', 'std')
    )
    .round(2)
)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis('off')  




# Figure for the Age Group Table
table = ax.table(
    cellText=age_group_cost_df.values,
    colLabels=age_group_cost_df.columns,
    rowLabels=age_group_cost_df.index,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)


plt.title(
    "Annual Medical Cost by Age Group",
    fontsize=13,
    pad=15
)


plt.savefig("age_group_cost_table.png", dpi=300, bbox_inches='tight')
plt.show()

#Chronic Disease Burden by BMI Category

bmi_chronic_table = (
    medical_df
    .groupby('bmi_category', observed=True)
    .agg(
        Avg_Chronic=('chronic_count', 'mean'),
        Chronic_Rate=('chronic_count', lambda x: (x > 0).mean() * 100)
    )
    .round(2)
)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis('off') 


table = ax.table(
    cellText=bmi_chronic_table.values,
    colLabels=bmi_chronic_table.columns,
    rowLabels=bmi_chronic_table.index,
    cellLoc='center',
    loc='center'
)

# Formatting
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

# Title
plt.title(
    "Chronic Disease Burden by BMI Category",
    fontsize=13,
    pad=15
)

# Save image
plt.savefig(
    "bmi_chronic_burden_table.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

#Visualizations

# Figure 1 - Annual Medical Cost 

plt.figure()
sns.regplot(
    data=medical_df,
    x='age',
    y='annual_medical_cost_log',
    scatter_kws={'alpha': 0.3}
)

plt.title("RegPlot: Plot of Annual Medical Cost with Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost ($)")
plt.annotate(
    "Clear upward trend\nHigher costs at older ages",
    xy=(60, medical_df['annual_medical_cost_log'].quantile(0.75)),
    xytext=(30, medical_df['annual_medical_cost_log'].quantile(0.9)),
    arrowprops=dict(arrowstyle="->")
)
plt.tight_layout()

# Save image
plt.savefig(
    "annual_medical_trend.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Plot 2: Medical Cost by Smoker Status

plt.figure()
sns.boxplot(
    data=medical_df,
    x='smoker',
    y='annual_medical_cost_log'
)

plt.title("BoxPlot: Impact of Smoking on Medical Cost")
plt.xlabel("Smoker Status")
plt.ylabel("Annual Medical Cost ($)")
plt.annotate(
    "Smokers show\nhigher median & variability",
    xy=(1, medical_df['annual_medical_cost_log'].median()),
    xytext=(0.5, medical_df['annual_medical_cost_log'].quantile(0.9)),
    arrowprops=dict(arrowstyle="->")
)
plt.tight_layout()

# Save image
plt.savefig(
    "smoking_vs_medical_cost.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Plot 3: Risk Score vs Claims Count
plt.figure()
sns.scatterplot(
    data=medical_df,
    x='risk_score',
    y='claims_count',
    alpha=0.4
)

plt.title("ScatterPlot: Risk Scores vs Claims Counts")
plt.xlabel("Risk Score")
plt.ylabel("Number of Claims")
plt.annotate(
    "Positive relationship\nbetween risk & utilization",
    xy=(0.6, 4),
    xytext=(0.2, 8),
    arrowprops=dict(arrowstyle="->")
)
plt.tight_layout()

# Save image
plt.savefig(
    "risk_scores_claim_counts.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Plot 4: Income vs Annual Medical Cost
medical_df['income_k'] = medical_df['income'] / 1000

plt.figure()

sns.regplot(
    data=medical_df,
    x='income_k',
    y='annual_medical_cost_log',
    scatter_kws={'alpha': 0.3}
)

plt.title("RegPlot: Relationship Between Income and Annual Medical Cost")
plt.xlabel("Annual Income ($ thousands)")
plt.ylabel("Annual Medical Cost($)")

plt.annotate(
    "Weak but positive trend:\nHigher income associated with\nslightly higher utilization",
    xy=(medical_df['income'].quantile(0.7),
        medical_df['annual_medical_cost_log'].quantile(0.6)),
    xytext=(medical_df['income'].quantile(0.3),
            medical_df['annual_medical_cost_log'].quantile(0.85)),
    arrowprops=dict(arrowstyle="->")
)
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()

# Save image
plt.savefig(
    "income_medical_cost.png",
    dpi=300,
    bbox_inches='tight'
)


plt.show()
