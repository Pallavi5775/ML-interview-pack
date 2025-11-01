🧠 How to Analyze Any Sample Data — Unified Process

1️⃣ Understand data type → numeric (continuous) or categorical (discrete)
2️⃣ Identify underlying distribution

Continuous + symmetric → Normal

Discrete + two outcomes → Binomial

Discrete + rare counts → Poisson
3️⃣ Visualize shape

sns.histplot, sns.kdeplot, sns.countplot
4️⃣ Perform appropriate test

Mean difference → t-test

Categorical independence → Chi-square
5️⃣ Interpret p-value

p < 0.05 → statistically significant
6️⃣ Conclude with business meaning

“There is a significant increase in purchase rate after campaign.”