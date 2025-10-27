📅 Day 1 – Statistics & Probability Refresher

🎯 Goal: Master key statistical concepts and tests for ML.

### ✅ Tasks
- [ ] Review mean, variance, skewness, kurtosis
- [ ] Study Normal, Binomial, Poisson distributions
- [ ] Perform t-test, chi-square test using scipy
- [ ] Visualize sample vs population distributions

### 📘 Deliverables
- [ ] Jupyter Notebook: Hypothesis Testing
- [ ] Summary: p-value interpretation & Type I/II error notes

### 💬 Interview Focus
- What does a p-value mean?
- How do you test if two features are independent?

📅 Day 2 – EDA & Feature Engineering
### ✅ Tasks
- [ ] Handle missing & outlier values
- [ ] Encode categorical features (one-hot, label, target)
- [ ] Scale numerical features
- [ ] Visualize correlation heatmap & boxplots

### 📘 Deliverables
- [ ] EDA notebook with 3 plots
- [ ] Summary: Top 5 EDA insights

### 💬 Interview Focus
- How do you handle outliers in a dataset?
- What are the key steps in an EDA workflow?

📅 Day 3 – Linear Regression Deep Dive
### ✅ Tasks
- [ ] Implement OLS manually and using sklearn
- [ ] Check regression assumptions
- [ ] Evaluate residuals, R², adjusted R²
- [ ] Interpret coefficients

### 📘 Deliverables
- [ ] Notebook: Regression analysis & diagnostics
- [ ] Residual plots & VIF table

### 💬 Interview Focus
- What assumptions does linear regression make?
- How do you detect multicollinearity?

📅 Day 4 – Regularization & Model Selection
### ✅ Tasks
- [ ] Study Ridge, Lasso, ElasticNet
- [ ] Run GridSearchCV for alpha tuning
- [ ] Visualize coefficient shrinkage

### 📘 Deliverables
- [ ] Notebook: Regularization comparison
- [ ] Chart: alpha vs coefficient magnitude

### 💬 Interview Focus
- How does Lasso differ from Ridge?
- What’s the bias-variance tradeoff?

📅 Day 5 – Project 1: Predict House Prices 🏠
### ✅ Tasks
- [ ] Dataset: House Prices (Kaggle)
- [ ] Perform EDA & baseline Linear Regression
- [ ] Compute RMSE and explain metrics

### 📘 Deliverables
- [ ] Notebook: end-to-end regression model
- [ ] README: dataset summary, baseline model, future improvements

### 💬 Interview Focus
- How did you select important features?
- What causes overfitting in regression models?

📅 Day 6 – Logistic Regression & Classification Metrics
### ✅ Tasks
- [ ] Train Logistic Regression model
- [ ] Compute confusion matrix, precision, recall, F1, ROC-AUC
- [ ] Tune threshold and visualize trade-off

### 📘 Deliverables
- [ ] Notebook: classification metrics visualization
- [ ] ROC vs PR curve chart

### 💬 Interview Focus
- Why can accuracy be misleading?
- What metric to use for imbalanced datasets?

📅 Day 7 – Support Vector Machine (SVM)
### ✅ Tasks
- [ ] Train SVM with linear & RBF kernels
- [ ] Tune C & gamma
- [ ] Visualize decision boundaries

### 📘 Deliverables
- [ ] Notebook: kernel comparison
- [ ] Plot: margin & support vectors

### 💬 Interview Focus
- How does the kernel trick work?
- When would you prefer SVM over Logistic Regression?

📅 Day 8 – Naive Bayes & Decision Tree
### ✅ Tasks
- [ ] Train Naive Bayes on text data
- [ ] Train Decision Tree on tabular data
- [ ] Compare interpretability & performance
- [ ] Visualize feature importance

### 📘 Deliverables
- [ ] Notebook: NB vs Tree comparison
- [ ] Table: Accuracy, Precision, Recall

### 💬 Interview Focus
- Why does Naive Bayes perform well on text data?
- What is entropy and information gain?

📅 Day 9 – Project 2: Spam Detection 📧
### ✅ Tasks
- [ ] Preprocess text (TF-IDF, tokenization)
- [ ] Train Naive Bayes & Decision Tree classifiers
- [ ] Evaluate F1, ROC, PR curve
- [ ] Perform error analysis

### 📘 Deliverables
- [ ] Notebook: spam classifier pipeline
- [ ] README: model comparison & key observations

### 💬 Interview Focus
- How do you balance precision and recall?
- How do you handle misclassified spam?

📅 Day 10 – Deep Learning Fundamentals
### ✅ Tasks
- [ ] Study neural network structure, activations, backprop
- [ ] Learn optimizers: SGD, Adam
- [ ] Read about dropout & batch normalization

### 📘 Deliverables
- [ ] One-page Deep Learning Cheat Sheet

### 💬 Interview Focus
- Explain the vanishing gradient problem.
- Why is ReLU preferred over sigmoid?

📅 Day 11 – Project 3: MNIST Classifier (ANN + CNN) 🧩
### ✅ Tasks
- [ ] Build ANN (Keras or PyTorch)
- [ ] Add CNN layers and compare performance
- [ ] Visualize training curves, confusion matrix

### 📘 Deliverables
- [ ] Notebook: MNIST CNN model
- [ ] Chart: training vs validation accuracy

### 💬 Interview Focus
- Why does CNN perform better than ANN for images?
- How do you prevent overfitting in deep networks?

📅 Day 12 – Big Data with PySpark (Concepts)
### ✅ Tasks
- [ ] Learn RDD vs DataFrame API
- [ ] Practice joins, groupBy, and window functions
- [ ] Understand partitioning, shuffles, caching

### 📘 Deliverables
- [ ] PySpark notebook: ETL transformation pipeline
- [ ] Summary: performance tuning insights

### 💬 Interview Focus
- What’s lazy evaluation in Spark?
- How does partitioning affect job performance?

📅 Day 13 – Project 4: Analyze 1M-row Dataset (PySpark) 💾
### ✅ Tasks
- [ ] Use a large dataset (NYC Taxi / Retail)
- [ ] Perform heavy joins & aggregations
- [ ] Optimize using repartitioning & caching

### 📘 Deliverables
- [ ] PySpark job script (.py)
- [ ] Report: steps, performance metrics, visual summaries

### 💬 Interview Focus
- How would you optimize a Spark job?
- What challenges occur with large joins?

📅 Day 14 – Data Visualization & Storytelling 📊
### ✅ Tasks
- [ ] Select KPIs from previous projects
- [ ] Build Plotly/Dash dashboard with interactivity
- [ ] Add filters and visual storytelling captions

### 📘 Deliverables
- [ ] Dashboard notebook or app (HTML export)
- [ ] 3-slide summary: insights and recommendations

### 💬 Interview Focus
- How do you decide which visualization to use?
- What makes a good data story?

📅 Day 15 – Integration & Mock Interview 🎤
### ✅ Tasks
- [ ] Finalize one end-to-end project (EDA → Model → Deployment)
- [ ] Write detailed README (problem, data, model, metrics)
- [ ] Prepare 8 short answers for model explainability
- [ ] Record 3-min project walkthrough video

### 📘 Deliverables
- [ ] Final GitHub repo with notebooks + README
- [ ] Talking Points document (common Q&A)
- [ ] Deployment plan (Flask/FastAPI or streamlit demo)

### 💬 Interview Focus
- Explain your project end-to-end in under 3 minutes.
- What trade-offs did you make and why?
- How would you deploy and monitor this model?
