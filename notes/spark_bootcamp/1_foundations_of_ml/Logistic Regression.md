Date: 24-09-2025
# Logistic Regression
## Why it matters
Useful for predicting a categorical label from continuous feature data. Classification achieved via probabilities. 
## Assumptions & Preconditions
- Predicting a class from a continuous feature vector
- Linear relationship between the features and the [log-odds](https://www.geeksforgeeks.org/machine-learning/role-of-log-odds-in-logistic-regression/) of the class prediction
- Absence of multicollinearity — for example, age and no. years work experience should be collapsed together via PCA/otherwise.
- Sufficient sample size for minority class (e.g. fraud detection has a low sample size)
## How it works (intuition)
You pass your linear combination of feature vectors $z=\beta_0+\beta_1x_1+\beta_2x_2+...+\beta_px_p$ which lies somewhere in $\mathbb{R}$ and map it to $[0,1]$ to get a probability via $\sigma(z)=\frac{1}{1+e^{-z}}$. Set some threshold; probabilities above are assigned class A, below get class B.
![Logistic Curve](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/500px-Logistic-curve.svg.png)
Assuming we're predicting $y\in\{0,1\}$, to find $\beta$, you look at your data $\{x_i,y_i\}_{i=1}^m$ and write down the likelihood function (just rewarding successes and disincentivizing failures as appropriate)
$$
L(\beta)=\prod_{i=1}^m\sigma(z_i)^{y_i}(1-\sigma(z_i))^{1-y_i}
$$
Optimising $L$ is equivalent to optimising the log-likelihood, 
$$
\mathcal{L}(\beta)=\sum_{i=1}^m [y_i\log(\sigma(z_i))+(1-y_i)\log(1-\sigma(z_i))]
$$
With some regularisation added to the loss function, this gets messy and the optimal parameters on the data are found numerically. [See docs](https://scikit-learn.org/stable/modules/linear_model.html#binary-case) and [my 2nd year notes](https://oliverkiranbrown.github.io/notes/third_year/math_stat_summary.pdf) on finding MLEs for details. 
## Minimal Recipe
```python
from sklearn.linear_model import LogisticRegression
y = df['target_label']
X = df.drop(['target_label'], axis=1)
# >>> scale the dataset, feature engineering... >>>
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression(c=300) # defaults to L2 regularisation
model.fit(X_train, y_train)

y_pred = model.predict(X_test) # predict class labels
y_prob = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int) # probablities for both classes and convert to a label
```

**Note**: multi-class labelling is easy to achieve. Lots of different implementations ([OvR](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier), [OvO](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html) or [softmaxing the multinomial](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)). Requires careful pre-processing. 
## Pitfalls 
**Remember**:
- You need a sufficient sample size — at least 20-30 observations per feature — otherwise you can easily overfit.
- By using buckets for classification, a 0.6 and a 0.99 will be given the same label despite wildly different levels of confidence. 
	- Consider more buckets (e.g. low risk, medium risk, high risk), using the underlying probabilities for maximum interpretability. 
## Metrics & Checks
When predicting a class label, you can be wrong in two ways:
1. **False positive** — predicted true when actually false
2. **False negative** — predicted false when actually true

These are measured as follows:
- **Accuracy**: What's the proportion of correctly classified predictions?
$$
\text{Accuracy}=\frac{\text{TP}+\text{TN}}{\text{TP}+\text{TN}+\text{FP}+\text{FN}}
$$
- **Recall**: What proportion of the predicted positives were genuine?
$$
\text{Recall}=\frac{\text{TP}}{\text{TP}+\text{FN}}
$$
- **Precision**: What fraction of the true positives are we capturing?
$$
\text{Precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}
$$
- **F1-Score**: harmonic mean of recall and precision — a single metric to balance the trade-off
$$
\text{F1-Score}=2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
$$
The relevance of these metrics is scenario dependent. For example...
- A new cancer screening can't afford a false negative so you'd require a high recall score for your model
- When detecting fraud, each investigation is costly so you'd prioritise high precision over recall to avoid wasted resource. 

## Tools & Workflows

- **Confusion Matrix**: how many TP, FP, TN, FN are there? You have a [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) to tell you and can also [visualise](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
mat = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(mat).plot;
print(classification_report(y_test, y_pred))
```

- [**RoC Curve**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html): it's useful to understand how your FP/FN rates change as you modify the threshold. 

The *True Positive Rate* is defined as

$$
\text{TPR}=\frac{\text{TP}}{\text{TP+FN}}=\frac{\text{True Positives}}{\text{All Actual Positives}}
$$
The *False Positive Rate* is defined as
$$
\text{FPR}=\frac{\text{FP}}{\text{FP+TN}}=\frac{\text{False Positives}}{\text{All Actual Negatives}}
$$
Plot these values against each other as the threshold increases to get an intuition for a good threshold value

```python
from sklearn.metrics import roc_curve, auc
y_probs = model.predict_proba(X_test)[:, 1] # probabilities for the first label only
fpr, tpr, thresholds = roc_curve(y_test, y_probs) # finds
roc_auc = auc(fpr, tpr)

# plot
plt.plot(fpr, tpr) # limit values to [0,1]^2
plt.xlabel('False Positives Rate') 
plt.ylabel('True Positives Rate')
```

You can then find an optimal index dependent on your situation ([fbeta score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) gives you maximum control). For example
```python
optimal_index = np.argmax(tpr * (1-fpr)) # balance between FP and TP
best_threshold = thresholds[optimal_index]
```

- **Dealing with class imbalances**: most financial transactions are not fraudulent, creating an imbalanced dataset. If we try and train a model on this dataset directly, the minority class will be predicted poorly. Two standard routes forwards:
	1) Move the data around — [undersampling](https://imbalanced-learn.org/stable/under_sampling.html), [oversampling](https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py), [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), [SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html). This can introduce unwanted bias so be careful. 
	2) Change the model — e.g. change class weights to force the model to care more about the minority class.