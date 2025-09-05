import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("project_delays.csv")
df['delayed'] = (df['actual_duration_days'] > df['planned_duration_days'] + 3).astype(int)
features = ['labor_shortage','material_delay','weather_days','subcontractor_issues','cost_overrun_pct']
X = df[features]
y = df['delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
fi.plot(kind='barh')
plt.title("Feature Importances for Delay Risk")
plt.tight_layout()
plt.savefig("delay_feature_importances.png", dpi=200)
