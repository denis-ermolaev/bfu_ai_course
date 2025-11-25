import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def train_models(df):
    # 1. Трансформация целевой переменной
    df = df.copy()
    df["target"] = df["mental_wellness_index_0_100"].apply(lambda x: 0 if x < 15 else 1)

    X = df.drop(["mental_wellness_index_0_100", "target"], axis=1)
    y = df["target"]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================================
    # 2. ОБУЧЕНИЕ МОДЕЛЕЙ
    # ==========================================================

    # ----- KNN -----
    # Выбор гиперпараметра n_neighbors:
    #   - слишком маленькое k → сильное переобучение
    #   - слишком большое k → недообучение
    # Оптимальный безопасный баланс обычно 3–7 → используем k = 5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # ----- Decision Tree -----
    # Выбор гиперпараметров:
    # max_depth ограничивает глубину:
    #   - слишком большая → переобучение
    #   - слишком маленькая → недообучение
    # max_depth=4 — баланс
    #
    # min_samples_split=4 — предотвращает создание веток с 1–2 наблюдениями.
    dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=4,
        random_state=42
    )
    dt.fit(X_train, y_train)

    # ==========================================================
    # 3. МЕТРИКИ
    # ==========================================================

    for model, name in [(knn, "KNN"), (dt, "DT")]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name}: {acc:.4f}; {f1:.4f}")

    # ==========================================================
    # 4. ПЛОТ ДЕРЕВА
    # ==========================================================
    plt.figure(figsize=(15, 8))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=["0", "1"])
    plt.show()


df = pd.read_csv("data/correct_df3.csv")
train_models(df)
