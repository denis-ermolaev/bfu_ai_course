import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

def classification_training(data: pd.DataFrame):
    # 1. Подготовка данных
    data["mental_wellness_index_0_100"] = data["mental_wellness_index_0_100"].map(lambda x: 0 if x < 15 else 1)

    if data["mental_wellness_index_0_100"].nunique() < 2:
        print("Ошибка: после преобразования в данных остался только один класс.")
        return

    # !!! ГЛАВНОЕ ИЗМЕНЕНИЕ: выбираем только числовые признаки для обучения !!!
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Удаляем целевую переменную из списка признаков
    if "mental_wellness_index_0_100" in numeric_features:
        numeric_features.remove("mental_wellness_index_0_100")s

    X = data[numeric_features]
    y = data["mental_wellness_index_0_100"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Обучение SVM с масштабированием
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, random_state=42))
    svm_pipeline.fit(X_train, y_train)
    svm_predictions = svm_pipeline.predict(X_test)
    print(f"SVM: {accuracy_score(y_test, svm_predictions)}; {f1_score(y_test, svm_predictions)}")

    # 3. Обучение Logistic Regression с масштабированием
    lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    lr_pipeline.fit(X_train, y_train)
    lr_predictions = lr_pipeline.predict(X_test)
    print(f"LR: {accuracy_score(y_test, lr_predictions)}; {f1_score(y_test, lr_predictions)}")



df = pd.read_csv("data/correct_df3.csv")

classification_training(df)