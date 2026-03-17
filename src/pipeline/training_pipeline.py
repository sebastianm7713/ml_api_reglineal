import joblib
from src.data.load_data import load_data
from src.data.preprocess import prepare_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate

def run_training():
    df = load_data("data/ventas_publicidad.csv")
    X, y = prepare_data(df)
    model, X_test, y_test = train_model(X, y)
    mse, r2 = evaluate(model, X_test, y_test)
    print("MSE:", mse)
    print("R2:", r2)
    joblib.dump(model, "models/modelo_regresion.pkl")
    print("Modelo guardado correctamente")

if __name__ == "__main__":
    run_training()