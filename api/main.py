from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load and train model
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_species(data: IrisInput):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(input_data)
    species = iris.target_names[prediction[0]]
    return {"prediction": species}
