import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000

data = pd.DataFrame({
    "age": np.random.randint(18, 70, N),
    "bmi": np.random.uniform(18, 40, N),
    "children": np.random.randint(0, 5, N),
    "smoker": np.random.choice(["yes", "no"], N),
    "region": np.random.choice(["north", "south", "east", "west"], N),
    "blood_pressure": np.random.randint(80, 180, N),
    "glucose_level": np.random.randint(70, 200, N),
    "physical_activity": np.random.randint(1, 10, N)
})

# Disease Risk Logic (classification target)
data["DiseaseRisk"] = (
    (data["bmi"] > 30).astype(int) +
    (data["smoker"] == "yes").astype(int) +
    (data["glucose_level"] > 140).astype(int)
)

data["DiseaseRisk"] = (data["DiseaseRisk"] > 1).astype(int)

# Treatment Cost (regression target)
data["charges"] = (
    data["age"] * 200 +
    data["bmi"] * 300 +
    data["blood_pressure"] * 50 +
    data["glucose_level"] * 30 +
    (data["smoker"] == "yes").astype(int) * 10000 +
    np.random.normal(0, 1000, N)
)

data.to_csv("data/patients.csv", index=False)

print("✅ Dataset generated: data/patients.csv")