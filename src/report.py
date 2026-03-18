import json
import os

REPORT_PATH = "outputs/reports"
os.makedirs(REPORT_PATH, exist_ok=True)


# -------- MAIN REPORT --------
def generate_report(reg_results, cls_results):

    report = {
        "Regression": reg_results,
        "Classification": cls_results
    }

    # Save JSON
    with open(f"{REPORT_PATH}/report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Save readable TXT
    with open(f"{REPORT_PATH}/report.txt", "w") as f:

        f.write("MediScope ML Report\n")
        f.write("====================\n\n")

        f.write("Regression Models:\n")
        for model, res in reg_results.items():
            f.write(f"{model}: {res}\n")

        f.write("\nClassification Models:\n")
        for model, res in cls_results.items():
            f.write(f"{model}: {res}\n")

    # CLI output
    print("\n📄 FINAL REPORT")
    print("====================")

    print("\n📊 Regression:")
    for model, res in reg_results.items():
        print(model, res)

    print("\n📊 Classification:")
    for model, res in cls_results.items():
        print(model, res)

    print("\n💾 Reports saved in outputs/reports")


# -------- PATIENT INSIGHT (🔥 CORE FEATURE) --------
def generate_patient_insight(patient):

    print("\n🧍 Patient Insight")
    print("----------------------")

    risk = "HIGH" if patient["DiseaseRisk"] == 1 else "LOW"

    print(f"Age: {patient['age']}")
    print(f"BMI: {round(patient['bmi'], 2)}")
    print(f"Smoker: {patient['smoker']}")

    print(f"\n⚠️ Risk Level: {risk}")

    insights = []

    if patient["bmi"] > 30:
        insights.append("High BMI detected")

    if patient["smoker"] == "yes":
        insights.append("Smoking risk present")

    if patient["glucose_level"] > 140:
        insights.append("High glucose level")

    if patient["blood_pressure"] > 140:
        insights.append("High blood pressure")

    print("\n🔍 Key Findings:")
    for i in insights:
        print(f"• {i}")

    print("\n💡 Recommendation:")
    print("- Improve diet")
    print("- Increase physical activity")
    print("- Regular health checkups")