from typing import Dict, List


class PredictionService:
    def __init__(self):
        self.config = {
            "prediction_types": {
                "diabetes": {
                    "name": "Diabetes Risk Assessment",
                    "required_metrics": ["glucose_fasting", "glucose_post_meal", "bmi"],
                },
                "heart": {
                    "name": "Heart Health Screening",
                    "required_metrics": [
                        "heart_rate",
                        "bp_systolic",
                        "bp_diastolic",
                        "bmi",
                    ],
                },
                "obesity": {
                    "name": "Weight Status Analysis",
                    "required_metrics": ["bmi"],
                },
            },
            "metrics": {
                "glucose": {
                    "fasting": {
                        "ranges": {
                            "normal": {"min": 70, "max": 99},
                            "prediabetes": {"min": 100, "max": 125},
                            "diabetes": {"min": 126, "max": 200},
                        }
                    },
                    "post_meal": {
                        "ranges": {
                            "normal": {"min": 70, "max": 139},
                            "prediabetes": {"min": 140, "max": 199},
                            "diabetes": {"min": 200, "max": 300},
                        }
                    },
                },
                "vitals": {
                    "heart_rate": {
                        "ranges": {
                            "low": {"min": 40, "max": 59},
                            "normal": {"min": 60, "max": 100},
                            "high": {"min": 101, "max": 130},
                        }
                    },
                    "blood_pressure_systolic": {
                        "ranges": {
                            "normal": {"min": 90, "max": 120},
                            "elevated": {"min": 121, "max": 129},
                            "high": {"min": 130, "max": 180},
                        }
                    },
                    "blood_pressure_diastolic": {
                        "ranges": {
                            "normal": {"min": 60, "max": 80},
                            "high": {"min": 81, "max": 120},
                        }
                    },
                },
                "bmi": {
                    "ranges": {
                        "underweight": {"min": 0, "max": 18.4},
                        "normal": {"min": 18.5, "max": 24.9},
                        "overweight": {"min": 25, "max": 29.9},
                        "obese": {"min": 30, "max": 50},
                    }
                },
            },
        }

    def calculate_bmi(self, weight: float, height: float) -> float:
        """Calculate BMI from weight (kg) and height (m)"""
        try:
            bmi = weight / (height**2)
            return round(bmi, 1)
        except:
            return 0.0

    def get_bmi_category(self, bmi: float) -> str:
        """Get BMI category based on value"""
        ranges = self.config["metrics"]["bmi"]["ranges"]
        for category, range_values in ranges.items():
            if range_values["min"] <= bmi <= range_values["max"]:
                return category
        return "undefined"

    def get_health_insights(
        self, metrics: Dict[str, float], prediction_type: str
    ) -> List[str]:
        """Generate health insights based on metrics"""
        insights = []

        if prediction_type == "diabetes":
            # Glucose insights
            if metrics.get("glucose_fasting"):
                if metrics["glucose_fasting"] < 100:
                    insights.append("Your fasting glucose is in the normal range.")
                elif metrics["glucose_fasting"] < 126:
                    insights.append("Your fasting glucose indicates pre-diabetes risk.")
                else:
                    insights.append(
                        "Your fasting glucose is elevated, suggesting diabetes risk."
                    )

            if metrics.get("glucose_post_meal"):
                if metrics["glucose_post_meal"] < 140:
                    insights.append("Your post-meal glucose is normal.")
                elif metrics["glucose_post_meal"] < 200:
                    insights.append(
                        "Your post-meal glucose suggests pre-diabetes risk."
                    )
                else:
                    insights.append(
                        "Your post-meal glucose is high, indicating diabetes risk."
                    )

        elif prediction_type == "heart":
            # Blood pressure insights
            if metrics.get("bp_systolic") and metrics.get("bp_diastolic"):
                if metrics["bp_systolic"] < 120 and metrics["bp_diastolic"] < 80:
                    insights.append("Your blood pressure is in the normal range.")
                elif metrics["bp_systolic"] < 130 and metrics["bp_diastolic"] < 80:
                    insights.append("Your blood pressure is slightly elevated.")
                else:
                    insights.append(
                        "Your blood pressure is high. Consider lifestyle changes."
                    )

            # Heart rate insights
            if metrics.get("heart_rate"):
                if 60 <= metrics["heart_rate"] <= 100:
                    insights.append("Your resting heart rate is normal.")
                elif metrics["heart_rate"] < 60:
                    insights.append(
                        "Your heart rate is low. This might be normal for athletes."
                    )
                else:
                    insights.append("Your heart rate is elevated.")

        # BMI insights
        if metrics.get("bmi"):
            bmi_category = self.get_bmi_category(metrics["bmi"])
            if bmi_category == "normal":
                insights.append("Your BMI is in the healthy range.")
            elif bmi_category == "underweight":
                insights.append("Your BMI indicates you may be underweight.")
            elif bmi_category == "overweight":
                insights.append("Your BMI indicates you may be overweight.")
            elif bmi_category == "obese":
                insights.append(
                    "Your BMI indicates obesity. Consider consulting a healthcare provider."
                )

        return insights

    def get_recommendations(self) -> List[str]:
        return [
            "Maintain a balanced diet rich in whole foods",
            "Engage in regular physical activity (150 minutes per week)",
            "Get 7-9 hours of quality sleep each night",
            "Manage stress through relaxation techniques",
            "Stay hydrated and limit alcohol consumption",
        ]
