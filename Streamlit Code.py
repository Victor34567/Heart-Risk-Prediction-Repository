import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


file_id = "1MMG-VyOMRrMRMcelEAnm9GCseOcgRsgD"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
df = pd.read_csv(url)

st.set_page_config(layout="wide")
if "random_person" not in st.session_state:
    st.session_state["random_person"] = None
    st.session_state["edited"] = False
    st.session_state["edit_mode"] = False
left, middle, right = st.columns([0.25, 0.6,0.15])

with left:
    st.header("Data Card")
    # either input individual data or choose a random person
    mode = st.radio(
        "Select Data Source",
        ["Random Person", "Enter your own Data"]
    )

    # Mode 1: Choose a random person from the dataset
        # Mode 1: Choose a random person from the dataset
    if mode == "Random Person":
        if st.button("Choose random person", use_container_width=True):
            rp = df.sample(1)
            # State & HadHeartAttack direkt nach dem Ziehen entfernen (falls vorhanden)
            for col in ["State", "HadHeartAttack"]:
                if col in rp.columns:
                    rp = rp.drop(columns=[col])
            st.session_state["random_person"] = rp
            st.session_state["edited"] = False
            st.session_state["edit_mode"] = False
            st.success("🟢 Person chosen")

        if st.session_state["random_person"] is not None:
            st.subheader("Health Data")

            temp_rp = st.session_state["random_person"]
            # NUR anzeigen, nichts mehr droppen – Spalten sind schon weg
            table_view = temp_rp.T

            edited_table = st.data_editor(
                table_view,
                height=450,
                use_container_width=True
            )

            if st.button("Apply changes & update model", use_container_width=True):
                st.session_state["random_person"] = edited_table.T.copy()
                st.success("🔄 Updated — running analysis now...")
                st.rerun()
        else:
            st.info("Please choose a person first.")

    
    # Mode 2: Enter individual health data
    else:
        st.subheader("Enter your health data")
    

        with st.form("manual_input_form"):
            # --- Basisdaten ---
            # state = st.text_input("State (optional, can stay empty)", value="")

            sex = st.selectbox("Sex", ["Female", "Male"])

            general_health = st.selectbox(
                "General Health",
                ["Excellent", "Very good", "Good", "Fair", "Poor"]
            )

            physical_health_days = st.number_input(
                "Physical Health Days (0–30 days with poor physical health, last 30 days)",
                min_value=0, max_value=30, step=1, value=0
            )

            mental_health_days = st.number_input(
                "Mental Health Days (0–30 days with poor mental health, last 30 days)",
                min_value=0, max_value=30, step=1, value=0
            )

            last_checkup = st.selectbox(
                "Last Checkup Time",
                [
                    "Within past year (anytime less than 12 months ago)",
                    "Within past 2 years (1 year but less than 2 years ago)",
                    "Within past 5 years (2 years but less than 5 years ago)",
                    "5 or more years ago"
                ]
            )

            physical_activities = st.selectbox(
                "Physical Activities (do you do regular physical activities?)",
                ["Yes", "No"]
            )

            sleep_hours = st.number_input(
                "Sleep Hours per night (e.g. 7.0)",
                min_value=0.0, max_value=24.0, step=0.5, value=7.0
            )

            removed_teeth = st.selectbox(
                "Removed Teeth",
                ["None of them", "1 to 5", "6 or more, but not all", "All"]
            )

            # --- Krankheiten ---
            had_heart_attack = st.selectbox("Had Heart Attack", ["Yes", "No"])
            had_angina = st.selectbox("Had Angina", ["Yes", "No"])
            had_stroke = st.selectbox("Had Stroke", ["Yes", "No"])
            had_asthma = st.selectbox("Had Asthma", ["Yes", "No"])
            had_skin_cancer = st.selectbox("Had Skin Cancer", ["Yes", "No"])
            had_copd = st.selectbox("Had COPD", ["Yes", "No"])
            had_depression = st.selectbox("Had Depressive Disorder", ["Yes", "No"])
            had_kidney = st.selectbox("Had Kidney Disease", ["Yes", "No"])
            had_arthritis = st.selectbox("Had Arthritis", ["Yes", "No"])

            had_diabetes = st.selectbox(
                "Had Diabetes",
                [
                    "Yes",
                    "Yes, but only during pregnancy (female)",
                    "No",
                    "No, pre-diabetes or borderline diabetes"
                ]
            )

            # --- Einschränkungen ---
            deaf = st.selectbox("Deaf or Hard of Hearing", ["Yes", "No"])
            blind = st.selectbox("Blind or Vision Difficulty", ["Yes", "No"])
            diff_conc = st.selectbox("Difficulty Concentrating", ["Yes", "No"])
            diff_walk = st.selectbox("Difficulty Walking", ["Yes", "No"])
            diff_dress = st.selectbox("Difficulty Dressing/Bathing", ["Yes", "No"])
            diff_errands = st.selectbox("Difficulty with Errands", ["Yes", "No"])

            # --- Lifestyle / Risikofaktoren ---
            smoker_status = st.selectbox(
                "Smoker Status",
                [
                    "Former smoker",
                    "Never smoked",
                    "Current smoker - now smokes every day",
                    "Current smoker - now smokes some days"
                ]
            )

            ecig_usage = st.selectbox(
                "E-Cigarette Usage",
                [
                    "Never used e-cigarettes in my entire life",
                    "Use them some days",
                    "Not at all (right now)",
                    "Use them every day"
                ]
            )

            chest_scan = st.selectbox("Chest Scan", ["Yes", "No"])

            race = st.selectbox(
                "Race/Ethnicity Category",
                [
                    "White only, Non-Hispanic",
                    "Black only, Non-Hispanic",
                    "Other race only, Non-Hispanic",
                    "Multiracial, Non-Hispanic",
                    "Hispanic"
                ]
            )

            age_category = st.number_input(
                "Age (years, e.g. 40)",
                min_value=18, max_value=120, step=1, value=40
            )

            height_m = st.number_input(
                "Height in meters (e.g. 1.70)",
                min_value=1.0, max_value=2.5, step=0.01, value=1.70
            )

            weight_kg = st.number_input(
                "Weight in kilograms (e.g. 75.0)",
                min_value=30.0, max_value=250.0, step=0.5, value=75.0
            )

            bmi = st.number_input(
                "BMI (e.g. 25.0)",
                min_value=10.0, max_value=60.0, step=0.1, value=25.0
            )

            alcohol = st.selectbox("Alcohol Drinkers", ["Yes", "No"])
            hiv = st.selectbox("HIV Testing", ["Yes", "No"])
            flu_vax = st.selectbox("Flu Vaccine last 12 months", ["Yes", "No"])
            pneumo_vax = st.selectbox("PneumoVax ever", ["Yes", "No"])

            tetanus = st.selectbox(
                "Tetanus last 10 years / Tdap",
                [
                    "Yes, received Tdap",
                    "Yes, received tetanus shot but not sure what type",
                    "Yes, received tetanus shot, but not Tdap",
                    "No, did not receive any tetanus shot in the past 10 years"
                ]
            )

            high_risk = st.selectbox("High Risk last year", ["Yes", "No"])

            covid_pos = st.selectbox(
                "Covid Positive",
                [
                    "Yes",
                    "No",
                    "Tested positive using home test without a health professional"
                ]
            )

            submitted = st.form_submit_button("Create person & run model")

        if submitted:
            manual_person = pd.DataFrame([{
                #"State": state,
                "Sex": sex,
                "GeneralHealth": general_health,
                "PhysicalHealthDays": physical_health_days,
                "MentalHealthDays": mental_health_days,
                "LastCheckupTime": last_checkup,
                "PhysicalActivities": physical_activities,
                "SleepHours": sleep_hours,
                "RemovedTeeth": removed_teeth,
                "HadHeartAttack": had_heart_attack,
                "HadAngina": had_angina,
                "HadStroke": had_stroke,
                "HadAsthma": had_asthma,
                "HadSkinCancer": had_skin_cancer,
                "HadCOPD": had_copd,
                "HadDepressiveDisorder": had_depression,
                "HadKidneyDisease": had_kidney,
                "HadArthritis": had_arthritis,
                "HadDiabetes": had_diabetes,
                "DeafOrHardOfHearing": deaf,
                "BlindOrVisionDifficulty": blind,
                "DifficultyConcentrating": diff_conc,
                "DifficultyWalking": diff_walk,
                "DifficultyDressingBathing": diff_dress,
                "DifficultyErrands": diff_errands,
                "SmokerStatus": smoker_status,
                "ECigaretteUsage": ecig_usage,
                "ChestScan": chest_scan,
                "RaceEthnicityCategory": race,
                "AgeCategory": age_category,
                "HeightInMeters": height_m,
                "WeightInKilograms": weight_kg,
                "BMI": bmi,
                "AlcoholDrinkers": alcohol,
                "HIVTesting": hiv,
                "FluVaxLast12": flu_vax,
                "PneumoVaxEver": pneumo_vax,
                "TetanusLast10Tdap": tetanus,
                "HighRiskLastYear": high_risk,
                "CovidPos": covid_pos
            }])

            st.session_state["random_person"] = manual_person
            st.session_state["edited"] = False
            st.session_state["edit_mode"] = False

            st.success("✅ Person created — running model with your inputs...")
            st.rerun()


with middle:
    st.header("Heart Disease Health Information")
    random_person = st.session_state["random_person"]

    if random_person is None:
        st.info("Please select or create a person on the left first.")
    else:
        # Features for the graph
        features = [
            "GeneralHealth", "AgeCategory", "PhysicalHealthDays",
            "LastCheckupTime", "PhysicalActivities", "SleepHours",
            "SmokerStatus", "ECigaretteUsage", "AlcoholDrinkers", "BMI"
        ]

        df_feat = df[features].copy()
        rp_feat = random_person[features].copy()

        # Mapping dictionaries
        mappers = {
            "GeneralHealth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
            "AgeCategory": {
                "Age 18 to 24": 21, "Age 25 to 29": 27, "Age 30 to 34": 32, "Age 35 to 39": 37,
                "Age 40 to 44": 42, "Age 45 to 49": 47, "Age 50 to 54": 52, "Age 55 to 59": 57,
                "Age 60 to 64": 62, "Age 65 to 69": 67, "Age 70 to 74": 72, "Age 75 to 79": 77,
                "Age 80 or older": 82
            },
            "LastCheckupTime": {
                "5 or more years ago": 0,
                "Within past 5 years (2 years but less than 5 years ago)": 1,
                "Within past 2 years (1 year but less than 2 years ago)": 2,
                "Within past year (anytime less than 12 months ago)": 3
            },
            "PhysicalActivities": {"Yes": 1, "No": 0},
            "SmokerStatus": {
                "Never smoked": 0,
                "Former smoker": 1,
                "Current smoker - now smokes some days": 2,
                "Current smoker - now smokes every day": 3
            },
            "ECigaretteUsage": {
                "Never used e-cigarettes in my entire life": 0,
                "Not at all (right now)": 1,
                "Use them some days": 2,
                "Use them every day": 3
            },
            "AlcoholDrinkers": {"No": 0, "Yes": 1}
        }

        # Apply mappings
        for col, mapping in mappers.items():
            df_feat[col] = df_feat[col].map(mapping)
            rp_feat[col] = rp_feat[col].map(mapping)

        # Ensure numeric for continuous vars
        df_feat["SleepHours"] = df_feat["SleepHours"].astype(float)
        rp_feat["SleepHours"] = rp_feat["SleepHours"].astype(float)
        df_feat["BMI"] = df_feat["BMI"].astype(float)
        rp_feat["BMI"] = rp_feat["BMI"].astype(float)

        # Mean & person values
        df_avg = df_feat.mean()
        rp_values = rp_feat.iloc[0]

        x = np.arange(len(features))
        avg_vals = df_avg.values
        rp_vals = rp_values.values

        higher_better = np.array([
            True,   # GeneralHealth: higher = better
            False,  # AgeCategory: higher = worse
            False,  # PhysicalHealthDays: higher = worse
            True,   # LastCheckupTime: higher = better
            True,   # PhysicalActivities: higher = better
            True,   # SleepHours: higher = better (bis zu einem Punkt)
            False,  # SmokerStatus: higher = schlechter
            False,  # ECigaretteUsage: höher = schlechter
            False,  # AlcoholDrinkers: No (0) besser als Yes (1)
            False   # BMI: höher = schlechter
        ])

        better_mask = np.where(higher_better, rp_vals > avg_vals, rp_vals < avg_vals)

        # Scaling 0–1
        df_min = df_feat[features].min().values
        df_max = df_feat[features].max().values
        avg_scaled = (avg_vals - df_min) / (df_max - df_min)
        rp_scaled = (rp_vals - df_min) / (df_max - df_min)

        # Plot showcase and markings if better/worse
        plt.figure(figsize=(10, 5))

        plt.plot(x, rp_scaled, marker='o', label='Your values')
        plt.plot(x, avg_scaled, marker='o', label='Average')

        # green = better, red = worse
        plt.fill_between(
            x, rp_scaled, avg_scaled,
            where=better_mask,
            alpha=0.3, interpolate=True, color='green'
        )
        plt.fill_between(
            x, rp_scaled, avg_scaled,
            where=~better_mask,
            alpha=0.3, interpolate=True, color='red'
        )

        plt.xticks(x, features, rotation=30, ha='right')
        plt.ylabel("Scaled values (0–1)")
        plt.title("Your profile vs. average")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

