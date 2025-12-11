import streamlit as st
def main_app():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    import joblib



    file_id = "1MMG-VyOMRrMRMcelEAnm9GCseOcgRsgD"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    data = pd.read_csv(url)
    feature_columns = [
        "AgeCategory",
        "ChestScan",
        "HadAngina",
        "GeneralHealth",
        "PhysicalHealthDays",
        "SmokerStatus",
        "ECigaretteUsage",
        "HadDiabetes",
        "BMI",
        "PhysicalActivities",
        "DifficultyWalking",
        "HadCOPD",
        "HadStroke",
        "SleepHours",
        "HadDepressiveDisorder",
        "AlcoholDrinkers",
        "LastCheckupTime"
    ]
    df = data[feature_columns]

    if "random_person" not in st.session_state:
        st.session_state["random_person"] = None
        st.session_state["edited"] = False
        st.session_state["edit_mode"] = False
    left, middle, right = st.columns([0.25, 0.6,0.15])

    with left:
        st.header("Health Data")
        # either input individual data or choose a random person
        mode = st.radio(
            "",
            ["Try Random Person", "Enter your own Data"]
        )

        # Mode 1: Choose a random person from the dataset
            # Mode 1: Choose a random person from the dataset
        if mode == "Try Random Person":
            if st.button("Choose random person", use_container_width=True):
                rp = df.sample(1)
                # State & HadHeartAttack direkt nach dem Ziehen entfernen (falls vorhanden)
                if "HadHeartAttack" in rp.columns:
                    rp = rp.drop(columns=[col])

                st.session_state["random_person"] = rp
                st.session_state["edited"] = False
                st.session_state["edit_mode"] = False
                st.success("🟢 Person chosen")

            rename_map = {
                "GeneralHealth": "General Health",
                "PhysicalHealthDays": "Physical Health (Days)",
                "LastCheckupTime": "Last Check-up",
                "PhysicalActivities": "Physically Active",
                "SleepHours": "Sleep (Hours)",
                "HadStroke": "Stroke (History)",
                "HadAngina": "Angina (History)",
                "HadCOPD": "COPD",
                "HadDepressiveDisorder": "Depressive Disorder (History)",
                "HadDiabetes": "Diabetes Status",
                "DifficultyWalking": "Difficulty Walking",
                "SmokerStatus": "Smoking Status",
                "ECigaretteUsage": "E-Cigarette Use",
                "ChestScan": "Chest Scan",
                "AlcoholDrinkers": "Alcohol Consumption",
                "AgeCategory": "Age",
                "BMI": "BMI",
            }
            
            

            if st.session_state["random_person"] is not None:
                st.subheader("Health Data")

                # Original-DF NICHT verändern
                temp_rp = st.session_state["random_person"]

                # Tabelle wie bisher (echte Spaltennamen)
                base_table = temp_rp.T

                # Kopie für Anzeige mit schönen Labels
                display_table = base_table.copy()
                display_table.index = display_table.index.to_series().replace(rename_map)

                edited_table = st.data_editor(
                    display_table,
                    height=450,
                    use_container_width=True
                )

                if st.button("Apply changes", use_container_width=True):
                    # Werte aus Editor holen
                    new_vals = edited_table.iloc[:, 0].copy()

                    # schöne Labels -> Original-Spaltennamen
                    inverse_map = {v: k for k, v in rename_map.items()}
                    new_vals.index = new_vals.index.to_series().replace(inverse_map)

                    # Werte zurückschreiben, aber im ORIGINAL-Datentyp
                    row_idx = st.session_state["random_person"].index[0]
                    for col in st.session_state["random_person"].columns:
                        orig_dtype = st.session_state["random_person"][col].dtype
                        val = new_vals[col]

                        # wenn Spalte ursprünglich numerisch war → wieder in Zahl casten
                        if pd.api.types.is_numeric_dtype(orig_dtype):
                            val = pd.to_numeric(val, errors="coerce")

                        st.session_state["random_person"].at[row_idx, col] = val

                    st.success("🔄 Updated...")
                    st.rerun()



            else:
                st.info("Please choose a person first.")



        
        # Mode 2: Enter individual health data
        else:
            st.subheader("Enter your health data")
        

            with st.form("manual_input_form"):
                # --- Basisdaten ---
                # state = st.text_input("State (optional, can stay empty)", value="")


                general_health = st.selectbox(
                    "General Health",
                    ["Excellent", "Very good", "Good", "Fair", "Poor"]
                )

                physical_health_days = st.number_input(
                    "Physical Health Days (0–30 days with poor physical health, last 30 days)",
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

                # --- Krankheiten ---
                had_angina = st.selectbox("Angina (History)", ["Yes", "No"])
                had_stroke = st.selectbox("Stroke (History)", ["Yes", "No"])
                had_copd = st.selectbox("COPD", ["Yes", "No"])
                had_depression = st.selectbox("Depressive Disorder (History)", ["Yes", "No"])
                had_diabetes = st.selectbox(
                    "Diabetes Status",
                    [
                        "Yes",
                        "Yes, but only during pregnancy (female)",
                        "No",
                        "No, pre-diabetes or borderline diabetes"
                    ]
                )

                # --- Einschränkungen ---
                diff_walk = st.selectbox("Difficulty Walking", ["Yes", "No"])
                diff_dress = st.selectbox("Difficulty Dressing/Bathing", ["Yes", "No"])

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

                chest_scan = st.selectbox("Had Chest Scan", ["Yes", "No"])

                age_category = st.selectbox(
                    "Age Category",
                    [
                        "Age 18 to 24",
                        "Age 25 to 29",
                        "Age 30 to 34",
                        "Age 35 to 39",
                        "Age 40 to 44",
                        "Age 45 to 49",
                        "Age 50 to 54",
                        "Age 55 to 59",
                        "Age 60 to 64",
                        "Age 65 to 69",
                        "Age 70 to 74",
                        "Age 75 to 79",
                        "Age 80 or older"
                    ]
                )

                bmi = st.number_input(
                    "BMI (e.g. 25.0)",
                    min_value=10.0, max_value=60.0, step=0.1, value=25.0
                )

                alcohol = st.selectbox("Alcohol Drinkers", ["Yes", "No"])
               
                submitted = st.form_submit_button("Risk Check")

            if submitted:
                manual_person = pd.DataFrame([{
                    #"State": state,
                    "GeneralHealth": general_health,
                    "PhysicalHealthDays": physical_health_days,
                    "LastCheckupTime": last_checkup,
                    "PhysicalActivities": physical_activities,
                    "SleepHours": sleep_hours,
                    "HadAngina": had_angina,
                    "HadStroke": had_stroke,
                    "HadCOPD": had_copd,
                    "HadDepressiveDisorder": had_depression,
                    "HadDiabetes": had_diabetes,
                    "DifficultyWalking": diff_walk,
                    "DifficultyDressingBathing": diff_dress,
                    "SmokerStatus": smoker_status,
                    "ECigaretteUsage": ecig_usage,
                    "ChestScan": chest_scan,
                    "AgeCategory": age_category,
                    "BMI": bmi,
                    "AlcoholDrinkers": alcohol
                }])

                import pandas as pd  # steht bei dir eh schon oben

                # Dtypes an df anpassen (wie bei der Edit-Tabelle)
                for col in manual_person.columns:
                    orig_dtype = df[col].dtype
                    if pd.api.types.is_numeric_dtype(orig_dtype):
                        manual_person[col] = pd.to_numeric(manual_person[col], errors="coerce")
                    else:
                        manual_person[col] = manual_person[col].astype(orig_dtype)

                st.session_state["random_person"] = manual_person.copy()
                st.session_state["edited"] = False
                st.session_state["edit_mode"] = False

                st.success("✅ Person created — running model with your inputs...")
                st.rerun()


    with right:

        st.header("Risk Check")

        rp = st.session_state["random_person"]

        if rp is None:
            msg = ""
            bg_color = "#0E0E0E"
            
        else:
            pipe_final = joblib.load("final_heart_model.pkl")
            drop_cols = [c for c in ["HadHeartAttack", "State"] if c in rp.columns]
            rp_model = rp.drop(columns=drop_cols)

            pred = pipe_final.predict(rp_model)[0]

            if pred == 1:
                msg = "High<br><br><br><br>Risk"
                bg_color = "#B02626"
                text_color = "#FFFFFF"

            else:
                msg = "Low<br><br><br><br><br>Risk"
                bg_color = "#0d751b"
                text_color = "#FFFFFF"

                # "#040E2BFF"
        text_color = "#FFFFFF"
        st.markdown(
            f"""
            <div style="
                margin-top: -20px;
                background-color: {bg_color};
                padding: 20px;
                border-radius: 12px;
                height: 435px;
                width: 90%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 30px;
                font-weight: bold;
                color: {text_color};
                text-align: center;
            ">
            {msg}
            </div>
            """,
            unsafe_allow_html=True
        )
        

    with middle:
        st.header("Strengths and Weaknesses")
        random_person = st.session_state["random_person"]

        if random_person is None:
            st.info("Select or create a person")
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

            from scipy.interpolate import PchipInterpolator

            
            # Smooth x-axis
            x_smooth = np.linspace(x.min(), x.max(), 300)

            # Smooth curves
            spl_you = PchipInterpolator(x, rp_scaled)
            spl_avg = PchipInterpolator(x, avg_scaled)

            rp_smooth = spl_you(x_smooth)
            avg_smooth = spl_avg(x_smooth)

            # Plot showcase and markings if better/worse
            plt.style.use("default")
            plt.figure(figsize=(10, 5))
            plt.plot(x_smooth, rp_smooth, label="Your values", color="#0098df")
            plt.plot(x_smooth, avg_smooth, label="Average", color="#fb4a4a")

            # saubere Segmentgrenzen ohne Lücke
            bounds = np.concatenate(([x[0] - 0.5],
                            (x[:-1] + x[1:]) / 2,
                            [x[-1] + 0.5]))

            for i in range(len(features)):
                seg = (x_smooth >= bounds[i]) & (x_smooth <= bounds[i+1])
                color = "#0a6917" if better_mask[i] else "#fb4a4a"
                plt.fill_between(x_smooth[seg], rp_smooth[seg], avg_smooth[seg],
                                alpha=0.5, color=color)

            
            ax = plt.gca()
            ax.set_yticks([])
            ax.set_yticklabels([])     
            plt.xticks(x, features, rotation=30, ha="right")
            plt.title("You vs. Average")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            
            
            
            
            # --------- AI Advice unter dem Plot ---------
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

            # Prediction für den Text (nochmal, unabhängig von der Box rechts)
            pipe_final = joblib.load("final_heart_model.pkl")
            drop_cols = [c for c in ["HadHeartAttack", "State"] if c in random_person.columns]
            random_person_df = random_person.drop(columns=drop_cols)
            random_person_pred = pipe_final.predict(random_person_df)[0]

            risk_status = "High risk" if random_person_pred == 1 else "Low risk"
            person_scaled_list = rp_scaled.tolist()
            average_scaled_list = avg_scaled.tolist()
            feature_list = features
            better_list = higher_better.tolist()

            direction_text = """
            GeneralHealth: higher = better
            AgeCategory: higher = worse
            PhysicalHealthDays: higher = worse
            LastCheckupTime: higher = better
            PhysicalActivities: higher = better
            SleepHours: higher = better
            SmokerStatus: higher = worse
            ECigaretteUsage: higher = worse
            AlcoholDrinkers: higher = worse
            BMI: higher = worse
            """

            prompt = f"""
            You are a simple, realistic health advisor, to append an Heart Disease Risk model.
            You MUST base all judgments only on the numeric lists and rules below.
            Do not use stereotypes or general knowledge beyond these values.

            Model prediction: {risk_status}

            Features in this order:
            {feature_list}
            Person's scaled values (same order as features):
            {person_scaled_list}
            Average scaled values (same order as features):
            {average_scaled_list}

            How to interpret higher values (each feature):
            {direction_text}

            TASK: Compare the person's medical values for assessings heart disease risk against the average values and give clear, medical advice for improvement.
            The response should be:

            1) Adressed to the person the values are of
            2) about 4-6 Sentences in english - structure them well dont make them too long (if needed exceed 4-6 setnences by spliting a long one in 2)
            3) If risk is low start with good things, if high start with bad - keep in mind the higher - lower logic 
            
            4) If a value is at the best level after the low/high logic dont suggest improvenements but tips to keep this as a strength
            5) If there are no clearly good areas, do not invent any. It is fine to say if most areas need attention. 
            6) Give clear, practical lifestyle suggestions where meaningful
            7) Keep the tone realistic and neutral, not overly optimistic.
            8) Do NOT use numbers in the response. Do NOT directly talk about scaling or data.
            """

            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[{"role": "user", "content": prompt}]
            )
            advice_text = response.choices[0].message.content

            risk_label = "🟢 You have Low Risk:" if random_person_pred == 0 else "⭕ You have High Risk:"
            st.subheader(risk_label)
            st.write(advice_text)   
    pass       
        
import streamlit as st
import os
st.set_page_config(layout="wide")

import streamlit as st

if "started" not in st.session_state:
    st.session_state["started"] = False

if not st.session_state["started"]:

    start_html = """
    <div style="
        width: 880px;
        height: 620px;
        margin: 40px auto;
        padding: 40px 42px;
        border-radius: 28px;
        background: linear-gradient(135deg, #020617, #0f172a 35%, #1d4ed8 100%);
        color: #f1f5f9;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        display: flex;
        box-shadow: none;
        flex-direction: column;
    ">

        <!-- H1 -->
        <h1 style="
            font-size: 48px;
            margin: 0;
            letter-spacing: 0.4px;
            font-weight: 700;
        ">
            HeartAware
        </h1>

        <!-- H2 (weiter unten + untergeordnet zu HeartAware) -->
        <h2 style="
            font-size: 26px;
            margin: 30px 0 26px 0;
            font-weight: 700;
            color: #dbeafe;
        ">
            Heart Attack Risk - powered by Machine Learning
        </h2>

        <div style="display: flex; flex-direction: column; gap: 26px; margin-top: 4px;">

            <!-- H3 Sections -->
            <div>
                <h3 style="font-size: 22px; margin: 0 0 6px 0; font-weight: 600;">
                    Instant insights
                </h3>
                <p style="font-size: 19px; margin: 0;">
                    Strong factors. Weak factors. Clear risk score.
                </p>
            </div>

            <div>
                <h3 style="font-size: 22px; margin: 0 0 6px 0; font-weight: 600;">
                    AI Guidance
                </h3>
                <p style="font-size: 19px; margin: 0;">
                    Improve your numbers. Build healthier habits.
                </p>
            </div>

            <div>
                <h3 style="font-size: 22px; margin: 0 0 6px 0; font-weight: 600;">
                    Your Data
                </h3>
                <p style="font-size: 19px; margin: 0;">
                    Enter your own stats - or test a random profile.
                </p>
            </div>

        </div>

        <!-- Footer line -->
        <div style="margin-top: auto; padding-top: 20px;">
            <p style="font-size: 26px; margin: 0; font-weight: 700;">
                Know your heart. Know your risks.
            </p>
        </div>

    </div>

    """

    st.html(start_html)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Button global größer stylen
    st.markdown("""
    <style>
    .big-start-btn button {
        font-size: 50px;
        font-weight: 600;
        padding-top: 0px;
        padding-bottom: 0px;
        border-radius: 999px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="big-start-btn">', unsafe_allow_html=True)
        start = st.button("Begin your heart assessment", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if start:
        st.session_state["started"] = True
        st.rerun()

else:
    main_app()

