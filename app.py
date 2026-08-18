import streamlit as st
import pandas as pd
import joblib
import time


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AYAN Salary Prediction",
    page_icon="💼",
    layout="centered"
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load("salary_model.pkl")
    encoders = joblib.load("encoders.pkl")
    features = joblib.load("features.pkl")

    return model, encoders, features


model, encoders, features = load_model()



# =====================================================
# DARK / LIGHT MODE
# =====================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


st.session_state.dark_mode = st.toggle(
    "🌙 Dark Mode",
    value=st.session_state.dark_mode
)



# =====================================================
# THEME SETTINGS
# =====================================================

if st.session_state.dark_mode:

    BACKGROUND = "#0E1117"
    CARD = "#1f2330"
    TEXT = "#ffffff"
    SUBTEXT = "#aaaaaa"
    RESULT = "#16251b"

else:

    BACKGROUND = "#f5f7fa"
    CARD = "#ffffff"
    TEXT = "#111111"
    SUBTEXT = "#555555"
    RESULT = "#e9f7ea"



# =====================================================
# CSS DESIGN
# =====================================================

st.markdown(
f"""

<style>


section[data-testid="stSidebar"]{{
    display:none;
}}



[data-testid="stAppViewContainer"]{{

    background:{BACKGROUND};

}}



.block-container{{

    max-width:900px;
    padding-top:40px;
    margin:auto;

}}



.title{{

    text-align:center;
    font-size:45px;
    font-weight:900;
    color:#4da6ff;

}}



.subtitle{{

    text-align:center;
    font-size:20px;
    color:{SUBTEXT};
    margin-bottom:30px;

}}



.card{{

    background:{CARD};
    color:{TEXT};

    padding:35px;

    border-radius:25px;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.35);

}}



.result-card{{

    background:{RESULT};

    padding:30px;

    border-radius:25px;

    margin-top:30px;

}}



.stButton button{{

    width:100%;

    height:55px;

    border-radius:15px;

    font-size:20px;

    font-weight:bold;

    background:#0066ff;

    color:white;

}}



.stButton button:hover{{

    background:#004ecc;

}}



</style>

""",
unsafe_allow_html=True
)



# =====================================================
# HEADER
# =====================================================

st.markdown(
"""

<div class="title">

💼 AYAN Technologies

</div>


<div class="subtitle">

AI Based Employee Salary Prediction System

</div>

""",
unsafe_allow_html=True
)



# =====================================================
# INPUT CARD
# =====================================================

st.markdown(
"<div class='card'>",
unsafe_allow_html=True
)


st.subheader("👤 Candidate Information")



# -----------------------------
# AGE
# -----------------------------

age = st.slider(

    "Age",

    min_value=18,

    max_value=70,

    value=25

)



# -----------------------------
# EXPERIENCE DYNAMIC
# -----------------------------


max_experience = max(0, age - 18)


experience = st.slider(

    "Years of Experience",

    min_value=0,

    max_value=max_experience,

    value=min(3,max_experience)

)



# -----------------------------
# VALIDATION
# -----------------------------


if experience > age - 18:

    st.error(
        "Invalid profile: Experience cannot exceed working years."
    )



col1,col2 = st.columns(
    2,
    gap="large"
)



with col1:


    gender = st.selectbox(

        "Gender",

        [
            "Male",
            "Female"
        ]

    )



with col2:


    education = st.selectbox(

        "Education Level",

        [
            "Bachelor's",
            "Master's",
            "PhD"
        ]

    )




job_title = st.selectbox(

    "Job Title",

    [

        "Software Engineer",
        "Software Developer",
        "Data Analyst",
        "Data Scientist",
        "Machine Learning Engineer",
        "Project Manager",
        "Product Manager",
        "Business Analyst",
        "HR Manager",
        "Financial Analyst",
        "Marketing Manager",
        "Sales Manager",
        "Operations Manager",
        "CEO"

    ]

)



st.write("")



predict = st.button(

    "🚀 Predict Salary"

)



st.markdown(

"</div>",

unsafe_allow_html=True

)




# =====================================================
# PREDICTION
# =====================================================


if predict:


    if experience > age-18:

        st.stop()



    progress = st.progress(0)



    status = st.empty()



    for i in range(100):

        time.sleep(0.01)

        progress.progress(i+1)



    progress.empty()



    status.success(
        "Candidate profile analyzed successfully"
    )



    # -----------------------------
    # CREATE INPUT DATA
    # -----------------------------


    input_data = pd.DataFrame(

        {

            "Age":[age],

            "Gender":[gender],

            "Education Level":[education],

            "Years of Experience":[experience],

            "Job Title":[job_title]

        }

    )



    # -----------------------------
    # ENCODING
    # -----------------------------


    for col in [

        "Gender",

        "Education Level",

        "Job Title"

    ]:


        input_data[col] = encoders[col].transform(

            input_data[col]

        )



    # -----------------------------
    # FEATURE ORDER
    # -----------------------------


    input_data = input_data[features]



    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------


    salary = model.predict(

        input_data

    )[0]



    # =================================================
    # EXPERIENCE CATEGORY
    # =================================================


    if experience <= 1:

        category = "Fresher (0-1 Years)"


    elif experience <=5:

        category = "Junior Professional (2-5 Years)"


    elif experience <=10:

        category = "Mid-Level Professional (6-10 Years)"


    elif experience <=15:

        category = "Experienced Professional (11-15 Years)"


    elif experience <=20:

        category = "Senior Professional (16-20 Years)"


    else:

        category = "Executive Level (20+ Years)"





    # =================================================
    # HR RECOMMENDATION
    # =================================================


    if experience <=2:


        recommendation = (

            "Entry level candidate. "

            "Provide training and skill development."

        )


    elif experience <=5:


        recommendation = (

            "Junior professional. "

            "Offer competitive package with growth opportunities."

        )


    elif experience <=10:


        recommendation = (

            "Mid-level candidate. "

            "Consider incentives and promotion opportunities."

        )


    elif experience <=15:


        recommendation = (

            "Experienced professional. "

            "Recommend senior benefits."

        )


    else:


        recommendation = (

            "Executive profile. "

            "Consider leadership compensation package."

        )





    # =================================================
    # OUTPUT
    # =================================================


    st.balloons()



    st.markdown(

    "<div class='result-card'>",

    unsafe_allow_html=True

    )



    st.success(

        "🎯 Salary Prediction Completed"

    )



    st.markdown(

    f"""

    <h1 style='text-align:center;color:#00cc66;'>

    ₹ {salary:,.0f}

    </h1>

    """,

    unsafe_allow_html=True

    )



    st.info(

        f"📈 Experience Category: {category}"

    )



    st.warning(

        f"💡 HR Recommendation: {recommendation}"

    )



    st.markdown(

    "</div>",

    unsafe_allow_html=True

    )





# =====================================================
# FOOTER
# =====================================================


st.markdown(

"""

<hr>

<center>

<b>AYAN Technologies</b><br>

AI Employee Salary Prediction System<br>

Built with Streamlit + Machine Learning

</center>

""",

unsafe_allow_html=True

)