###
##Streamlit app - Lifestyle Buddy - one app to predict whether a person is high risk of having/towards chronic disease, 
##and a lifestyle guide to meal choices based on one's health risk and meal preference. 
##
##This app is hosted on streamlit.io 
##The code repository linked to streamlit.io is available on https://github.com/cy-chin/HealthHub-Chroniscope-demo
##


import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
from PIL import Image
import pickle



#### External resources loading ###
pkl_path = Path(__file__).parents[1] / "streamlit/trained_model.pkl"
with open(pkl_path, 'rb') as model_file:
    model = pickle.load(model_file)

hh_favicon_path = Path(__file__).parents[1] / "streamlit/h365.png"
hh_favicon = Image.open(hh_favicon_path)

hh_path = Path(__file__).parents[1] / "streamlit/h365.png"
hh_image = Image.open(hh_path)

food_path = Path(__file__).parents[1] / "streamlit/food_data_v3.csv"
df_food = pd.read_csv(food_path)

diet_path = Path(__file__).parents[1] / "streamlit/category_diet_v3.csv"
df_diet = pd.read_csv(diet_path)



#### Variable configuration ### 
predict_output = -1
x_weight_WTKG3 = 0
x_height_HTM4 = 0
x_age_AGEG5YR = 0 
x_edu_EDUCAG = 0 
x_skincancer_CHCSCNCR = 0
x_cancer_CHCOCNCR = 0
x_gender_SEX = 0
x_marital_MARITAL = 0
x_employment_EMPLOY1 = 0
x_blind_BLIND = 0
x_walk_DIFFWALK = 0
x_alch_day_ALCDAY5 = 0
x_hbp_RFHYPE5 = 0 
x_smoker_SMOKER3 = 0
x_alch_drink_DRNKANY5 = 0
x_alch_drink_DRNKWEK = 0
x_alch_binge_RFBING5 = 0
x_cholesterol_RFCHOL = 0 
x_fruit_consume_FRTLT1 = 0 
x_vege_consume_VEGLT1 = 0 
x_exercise_PA150R2= 0

x_user_input = []

edu_dict = {
    'Primary School': 0,
    'Secondary School' : 0,
    'Junior College' : 1,
    'ITE' : 1, 
    'PolyTechnique': 1, 
    'University- Undergraduate': 2,
    'University- Master/PhD': 2
}

yes_no_dict = {
    'No' : 0,
    'Yes' : 1 
}

gender_dict = {
    'Female' : 0, 
    'Male' : 1
}

marital_dict = {
    'Single' : 0,
    'Married' : 1
}

employment_dict = {
    'Employed' : 1, 
    'UnEmployed' : 0
}

smoker_status_dict = {
    'Never smoke': 0,
    'Former smoker' : 1,
    'Current smoker (smoke some days)' : 2,
    'Current smoker (smoke every day)' : 3
}

physical_activity_dict = {
    '0 minutes' : 0,
    '1 to 149 minutes' : 1, 
    'more than 150 minutes' : 2
}

meal_cuisine_list = ['chinese', 'indian','malay', 'others','snack','western']


#### Functions implementation ## 


# map the age to a age group per dictionary definition
def get_age_group(age):
    # if 18 <= age <= 24: 
    if age <= 24:   ## assuming all below 24 yr old will be "1", to avoid data entry error
        return 1
    elif 25 <= age <= 29:
        return 2
    elif 30 <= age <= 34:
        return 3
    elif 35 <= age <= 39:
        return 4
    elif 40 <= age <= 44:
        return 5
    elif 45 <= age <= 49:
        return 6
    elif 50 <= age <= 54:
        return 7
    elif 55 <= age <= 59:
        return 8
    elif 60 <= age <= 64:
        return 9
    elif 65 <= age <= 69:
        return 10
    elif 70 <= age <= 74:
        return 11
    elif 75 <= age <= 79:
        return 12
    elif age >= 80:
        return 13


# calculate BMI using height(cm) and weight(kg)
def get_BMI(weight, height):
    height_m = height / 100
    return weight / (height_m**2)


# mean center adjustment (for cosine similarity) 
def mean_center_rows(data):
    # Create a copy of the data so we don't overwrite original
    data = data.copy()
    data['avg_rating'] = data.mean(axis=1)
    
    # Let's subtract the original values with calculated average
    for col in data.columns:
        data[col] = data[col] - data['avg_rating']
        
    # Drop the user_avg_rating column and any NaN remains will be filled "0"
    data = data.drop(columns=['avg_rating'])
    data = data.fillna(0)
    return data


# construct profile for the recommender's diet profile ID  
def user_profile_mapping(user_profile):

    profile_cat = ""

    if user_profile["_AGEG5YR"] in [1,2]:
        profile_cat = "18 - 29"
    elif user_profile["_AGEG5YR"] in [3,4,5,6]:
        profile_cat = "30 - 50"
    elif user_profile["_AGEG5YR"] in [7,8]:
        profile_cat = "51 - 59"
    else:
        profile_cat = "> 60"

    if user_profile["SEX"]  == 1:
        profile_cat += "Male"
    else:
        profile_cat += "Female"
            
    if user_profile["_PA150R2"]  == 0:
        profile_cat += "Low"
    elif user_profile["_PA150R2"] == 1:
        profile_cat += "Moderate"
    else:
        profile_cat += "Active"
    
    if user_profile["CD"]  == 0:
        profile_cat += "0"
    else:
        profile_cat += "1"
        
    return profile_cat


# Implement the Content-Based Filtering
def recommender_food(user_profile) : 
    profile_cat = user_profile_mapping(user_profile)
    profile = pd.DataFrame(df_diet[df_diet["food_name"]==profile_cat])
    merged_df = pd.concat([df_food, profile], axis = 0) 
    merged_df.loc[merged_df['food_name']==profile_cat, "cuisine"] = merged_df.loc[merged_df['food_name']==profile_cat, "cuisine"].fillna(user_profile['cuisine'])
    merged_df = merged_df[merged_df['cuisine']==user_profile['cuisine']]

    merged_df = merged_df.drop(columns = "cuisine")
    new_merged_df = merged_df.reset_index(inplace = False).drop(columns = "index")
    merged_df_name = new_merged_df["food_name"]
    merged_df_sc = new_merged_df.copy()
    merged_df_sc = merged_df_sc.set_index(new_merged_df.columns[0])
    merged_df_new_columns = merged_df_sc.columns
    sc = StandardScaler()
    merged_df_sc = sc.fit_transform(merged_df_sc)
    merged_df_sc = pd.DataFrame(merged_df_sc)
    merged_df_sc = pd.concat([merged_df_name, merged_df_sc], axis=1)
    merged_df_sc  = merged_df_sc.set_index(merged_df_sc.columns[0])
    merged_df_sc.columns = merged_df_new_columns
    merged_df_sc_mc = mean_center_rows(merged_df_sc)
    sim_matrix = cosine_similarity(merged_df_sc_mc)
    food_sim = pd.DataFrame(sim_matrix, columns=merged_df_name, index=merged_df_name)
    profile_sim_withother = pd.DataFrame(food_sim.loc[profile_cat]).drop(index = profile_cat).sort_values(by = profile_cat, ascending = False) 
    profile_sim_withother.reset_index(inplace = True)
    return  profile_sim_withother["food_name"].head(10).values


###### Streamlit app page layout and data logic -- start here ###### 
st.set_page_config(page_title='HealthHub - ChroniScope (POC)', 
                   page_icon=hh_favicon, 
                   layout='centered', 
                   initial_sidebar_state='collapsed',
                   menu_items= {
                       'Get Help':'http://localhost:8501',
                       'Report a bug':'http://localhost:8501',
                       'About':'http://localhost:8501'                                              

                   })

if "prediction_outcome" not in st.session_state:
    st.session_state.prediction_outcome = -1

st.write("Welcome, :blue[Conrius]!")

st.image(hh_image)

tab1, tab2 = st.tabs(["Questionnaire","Food Guide"])

with tab1:
    st.header("⏳Chroniscope⏳")
    st.subheader(" Chronic Disease Risk Assessment")

    with st.form("questionnaire_form"):

        columns_to_check = ['cpd_bronchitis', 'depression', 'arthritis', 'heart_attack', 'stroke', 'asthma', 'diabetes', 'kidney_disease', 'heart_disease']
        
        st.markdown("***Personal Particulars***")

        left, right = st.columns((1,1))
        x_age_AGEG5YR = get_age_group(left.number_input("Age", min_value = 18, max_value=99, value=31))
        x_gender_SEX = gender_dict[right.selectbox("Gender",gender_dict.keys(), index=1)]
        x_weight_WTKG3 = left.number_input("Weight (in KG)", min_value = 30, max_value = 150, value=68 ) * 100
        x_height_HTM4 = right.number_input("Height (in CM)", min_value = 100, max_value = 220, value=175)
        x_marital_MARITAL = marital_dict[left.selectbox("Marital Status", marital_dict.keys(), index = 0)]
        x_edu_EDUCAG = edu_dict[right.selectbox("Highest Education Level", edu_dict.keys(), index = 5 )]
        x_employment_EMPLOY1 = employment_dict[left.selectbox("Employment Status", employment_dict.keys())]
        x_blind_BLIND = yes_no_dict[ right.selectbox("Difficulty seeing, even when wearing glasses?", yes_no_dict.keys())]
        
        st.divider()
        st.markdown("***Lifestyle Habit***")
        left, right = st.columns((1,1))
        x_fruit_consume_FRTLT1 = yes_no_dict[ left.selectbox("Consume fruits 1 or more times per day", yes_no_dict.keys())]
        x_vege_consume_VEGLT1 = yes_no_dict[ right.selectbox("Consume vegetables 1 or more times per day", yes_no_dict.keys())]
        x_alch_day_ALCDAY5 = left.number_input("How many days in a month did you have at least one alcoholic drink in a day?", min_value=0, max_value = 30)
        x_alch_drink_DRNKWEK = right.number_input("How many total number of alcoholic beverages consumed per week?", min_value=0, max_value = 50)
        x_alch_drink_DRNKANY5 = yes_no_dict[ left.selectbox("had at least one drink of alcohol in the past 30 days.?", yes_no_dict.keys())]
        x_alch_binge_RFBING5 = yes_no_dict[ right.selectbox("Binge drinker ( >4 drinks on one occasion)?", yes_no_dict.keys())]
        x_walk_DIFFWALK = yes_no_dict[ left.selectbox("Have difficulty walking or climbing stairs?", yes_no_dict.keys())]
        x_exercise_PA150R2 = physical_activity_dict[ right.selectbox("Physical exercise minute per week", physical_activity_dict.keys()) ]
        x_smoker_SMOKER3 = smoker_status_dict[ left.selectbox("Smoker Status",smoker_status_dict.keys()) ]

        st.divider()
        st.markdown("***Health Matter***")
        left, right = st.columns((1,1))
        x_hbp_RFHYPE5 = yes_no_dict[ left.selectbox("(Ever told) you had high blood pressure?", yes_no_dict.keys())]
        x_cholesterol_RFCHOL = yes_no_dict[ right.selectbox("(Ever told) you had high cholesterol?", yes_no_dict.keys())]
        x_skincancer_CHCSCNCR = yes_no_dict[ left.selectbox("(Ever told) you had skin cancer?", yes_no_dict.keys())]
        x_cancer_CHCOCNCR = yes_no_dict[ right.selectbox("(Ever told) you had any other types of cancer?", yes_no_dict.keys())]



        x_user_input.append( {
            'BMI' : get_BMI(x_weight_WTKG3, x_height_HTM4), 
            'age' : x_age_AGEG5YR,
            'education' : x_edu_EDUCAG, 
            'skin_cancer' : x_skincancer_CHCSCNCR,  
            'other_cancer' : x_cancer_CHCOCNCR,
            'sex' : x_gender_SEX,
            'martial' : x_marital_MARITAL,
            'employment_status' : x_employment_EMPLOY1,
            'blind' : x_blind_BLIND,
            'diff_walking' : x_walk_DIFFWALK,
            'occasion_drink_30days' : x_alch_day_ALCDAY5,
            'high_bp' : x_hbp_RFHYPE5,
            'smoker_status' : x_smoker_SMOKER3,
            'one_alc_per_day' : x_alch_drink_DRNKANY5,
            'ave_drink_week' : x_alch_drink_DRNKWEK,
            'binge_drink' : x_alch_binge_RFBING5,
            'high_cholesterol' : x_cholesterol_RFCHOL,
            'fruit' : x_fruit_consume_FRTLT1,
            'vegetable' : x_vege_consume_VEGLT1,
            'exercise_cat' : x_exercise_PA150R2, 
        })


        st.divider()
        submit1 = st.form_submit_button('Submit') 

        if submit1:

            input_df = pd.DataFrame(x_user_input)

            y_pred = model.predict(input_df)
            
            st.session_state.prediction_outcome = y_pred[0]

            if y_pred[0] == 1:
                text_high_risk = f"❕You are at <b>HIGH RISK</b> of developing chronic disease❕"
                html_high_risk = f"""<p style='background-color: rgb(250, 60, 60, 1); color: rgb(255,255,255,1); font-size:20px; 
                                        border-radius: 7px; padding-left: 12px; padding-top: 13px; padding-bottom: 13px; line-height: 25px;'>
                                    {text_high_risk}</style><BR></p>"""
                st.markdown(html_high_risk, unsafe_allow_html=True)
                # st.success(f"Prediction Output = You are at high risk of developing chronic disease", icon = "❗" )
            else:
                text_low_risk = f"⭐ You are at <b>LOW RISK</b> of developing chronic disease ⭐"
                html_low_risk = f"""<p style='background-color: rgb(0, 204, 102, 1); color: rgb(255,255,255,1); font-size:20px; 
                                        border-radius: 7px; padding-left: 12px; padding-top: 13px; padding-bottom: 13px; line-height: 25px;'>
                                    {text_low_risk}</style><BR></p>"""
                st.markdown(html_low_risk, unsafe_allow_html=True)
                # st.success(f"Prediction Output = Low risk of developing chronic disease", icon = "⭐")

with tab2:
    st.header("✨ Food Guide ✨")

    with st.form("food_recommender"):

        selected_cuisine = st.selectbox("Mouth-watering cuisine of choice?",meal_cuisine_list )
        submit2 = st.form_submit_button('Recommend me! ') 

        if submit2:
            
            if st.session_state.prediction_outcome == -1:
                st.write(f"Please complete and submit Questionnaire first. Come back here when is submitted")
            
            else:
                # st.write(f"[DEBUG] Health predicted outcome {st.session_state.prediction_outcome}")
            
                user_profile = {
                    "_AGEG5YR": x_age_AGEG5YR,
                    "SEX": x_gender_SEX,
                    "_PA150R2": x_exercise_PA150R2,
                    "CD": st.session_state.prediction_outcome,
                    "cuisine": selected_cuisine
                }
                          
                # st.write(f"[DEBUG] Using user profile: {user_profile}")
                # st.write(f"Recommender return this: {recommender.recommender_food(user_profile)}")
                
                recommended_list = recommender_food(user_profile)
                food_details = df_food.copy()
                food_details.set_index("food_name", inplace=True)
                food_details.drop(columns = "cuisine", inplace=True)
                st.dataframe(food_details.loc[recommended_list[0:5]], use_container_width=True)
                
                # st.write("[DEBUG] full list")
                # st.write(recommended_list[0:5])
                    



