import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration with the Airbnb icon
st.set_page_config(page_title='Employee Attrition Model Analysis ', page_icon="hr.png", layout="wide")

# Front Page Design
st.markdown("<h1 style='text-align: center; font-weight: bold; font-family: Comic Sans MS;'>Employee Attrition Model Analysis Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Hello Connections! 👋 Welcome to My Project Presentation 🙏</h3>", unsafe_allow_html=True)

selected_page = option_menu(
    menu_title='',
    options=["Home","DashBoard","Analysis Zone","Charts","About"],
    icons=["house","grid","bar-chart","trophy","info-circle"],
    default_index=1,
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "#4BD6FF","size":"cover", "width": "100"},
            "icon": {"color": "yellow", "font-size": "25px"},
            "nav-link": {"font-size": "15px", "text-align": "center", "margin": "-2px", "--hover-color": "#white"},
            "nav-link-selected": {"background-color": "indigo"}})

if selected_page == "About":
    st.header(" :Green[Project Conclusion]")
    tab1,tab2 = st.tabs(["Features","Connect with me on"])
    with tab1:
        st.header("This Streamlit application allows users to access and analyze data from dataset.", divider='rainbow')
        st.subheader("1.    Users can select specific criteria such as Age, Business Travel, Department, Job Role, Hike %, and Avg. Working hours to retrieve relevant data and analyze trends.")
        st.subheader("2.    Users can access slicers and filters to explore data. They can customize the filters based on their preferences.")
        st.subheader("3.    The analysis zone provides users with access to filters derived through Python scripting.")
        st.subheader("4.    They can explore advanced predicted values to gain deeper insights into dataset.")
    with tab2:
             # Create buttons to direct to different website
            linkedin_button = st.button("LinkedIn")
            if linkedin_button:
                st.write("[Redirect to LinkedIn Profile > (https://www.linkedin.com/in/santhosh-r-42220519b/)](https://www.linkedin.com/in/santhosh-r-42220519b/)")

            email_button = st.button("Email")
            if email_button:
                st.write("[Redirect to Gmail > santhoshsrajendran@gmail.com](santhoshsrajendran@gmail.com)")

            github_button = st.button("GitHub")
            if github_button:
                st.write("[Redirect to Github Profile > https://github.com/Santhosh-1703](https://github.com/Santhosh-1703)")

elif selected_page == "Home":
    tab1,tab2 = st.tabs(["Employee Attrition Model Analysis","  Applications and Libraries Used! "])
    with tab1:
        st.write("Employee Attrition Model Analysis using a Machine Learning helps users gather valuable insights about Employees, and performance. By combining this data with information, users can get a comprehensive view of their organisation and employee engagement. This approach enables data-driven decision-making and more effective content strategies.")
        
        if st.button("Click here to know about this Model"):
            col1, col2 = st.columns(2)
            with col1:
                giphy_url = "https://giphy.com/embed/3oEdv1vkhqxcynkB5C"
                giphy_iframe = f'<iframe src="{giphy_url}" width="600" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>'
                giphy_link = '<p><a href="https://giphy.com/gifs/hrcloud-hr-human-resources-cloud-3oEdv1vkhqxcynkB5C">via GIPHY</a></p>'

                st.markdown(giphy_iframe + giphy_link, unsafe_allow_html=True)
            with col2:
                st.header(':white[Application info]', divider='rainbow')
                st.subheader(":star: Employee Attrition Model Prediction Project involves to predict Attrition based on Employee performance")
                st.subheader(":star: To predict the Attrition, Classification Trained Model is used. ")
                st.subheader(":star: This project aims to construct a machine learning model and implement, it as a user-friendly online application in order to provide accurate predictions about the Attrition possibility in Organisation ")
                st.subheader(":star: Attrition are influenced by a wide variety of criteria, including Age, Performance rating, annual data, and the Manager survey data ")
                st.subheader(":star: The provision of employees with an expected attrition based on these criteria is one of the ways in which a predictive model may assist in the overcoming of these obstacles. ")
                
            
    with tab2:
                st.subheader("  :bulb: Python")
                st.subheader("  :bulb: Numpy")
                st.subheader("  :bulb: Pandas")
                st.subheader("  :bulb: Scikit-Learn")
                st.subheader("  :bulb: Streamlit")
                st.subheader("  :bulb: PowerBI")

elif selected_page == "DashBoard":
    power_bi_report_url = "https://app.powerbi.com/reportEmbed?reportId=1a7244b8-aea6-4eb6-91cf-1b75e7a2ddb3&autoAuth=true&ctid=00f9cda3-075e-44e5-aa0b-aba3add6539f"
    st.markdown(f'<iframe title="EA_Dashboard" width="1600" height="700" src="{power_bi_report_url}" frameborder="0" allowFullScreen="true"></iframe>',unsafe_allow_html=True)

elif selected_page == "Analysis Zone":
            data = pd.read_csv(r"EA_Raw_data.csv")
            df = pd.DataFrame(data)
            with st.form("form2"):
                    col1,col2=st.columns(2)
                    with col1:
                        # Load the pickled encoders
                        with open(r"EA_encoders.pkl", 'rb') as f:
                            encoders = pickle.load(f)

                        # Load the pickled model
                        with open(r"EA_classification_model.pkl", 'rb') as file:
                            model = pickle.load(file)

                        # Define a function to preprocess input features
                        def preprocess_input(Age, BusinessTravel, Department, DistanceFromHome, Education, EducationField, Gender, JobLevel, 
                                             JobRole, MaritalStatus, MonthlyIncome, NumCompaniesWorked,PercentSalaryHike,StockOptionLevel,
                                             TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, JobInvolvement, 
                                             PerformanceRating, EnvironmentSatisfaction,JobSatisfaction,WorkLifeBalance,Avg_Working_time):
                            # Encode categorical features
                            BusinessTravel_encoded = encoders['BusinessTravel'].transform([BusinessTravel])[0]
                            Department_encoded = encoders['Department'].transform([Department])[0]
                            EducationField_encoded = encoders['EducationField'].transform([EducationField])[0]
                            Gender_encoded = encoders['Gender'].transform([Gender])[0]
                            JobRole_encoded = encoders['JobRole'].transform([JobRole])[0]
                            MaritalStatus_encoded = encoders['MaritalStatus'].transform([MaritalStatus])[0]

                            # Create a dataframe with the preprocessed features
                            data = pd.DataFrame({
                                        'Age': [Age],
                                        'BusinessTravel': [BusinessTravel_encoded],
                                        'Department': [Department_encoded],
                                        'DistanceFromHome': [DistanceFromHome],
                                        'Education': [Education],
                                        'EducationField': [EducationField_encoded],
                                        'Gender': [Gender_encoded],
                                        'JobLevel': [JobLevel],
                                        'JobRole': [JobRole_encoded],
                                        'MaritalStatus': [MaritalStatus_encoded],
                                        'MonthlyIncome': [MonthlyIncome],
                                        'NumCompaniesWorked': [NumCompaniesWorked],
                                        'PercentSalaryHike': [PercentSalaryHike],
                                        'StockOptionLevel': [StockOptionLevel],
                                        'TotalWorkingYears': [TotalWorkingYears],
                                        'TrainingTimesLastYear': [TrainingTimesLastYear],
                                        'YearsAtCompany': [YearsAtCompany],
                                        'YearsSinceLastPromotion': [YearsSinceLastPromotion],
                                        'YearsWithCurrManager': [YearsWithCurrManager],
                                        'JobInvolvement': [JobInvolvement],
                                        'PerformanceRating': [PerformanceRating],
                                        'EnvironmentSatisfaction': [EnvironmentSatisfaction],
                                        'JobSatisfaction': [JobSatisfaction],
                                        'WorkLifeBalance': [WorkLifeBalance],
                                        'Avg_Working_time': [Avg_Working_time],
                                    })
                            return data

                        Age = st.slider("Select Age", min_value=18, max_value=60, value=18)
                        BusinessTravel = st.selectbox("Select Business Travel", df['BusinessTravel'].unique())
                        Department = st.selectbox("Select Department", df['Department'].unique())
                        DistanceFromHome = st.number_input("Enter Distance From Home", min_value=1, max_value=29, value=1)
                        Education = st.number_input("Enter Education", min_value=1, max_value=5, value=1)
                        EducationField = st.selectbox("Choose Education Field", df['EducationField'].unique())
                        Gender = st.selectbox("Choose Gender", df['Gender'].unique())
                        JobLevel = st.number_input("Enter Job Level", min_value=1, max_value=5, value=1)
                        JobRole = st.selectbox("Choose Job Role", df['JobRole'].unique())
                        MaritalStatus = st.selectbox("Choose Marital Status", df['MaritalStatus'].unique())
                        MonthlyIncome = st.number_input("Enter Income Range", min_value=10000, max_value=200000)
                        NumCompaniesWorked = st.number_input("Enter No. Companies Worked", min_value=1, max_value=9, value=1)
                        PercentSalaryHike = st.number_input("Enter Salary Hike%", min_value=11, max_value=25)
                    
                    with col2:
                        StockOptionLevel = st.number_input("Enter Stock Option Level", min_value=0, max_value=3)
                        TotalWorkingYears = st.number_input("Enter Total Working Years", min_value=0, max_value=40)
                        TrainingTimesLastYear = st.number_input("Enter Training Times LastYear", min_value=0, max_value=6)
                        YearsAtCompany = st.number_input("Enter Years @ Company", min_value=0, max_value=40)
                        YearsSinceLastPromotion = st.number_input("Enter Years Since Last Promotion", min_value=0, max_value=15)
                        YearsWithCurrManager = st.number_input("Enter Years With Curr Manager", min_value=0, max_value=17)
                        JobInvolvement = st.number_input("Enter Job Involvement", min_value=1, max_value=4)
                        PerformanceRating = st.number_input("Enter Performance Rating", min_value=3, max_value=4)
                        EnvironmentSatisfaction = st.number_input("Enter Environment Satisfaction", min_value=1, max_value=4)
                        JobSatisfaction = st.number_input("Enter Job Satisfaction", min_value=1, max_value=4)
                        WorkLifeBalance = st.number_input("Enter Work Life Balance", min_value=1, max_value=4)
                        Avg_Working_time = st.number_input("Enter Avg. Working Hours", min_value=5, max_value=11)
                        
                        # Submit Button for PREDICT RESALE PRICE
                        submit_button1 = st.form_submit_button(label="PREDICT ATTRITION STATUS")
                    
                        if submit_button1:
                                # Preprocess the input features
                                input_data = preprocess_input(Age, BusinessTravel, Department, DistanceFromHome, Education, EducationField, Gender, JobLevel, 
                                                JobRole, MaritalStatus, MonthlyIncome, NumCompaniesWorked,PercentSalaryHike,StockOptionLevel,
                                                TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, JobInvolvement, 
                                                PerformanceRating, EnvironmentSatisfaction,JobSatisfaction,WorkLifeBalance,Avg_Working_time)
                                
                                # Predict the resale price using the model
                                x = model.predict(input_data.values)
                                if x[0] == 1.0:
                                    st.write(f'## :Blue[Predicted Attrition Status:] :red[Yes]')
                                    
                                    
                                elif x[0] == 0.0:
                                    st.write(f'## :Blue[Predicted Attrition Status:] :green[No]')

elif selected_page == "Charts":
    col1,col2,col3=st.columns(3)
    with col2:
        df = pd.read_csv(r"EA_Raw_data.csv")
        # Create a pie chart
        st.subheader("Employee Attrition Rate")
        attrition_rate = df["Attrition"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["No", "Yes"]
        sizes = [attrition_rate["No"], attrition_rate["Yes"]]
        colors = ["#1d7874", "#AC1F29"]
        explode = (0, 0.1)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.2f%%", startangle=90)
        ax.axis("equal")
        # Display the pie chart
        st.pyplot(fig)

        rate_att = df.groupby(['MonthlyIncome', 'Attrition']).apply(lambda x: x['MonthlyIncome'].count()).reset_index(name='Counts')
        rate_att['MonthlyIncome'] = round(rate_att['MonthlyIncome'], -3)
        rate_att = rate_att.groupby(['MonthlyIncome', 'Attrition']).apply(lambda x: x['MonthlyIncome'].count()).reset_index(name='Counts')

        # Plotting with Seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=rate_att, x='MonthlyIncome', y='Counts', hue='Attrition', ax=ax)
        st.subheader("Monthly Income Based Attrition")
        plt.xlabel('Monthly Income')
        plt.ylabel('Counts')
        # Show the plot using Streamlit's 'pyplot' command
        st.pyplot(fig)

        st.subheader("")
        plt.figure(figsize=(10, 6))
        # Create a bar plot with hue
        ax = sns.countplot(x="Percent_Salary_Hike_Bin", hue="Attrition", data=df, palette="viridis")

        # Add value counts on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Set labels and title
        plt.xlabel("Percent Salary Hike Group")
        plt.ylabel("Count")
        st.subheader("Percent Salary Hike Group Attrition")

        # Show the plot
        plt.legend(title="Attrition")
        st.pyplot(plt)

    with col1:
        st.subheader("Distribution of Gender Attrition")
        # Create a bar plot with hue
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x="Gender", hue="Attrition", data=df)

        # Add value counts on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Set labels and title
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.title("Distribution of Gender Attrition")
        # Show the plot
        plt.legend(title="Attrition")
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        # Create a bar plot with hue
        ax = sns.countplot(x="Department", hue="Attrition", data=df, palette="plasma")

        # Add value counts on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Set labels and title
        plt.xlabel("Department")
        plt.ylabel("Count")
        st.subheader("Department wise Attrition")
        # Show the plot
        plt.legend(title="Attrition")
        # Display the plot in Streamlit
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))

        # Create a bar plot with hue
        ax = sns.countplot(x="EnvironmentSatisfaction", hue="Attrition", data=df, palette="plasma")

        # Add value counts on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Set labels and title
        plt.xlabel("Environment Satisfaction Group")
        plt.ylabel("Count")
        st.subheader("Environment Satisfaction Group Attrition")

        # Show the plot
        plt.legend(title="Attrition")  # Add legend with the hue column title
        st.pyplot(plt) 

    with col3:
        st.subheader("Employee Age and Attrition")
        plt.figure(figsize=(10, 6))
        sns.histplot(x="Age", hue="Attrition", data=df, kde=True, bins=30)
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.legend(title="Attrition", labels=["No", "Yes"])

        # Display the plot in Streamlit app
        st.pyplot(plt)

        st.subheader("Job Level Attrition")
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(x="JobLevel", hue="Attrition", data=df, palette="deep")
        # Add value counts on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.xlabel("Job Level")
        plt.ylabel("Count")
        plt.legend(title="Attrition", labels=["No", "Yes"])

        # Display the plot in Streamlit
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))

        # Create a bar plot with hue
        ax = sns.countplot(x="JobSatisfaction", hue="Attrition", data=df, palette="deep")

        # Add value counts on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Set labels and title
        plt.xlabel("Job Satisfaction Group")
        plt.ylabel("Count")
        st.subheader("Job Satisfaction Group Attrition")

        # Show the plot
        plt.legend(title="Attrition")
        st.pyplot(plt)

    
     
