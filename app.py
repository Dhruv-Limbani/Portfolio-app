import streamlit as st
from PIL import Image
import base64
from io import BytesIO

from streamlit_lottie import st_lottie
import json

# Function to load local Lottie JSON file
def load_lottie_local(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Load your local Lottie JSON file
intro_animation = load_lottie_local("Intro_animation.json")
achievement_animation = load_lottie_local("Achievement_animation.json")
contact_animation = load_lottie_local("Contact_animation.json")
skills_animation = load_lottie_local("Skills_animation.json")
education_animation = load_lottie_local("Education_animation.json")

# Function to convert image to base64
def image_to_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Path to the uploaded image
profile_pic = image_to_base64("profile_pic.jpeg")
samsung_logo = image_to_base64("Samsung_Logo.png")

st.set_page_config(layout="wide")

# CSS for circular image
st.sidebar.markdown(
    """
    <style>
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 80%; 
        height: auto; 
        border-radius: 50%; 
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f'<img src="data:image/png;base64,{profile_pic}" class="img">',
        unsafe_allow_html=True
)

# Sidebar content with anchor links
st.sidebar.title("Dhruv Jitendra Limbani")
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#introduction' style='text-decoration:none; color: inherit;'>Introduction</a></h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#education' style='text-decoration:none; color: inherit;'>Education</a></h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#skills' style='text-decoration:none; color: inherit;'>Skills</a></h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#experience' style='text-decoration:none; color: inherit;'>Experience</a></h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#projects' style='text-decoration:none; color: inherit;'>Projects</a></h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#research' style='text-decoration:none; color: inherit;'>Research</a></h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#extracurricular-activities-and-achievements' style='text-decoration:none; color: inherit;'>Extracurricular Activities and Achievements</a></h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:#00A9E0;'><a href='#contact' style='text-decoration:none; color: inherit;'>Contact</a></h3>", unsafe_allow_html=True)


st.markdown(f"<h3 style='color:#9B59B6;'>Introduction</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)

intro, intro_anim = st.columns([1.75, 1])  # 1 parts for text, 1 part for animation
# Column 1: Introduction text
with intro:
    st.write(
        """
        Hello! I'm Dhruv, a Master's student in Computer Science at Columbia University, specializing in Machine Learning.

        I build and deploy intelligent systems to solve complex problems. My work spans Machine Learning, Natural Language Processing, and Computer Vision, with a focus on creating end-to-end solutions—from architecting data pipelines to training models and developing scalable applications.

        I am actively seeking full-time opportunities as a Data Scientist or Machine Learning Engineer starting December 2025. Let's connect!
        """
    )
    import streamlit as st

    with open("resume.pdf", "rb") as resume_file:
        st.download_button(
            label="Download Resume",
            data=resume_file,
            file_name="resume_Dhruv_Limbani.pdf",
            mime="application/pdf"
        )


# Column 2: Lottie Animation
with intro_anim:
    st_lottie(intro_animation, speed=1, width=400, height=300, key="intro_anim")


st.markdown(f"<h3 style='color:#9B59B6;'>Education</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)

edu, edu_anim = st.columns([1.75, 1])
with edu:
    st.markdown(f"<h5 style='color:#50C878;'> - Columbia University, New York City, US</h5>", unsafe_allow_html=True)
    st.markdown("""
        - MS in Computer Science
            * ***Track***: Machine Learning
            * Sep 2024 - Dec 2025
            * ***GPA***: 4.0

    """)
    st.markdown(f"<h5 style='color:#50C878;'> - SRM Institute of Science and Technology, Chennai, India</h5>", unsafe_allow_html=True)
    st.markdown("""
        - B.Tech in Computer Science and Engineering
            * ***Track***: Big Data Analytics
            * Sep 2020 - Jun 2024
            * ***GPA***: 9.79/10
""")

with edu_anim:
    st_lottie(education_animation, speed=1, width=400, height=300, key="edu_anim")

st.markdown(f"<h3 style='color:#9B59B6;'>Skills</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)
skills, skills_anim = st.columns([1.75, 1])
with skills:
    st.markdown(
        """
        <p><b><span style='color:#50C878;'>Programming:</span></b> Python, SQL, C++, C</p>
        <p><b><span style='color:#50C878;'>Machine Learning:</span></b> PyTorch, TensorFlow, Scikit-Learn, OpenCV, NLTK, LangChain, LangGraph, MCP</p>
        <p><b><span style='color:#50C878;'>Data Engineering & Analytics::</span></b> Pandas, NumPy, Apache Airflow, Matplotlib, Seaborn</p>
        <p><b><span style='color:#50C878;'>Databases:</span></b> Snowflake, PostgreSQL, ClickHouse, Pinecone, MySQL</p>
        <p><b><span style='color:#50C878;'>Additional Tools:</span></b> Git, Visual Studio Code, Jupyter, Google Colab, MySQL Workbench</p>
        <p><b><span style='color:#50C878;'>Soft Skills:</span></b> Problem Solving, Collaboration, Time Management, Adaptability, Active Listening, Proactive Communication</p>
        """,
        unsafe_allow_html=True
    )
with skills_anim:
    st_lottie(skills_animation, speed=1, width=400, height=300, key="skills_anim")


st.markdown(f"<h3 style='color:#9B59B6;'>Experience</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)

st.markdown(f"<h5 style='color:#50C878;'> - Chartmetric, New York, NY, US</h5>", unsafe_allow_html=True)
st.markdown("""
    - Data Analytics Intern, May 2025 - Aug 2025
        * Built analytics dashboards for music curation networks, enabling real-time trend insights, suspicious user detection
        (data scrapers) and product feature prioritization; used by C-suite for decision-making
        * Optimized database schema by normalizing 50+ tables into relational structures, reducing system errors, storage
        overhead, and improving query performance for LLM-driven text-to-SQL pipelines
        * Engineered a RAG proof-of-concept for text-to-SQL, reducing LLM token usage by 44% and operational costs by
        35% while increasing query accuracy
""")
st.markdown(f"<h5 style='color:#50C878;'> - Columbia University, Internet Real-Time Lab, New York, NY, US</h5>", unsafe_allow_html=True)
st.markdown("""
    - Research Assistant (NLP/LLMs), Jan 2025 - May 2025
        * Developed a curated and annotated dataset of 450+ privacy policies with clauses on AI training, cross-border data
        transfers, and advertiser data sharing to conduct research on automated privacy policy analysis using LLMs
        * Implemented benchmarking of multiple models (GPT-4o, DeepSeek, Mistral, BERT-family) under zero-shot, few-
        shot, and Retrieval-Augmented Generation (RAG) setups, achieving up to 99% recall in classification tasks
""")


st.markdown(f"<h5 style='color:#50C878;'> - Samsung R&D Institute India-Bangalore (SRI-B) Bangalore, IN</h5>", unsafe_allow_html=True)
st.markdown("""
    - Software Development Intern, May 2023 - Jul 2023
        * Collaborated with the On-Device AI Solutions team to develop an RNN-based model for predicting smartphone
        tasks, using monthly data from 10+ apps
        * Reconstructed a graph-based approach to log user activity patterns, utilizing previous 7 days’ graphs as sequential
        adjacency matrices to predict next day’s graph, testing its applicability to the current problem statement
        * Identified and communicated inefficiency and incompatibility of graph-based approach due to high memory usage
        despite an RMSE of 0.2 in task prediction, informing the team’s decision-making process
""")
st.markdown("""
    - Samsung PRISM - ML Research Intern (On-Device AI Team), Jul 2022 - Feb 2023
        * Partnered on a Sensor-based Mood Profiling system to detect emotions in real-time, integrating accelerometer,
        gyroscope, and heart rate data from 78 volunteers for accurate mood prediction
        * Developed two Android WearOS apps using Java and Android Studio for seamless data collection and mood tracking,
        with Firebase as the backend database to store and sync sensor data in real-time
        * Engineered a lightweight TFLite model based on a Multi-Layer Perceptron architecture, achieving 93.75% accuracy
        by optimizing sensor data inputs (Accelerometer, Gyroscope, Heart Rate) for mood prediction
        * Presented and published findings at the 2023 IEEE CONECCT, earning a Certificate of Excellence
""")


st.markdown(f"<h3 style='color:#9B59B6;'>Projects</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)

options = st.selectbox("Choose Domain:", ['LLMs and Agentic AI','Data Engineering, Analytics and Predictive Modeling', 'Computer Vision', 'Natural Language Processing', 'Software Development and Database'])

if options == 'Data Engineering, Analytics and Predictive Modeling':
    
    st.markdown(f"<h5 style='color:#50C878;'> - Renewable Energy Market and Risk Analysis</h5>", unsafe_allow_html=True)
    st.markdown("""
                * Designed a scalable ETL pipeline with Apache Airflow and PostgreSQL, automating data ingestion, transformation,
                and storage for energy and weather data across five Spanish cities
                * Built an LSTM (Long Short-Term Memory) model in PyTorch, achieving 0.02 RMSE in energy price forecasting
                * Computed 7-day VaR, CVaR, and volatility using SQL, integrating insights into a real-time Power BI dashboard
                for risk analysis
            """)
    st.link_button('Check Project', 'https://github.com/Dhruv-Limbani/Renewable-Energy-Price-Forecasting-and-Risk-Management-Dashboard')
    
    proj_1, proj_2 = st.columns([1,1])
    with proj_1:
        st.markdown(f"<h5 style='color:#50C878;'> - German Credit Risk Analysis and Classification Model</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Preprocessed data, performed EDA with visualizations and feature engineering on German Credit Data
            * Selected 12 key attributes using statistical tests and performed hyperparameter tuning on classification algorithms
            * Developed and deployed a full-stack web app with an interactive dashboard and SVM-based classification model
            (78% accuracy) on Streamlit Cloud for real-time predictions
        """)
        st.link_button('Check Project', 'https://github.com/Dhruv-Limbani/German-Credit-Data-Analysis-and-Modeling')

    with proj_2:
        st.markdown(f"<h5 style='color:#50C878;'> - Superstore Sales Data Analysis</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Built an ETL pipeline using Power Query to clean and analyze sales data across four regions, time, and products
            * Analyzed top-performing regions, low-profit areas, and top 10 best-selling products by volume and revenue, with
            insights on customer segments, shipping mode impact, and order trends
            * Created interactive visualizations to track sales, profit, and profit margin trends for data-driven decision-making
        """)
        st.link_button('Check Project', 'https://github.com/Dhruv-Limbani/Superstore-Sales-Data-Analysis')
    
    st.markdown(f"<h5 style='color:#50C878;'> - HR Data Analysis for Employee Churn Prediction</h5>", unsafe_allow_html=True)
    st.markdown("""
        * Performed data analysis and predictive modelling on HR data using Python, Pandas and Seaborn
        * Developed a machine learning model using Scikit-Learn to predict employee churn based on various factors such 
        as department, salary range, promotion in last 5 years, work accidents, time spent in company etc.
    """)

    
if options == 'Computer Vision':
    proj_3, proj_4 = st.columns([1,1])
    with proj_3:
        st.markdown(f"<h5 style='color:#50C878;'> - Pediatric Pneumonia Detection from Chest X-ray Images</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Teamed up on developing a CNN model to detect pneumonia from chest X-ray images, achieving 95.97% accuracy
            * Designed and trained a DCGAN to generate synthetic images for the minority class, addressing class imbalance
            * Outperformed a fine-tuned pre-trained VGG16 model by 2% accuracy, with a recall of 98% for Pneumonia class
            and 91% for Normal class on the benchmark dataset
        """)
        st.link_button('Check Website', 'https://github.com/Dhruv-Limbani/Pediatric-Pneumonia-Detection')

    with proj_4:
        st.markdown(f"<h5 style='color:#50C878;'> - CNN based American Sign Language Alphabets Translation</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Collected and labeled 5200 ASL hand image data for training model using Python, OpenCV, and Mediapipe
            * Constructed a CNN architecture using TensorFlow to classify 26 ASL alphabets with an accuracy of 99.71%
            * Fine-tuned MobileNetV2 model to enhance performance of system and improved accuracy to 99.81%
        """)
        st.link_button('Check Website', 'https://github.com/Dhruv-Limbani/Sign-Language-Translator')

if options == 'Natural Language Processing':
    proj_5, proj_6 = st.columns([1,1])
    with proj_5:
        st.markdown(f"<h5 style='color:#50C878;'> - Financial Sentiment Analysis and Categorization</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Carried out text preprocessing with NLP techniques (tokenization, stopword removal, lemmatization, n-gram extraction) to improve model performance on financial datasets
            * Evaluated various models (Logistic Regression, SVM, Random Forest, DNNs, LSTM, BiLSTM, BERT) to identify
            the best-performing approach
            * Achieved up to 90.5% accuracy on test data using Bidirectional LSTM for financial sentiment classification
        """)
        st.link_button('Check Website', 'https://github.com/Dhruv-Limbani/Financial-Sentiment-Analysis-using-NLP')

    with proj_6:
        st.markdown(f"<h5 style='color:#50C878;'> - Recipe Recommendation System based on Ingredients</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Trained a custom NER model on TASTEset (700 recipes, 13,000+ entities) to extract ingredients from raw text
            * Preprocessed 6,000+ recipes followed by NER-based ingredient extraction and TF-IDF vectorization
            * Developed a recipe recommendation model based on ingredient, cuisine, and dietary preferences using cosine simi-
            larity across 10 diets, 20 courses, and 50+ cuisines
        """)
        st.link_button('Check Website', 'https://github.com/Dhruv-Limbani/Indian-Food-Recommendation-based-on-Ingredients')

if options == 'Software Development and Database':

    st.markdown(f"<h5 style='color:#50C878;'> - SaaSy-Events - Event Management Portal</h5>", unsafe_allow_html=True)
    st.markdown("""
            * Collaborated with a team of six to develop a microservices-based event management application, with an Angular
            frontend deployed on an S3 bucket, a Python FastAPI backend, and a MySQL database hosted on AWS RDS
            * Designed and implemented a robust MySQL database to manage users, organizers, events, and tickets, optimizing
            query performance for seamless event operations
            * Developed a user management microservice with CRUD operations, Google OAuth 2.0 authentication, and JWTbased
            authorization, ensuring secure access control and integrated third-party APIs for ticket QR code generation
            """)
    st.link_button('Check Project', 'https://github.com/SaaSy-Pants')

    proj_7, proj_8 = st.columns([1,1])
    with proj_7:
        st.markdown(f"<h5 style='color:#50C878;'> - NoCodeML: Simplifying the Data Science Workflow</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Built a no-code platform enabling users to perform end-to-end data science workflows, tested on 20+ datasets
            * Implemented features for data cleaning, transformation and exploratory data analysis with visualizations
            * Streamlined model training/testing and data preparation (train-test split, normalization, encoding), reducing data
            preparation time by 50%
        """)
        st.link_button('Check Website', 'https://github.com/Dhruv-Limbani/NoCodeML')
    
    with proj_8:
        st.markdown(f"<h5 style='color:#50C878;'> - Online Banking System</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Designed an online banking system for banking operations like account management, fund transfers, credit/debit
            card management, and loan repayment
            * Optimized MySQL databases and queries for transaction history, account balances, and secure fund transfers
            * Built an interactive front-end, integrating real-time data from MySQL for an enhanced user experience
        """)
        st.link_button('Check Website', 'https://github.com/Dhruv-Limbani/Online-Banking-System')

if options == 'LLMs and Agentic AI':
    st.markdown(f"<h5 style='color:#50C878;'> - Interactive Data Analytics Assistant using LLMs</h5>", unsafe_allow_html=True)
    st.markdown("""
        * Developed an intelligent assistant powered by Google Gemini-2.5-Flash, capable of performing end-to-end data
        analytics through conversational queries, including statistical analysis, insights generation, and visualizations
        * Integrated multiple MCP-based tools (csv_analyzer, local_python_executor) to dynamically inspect datasets,
        run Python/NumPy/Pandas code, and generate visual outputs with Matplotlib and Seaborn
        * Implemented system prompts and memory for context retention, enabling accurate and reliable multi-turn analysis
    """)
    st.link_button('Check Website', 'https://github.com/Dhruv-Limbani/Data-Analyst-Agent')
st.markdown(f"<h3 style='color:#9B59B6;'>Research</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)
 
res_1, res_2 = st.columns([1,1])
with res_1:
    st.markdown(f"<h5 style='color:#50C878;'> WEARS: Wearable Emotion AI with Real-time Sensor data</h5>", unsafe_allow_html=True)
    st.markdown("""
        * Published in: 2023 IEEE International Conference on Electronics, Computing and Communication Technologies
        (CONECCT)
        * Developed a real-time smartwatch-based emotion prediction system, utilizing a self-created multilingual video
        dataset as visual stimuli for data collection.
        * Experimented with various machine-learning algorithms and achieved a 93.75% accuracy in binary classification of pleasant-unpleasant
        moods by analyzing smartwatch sensor data, including Heart Rate, Accelerometer, and Gyroscope inputs.
    """)
    st.link_button('View Publication', 'https://ieeexplore.ieee.org/document/10234730')

with res_2:
    st.markdown(f"<h5 style='color:#50C878;'> - Facial Recognition for Humanitarian Efforts: A Deep Learning based Solution for Missing Person Identification</h5>", unsafe_allow_html=True)
    st.markdown("""
        * Developed a novel face recognition method for missing persons identification, reducing computational complexity with 
        Principal Component Analysis (PCA) for dimensionality reduction.
        * Used VGGFace to extract face embeddings and trained a Dense Neural Network on 128-dimensional reduced embeddings, improving efficiency.
        * Achieved O(1) classification time (vs. traditional O(n)), significantly enhancing performance in large-scale databases.
        * Reached an accuracy of 98.75%, outperforming existing methods and demonstrating strong potential for real-world applications.
    """)
    st.link_button('View Publication', 'https://ieeexplore.ieee.org/document/10602230')


st.markdown(f"<h3 style='color:#9B59B6;'>Extracurricular Activities and Achievements</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)
achieve, achieve_anim = st.columns([1.75,1])
with achieve:
    st.markdown("""
        - Vice Domain Leader of Machine Learning team at Think Digital Student Club at SRM
        - Technical team member and intraclub project leader of Guvi Code Camp, SRM (Student Club)
        - 2nd runner up in BugOut (Debugging Competetion) at Datakon-2022, SRMIST
        - Class Topper Award at SRM Institute of Science and Technology
        - Performance Based Scholarship for 1st Rank in department at SRM Institute of Science and Technology
        - SHE scholarship for being in top 1% in Gujarat State Higher Secondary Board Examination
    """)
with achieve_anim:
    st_lottie(achievement_animation, speed=1, width=400, height=300, key="achieve_anim")


st.markdown(f"<h3 style='color:#9B59B6;'>Contact</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)
contact, contact_anim = st.columns([1.75,1])
with contact:
    st.markdown(
        """
        <p><b><span style='color:#50C878;'>Phone Number:</span></b> +1 (646) 281-9850</p>
        <p><b><span style='color:#50C878;'>Email:</span></b> <a href="mailto:djl2204@columbia.edu" style='text-decoration:none; color: inherit;'>djl2204@columbia.edu</a></p>
        """,
        unsafe_allow_html=True
    )
    st.link_button('GitHub', 'https://github.com/Dhruv-Limbani')
    st.link_button('Connect on LinkedIn', 'https://www.linkedin.com/in/dhruvlimbani/')

with contact_anim:
    st_lottie(contact_animation, speed=1, width=400, height=300, key="contact_anim")
