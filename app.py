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
        Hello! I'm Dhruv, a graduate student at Columbia University pursuing MS in Computer Science with a focus on Machine Learning.
        I completed my Bachelor's in Computer Science and Engineering at SRM Institute of Science and Technology, where I developed a 
        passion for solving real-world problems using Machine Learning, Computer Vision, Natural Language Processing, and Generative AI.
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
            * ***Expected Graduation***: December 2025
            * ***GPA***: 4.0

    """)
    st.markdown(f"<h5 style='color:#50C878;'> - SRM Institute of Science and Technology, Chennai, India</h5>", unsafe_allow_html=True)
    st.markdown("""
        - B.Tech in Computer Science and Engineering
            * ***Track***: Big Data Analytics
            * September 2020 - June 2024
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
        <p><b><span style='color:#50C878;'>Programming Languages:</span></b> C, C++, Python, SQL</p>
        <p><b><span style='color:#50C878;'>Data Analysis and Visualization:</span></b> Numpy, Pandas, Matplotlib, Seaborn</p>
        <p><b><span style='color:#50C878;'>Machine Learning:</span></b> Scikit-Learn, TensorFlow, PyTorch, OpenCV, NLTK</p>
        <p><b><span style='color:#50C878;'>Web Development:</span></b> Streamlit, FastAPI</p>
        <p><b><span style='color:#50C878;'>Soft Skills:</span></b> Problem Solving, Collaboration, Time Management, Adaptability, Active Listening</p>
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

st.markdown(f"<h5 style='color:#50C878;'> - Samsung R&D Institute, Bangalore, India</h5>", unsafe_allow_html=True)
st.markdown("""
    - Software Developer Intern, May 2023 - July 2023
        * Collaborated with the On-Device AI Solutions team to develop and train a Recurrent Neural Network (RNN) model for predicting 
        user smartphone tasks from monthly smartphone usage data of 10 different apps
        * Reconstructed a graph-based approach to log user activity patterns in form of sequential adjacency matrices
        * Identified and communicated inefficiency and incompatibility of chosen approach to team, leading to valuable
        insights and process improvements
""")
st.markdown("""
    - Samsung PRISM Research Intern, July 2022 - February 2023
        * Partnered on Sensor based Mood Profiling system to detect emotion in real-time and developed two Android
        WearOS based apps using Java and Android studio for data collection and mood prediction
        * Delivered lightweight TFLite model of Multi Layer Perceptron with 93.75% accuracy with an architecture of optimum
        set of sensors (Accelerometer, Gyroscope, Heart Rate)
        * Presented and published the work at 2023 IEEE (CONECCT) and was honoured with Certificate of Excellence
""")


st.markdown(f"<h3 style='color:#9B59B6;'>Projects</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: 2px solid #00A9E0; margin-top: 0px; margin-bottom: 0px;">
    """, 
    unsafe_allow_html=True
)

options = st.selectbox("Choose Domain:", ['Data Analysis and Predictive Modeling', 'Computer Vision', 'Natural Language Processing', 'Web Development'])

if options == 'Data Analysis and Predictive Modeling':
    proj_1, proj_2 = st.columns([1,1])
    with proj_1:
        st.markdown(f"<h5 style='color:#50C878;'> - German Credit Risk Analysis and Classification Model</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Performed data preprocessing, exploratory data analysis (EDA) with visualizations, and feature engineering on 21 attributes 
            from Statlog (German Credit Data)
            * Identified 12 most impactful attributes and conducted hyperparameter tuning on multiple classification models
            * Developed a full-stack web application displaying an interactive dashboard for data analysis and deployed classification
            model based on SVM algorithm with 78% accuracy on Streamlit Cloud for users to make real-time predictions 
        """)
        st.link_button('Interactive Dashboard', 'https://german-credit-analysis-and-modelling-by-dhruv-limbani.streamlit.app/')

    with proj_2:
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
            * Teamed up on developing a CNN model to detect pneumonia from chest X-ray images, achieving an accuracy of 95.97%
            * Addressed class imbalance by utilizing a Deep Convolutional Generative Adversarial Network (DCGAN) to generate
            synthetic images for minority class
            * Outperformed a fine-tuned pre-trained VGG16 model by 2% accuracy with CNN model on benchmark dataset
        """)

    with proj_4:
        st.markdown(f"<h5 style='color:#50C878;'> - CNN based American Sign Language Alphabets Translation</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Collected and labeled 5200 ASL hand image data for training model using Python, OpenCV, and Mediapipe
            * Constructed a CNN architecture using TensorFlow to classify 26 ASL alphabets with an accuracy of 99.71%
            * Fine-tuned MobileNetV2 model to enhance performance of system and improved accuracy to 99.81%
        """)

if options == 'Natural Language Processing':
    proj_5, proj_6 = st.columns([1,1])
    with proj_5:
        st.markdown(f"<h5 style='color:#50C878;'> - Financial Sentiment Analysis and Categorization</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Carried out a thorough text analysis with NLP techniques such as tokenization, stopword removal, lemmatization, 
            and n-gram extraction to enhance model performance on financial datasets
            * Evaluated various models, including three traditional classifiers (Logistic Regression, SVM, Random Forest), Dense Neural Networks,
            Recurrent Neural Networks (LSTM, BiLSTM) and transformer architecture (BERT), to identify the best-performing approach
            * Achieved up to 90.5% accuracy on test data using Bidirectional LSTM for financial sentiment classification
        """)

    with proj_6:
        st.markdown(f"<h5 style='color:#50C878;'> - Recipe Recommendation System based on Ingredients</h5>", unsafe_allow_html=True)
        st.markdown("""
            * Trained a Named Entity Recognition (NER) model on TASTEset, consisting of 700 recipes with more than 13,000 entities, 
            to extract ingredients from raw text
            * Applied advanced NLP techniques to clean and preprocess a dataset of 6,000+ recipes, including stopword removal and 
            lemmatization, followed by NER based ingredients extraction and vectorization leveraging TF-IDF Vectorizer
            * Developed a recipe recommendation model based on ingredient, cuisine, and dietary preferences using cosine similarities 
            calculated across recipes under 10 different diets, 20 courses and over 50 cuisines
        """)

if options == 'Web Development':
    st.markdown(f"<h5 style='color:#50C878;'> - NoCodeML: An End-to-End Platform for Data Analysis and ML Model Building</h5>", unsafe_allow_html=True)
    st.markdown("""
        * Built a comprehensive platform using Python and Streamlit, enabling users to perform end-to-end data analysis
        and model building without writing code, tested on over 20 different datasets
        * Implemented features for data cleaning, transformation and exploratory data analysis with visualizations
        * Facilitated seamless model training/testing and downloading with features for data preparation (train-test split,
        normalization, encoding) reducing data preparation time by up to 50%
    """)
    st.link_button('Check Website', 'https://dhruv-limbani-nocodeml-app-ddipho.streamlit.app/')


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
