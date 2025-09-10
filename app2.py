import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="RP Field Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .subject-input {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 RP Field Recommendation System</h1>
    <p>Get personalized field recommendations based on your academic performance</p>
</div>
""", unsafe_allow_html=True)

def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_json("dataset/rp_merged_dataset_cleaned_marks_to_80_where_was_1.json")
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the JSON file is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def prepare_data(df):
    """Prepare and clean the dataset for modeling"""
    if df is None:
        return None, None, None, None, None, None, None
    
    # Data cleaning (same as your original script)
    subject_columns = []
    for idx, row in df.iterrows():
        if isinstance(row['marks'], dict):
            subject_columns.extend(list(row['marks'].keys()))
        elif isinstance(row['marks'], str):
            try:
                marks_dict = json.loads(row['marks'])
                subject_columns.extend(list(marks_dict.keys()))
            except:
                continue

    subject_columns = list(set(subject_columns))
    
    # Create cleaned DataFrame
    df_clean = df[['examinationBoard', 'combination', 'department', 'field', 'yearStudy', 'average_score']].copy()
    
    # Extract marks into separate columns
    for subject in subject_columns:
        df_clean[subject] = np.nan
    
    # Fill in the marks data
    for idx, row in df.iterrows():
        marks_data = None
        if isinstance(row['marks'], dict):
            marks_data = row['marks']
        elif isinstance(row['marks'], str):
            try:
                marks_data = json.loads(row['marks'])
            except:
                continue
        
        if marks_data:
            for subject, score in marks_data.items():
                if subject in df_clean.columns and pd.notna(score):
                    df_clean.at[idx, subject] = float(score)
    
    # Handle 'Synthetic Course'
    df_clean['field'] = df_clean['field'].apply(
        lambda x: 'Not recommended to join RP' if x == 'Synthetic Course' else x
    )
    
    # Fill missing values
    subject_cols = [col for col in df_clean.columns if col not in 
                   ['examinationBoard', 'combination', 'department', 'field', 'yearStudy', 'average_score']]
    
    for col in subject_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    df_clean = df_clean.dropna(subset=['field'])
    
    # Remove fields with very few samples
    field_counts = df_clean['field'].value_counts()
    valid_fields = field_counts[field_counts >= 5].index
    df_clean = df_clean[df_clean['field'].isin(valid_fields)]
    
    return df_clean, subject_cols, subject_columns, field_counts, df_clean['examinationBoard'].unique(), df_clean['combination'].unique(), df_clean['department'].unique()

def analyze_combination_subjects(df):
    """Analyze the dataset to create a mapping of combinations to their actual subjects"""
    combination_subject_mapping = {}
    
    if df is None:
        return combination_subject_mapping
    
    # Get all subject columns
    subject_columns = []
    for idx, row in df.iterrows():
        if isinstance(row['marks'], dict):
            subject_columns.extend(list(row['marks'].keys()))
        elif isinstance(row['marks'], str):
            try:
                marks_dict = json.loads(row['marks'])
                subject_columns.extend(list(marks_dict.keys()))
            except:
                continue
    
    subject_columns = list(set(subject_columns))
    
    # For each unique combination, find subjects that are consistently taken
    unique_combinations = df[['examinationBoard', 'combination']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        board = row['examinationBoard']
        combination = row['combination']
        key = f"{board}_{combination}"
        
        # Filter students with this combination
        combo_students = df[(df['examinationBoard'] == board) & (df['combination'] == combination)]
        
        if len(combo_students) == 0:
            continue
            
        # Analyze which subjects are most commonly taken by students in this combination
        subject_scores = {}
        
        for _, student in combo_students.iterrows():
            marks_data = None
            if isinstance(student['marks'], dict):
                marks_data = student['marks']
            elif isinstance(student['marks'], str):
                try:
                    marks_data = json.loads(student['marks'])
                except:
                    continue
            
            if marks_data:
                for subject, score in marks_data.items():
                    if pd.notna(score) and score > 0:  # Valid score
                        if subject not in subject_scores:
                            subject_scores[subject] = []
                        subject_scores[subject].append(float(score))
        
        # Filter subjects that are taken by at least 30% of students in this combination
        min_students = max(1, len(combo_students) * 0.3)
        relevant_subjects = []
        
        for subject, scores in subject_scores.items():
            if len(scores) >= min_students:
                relevant_subjects.append(subject)
        
        combination_subject_mapping[key] = sorted(relevant_subjects)
    
    return combination_subject_mapping

def get_combination_subject_mapping(df):
    """Get the combination-subject mapping"""
    return analyze_combination_subjects(df)

def get_subjects_for_combination(df, board, combination, subject_mapping):
    """Get subjects that are actually taken in the selected combination"""
    key = f"{board}_{combination}"
    
    if key in subject_mapping:
        return subject_mapping[key]
    else:
        return []

def train_model(df_clean, subject_cols):
    """Train the recommendation model"""
    if df_clean is None or len(df_clean) < 10:
        return None, None, None, None
    
    # Encode categorical variables
    le_field = LabelEncoder()
    le_board = LabelEncoder()
    le_combination = LabelEncoder()
    le_department = LabelEncoder()
    
    df_model = df_clean.copy()
    df_model['field_encoded'] = le_field.fit_transform(df_model['field'])
    df_model['board_encoded'] = le_board.fit_transform(df_model['examinationBoard'])
    df_model['combination_encoded'] = le_combination.fit_transform(df_model['combination'])
    df_model['department_encoded'] = le_department.fit_transform(df_model['department'])
    
    # Prepare features
    feature_cols = ['board_encoded', 'combination_encoded', 'department_encoded', 'average_score'] + subject_cols
    feature_cols = [col for col in feature_cols if col in df_model.columns]
    
    X = df_model[feature_cols]
    y = df_model['field_encoded']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    return model, scaler, (le_field, le_board, le_combination, le_department), feature_cols

def get_combinations_for_board(df, board):
    """Get available combinations for a specific examination board"""
    if df is None:
        return []
    return sorted(df[df['examinationBoard'] == board]['combination'].unique().tolist())

def make_prediction(model, scaler, encoders, feature_cols, df_clean, subject_cols, 
                   examination_board, combination, subject_scores, department=None):
    """Make field recommendation prediction"""
    try:
        le_field, le_board, le_combination, le_department = encoders
        
        # Calculate average score
        avg_score = np.mean(list(subject_scores.values())) if subject_scores else 75
        
        # Create input data
        input_data = {
            'average_score': avg_score,
            'board_encoded': le_board.transform([examination_board])[0],
            'combination_encoded': le_combination.transform([combination])[0],
        }
        
        # Add department if provided
        if department and department in le_department.classes_:
            input_data['department_encoded'] = le_department.transform([department])[0]
        else:
            input_data['department_encoded'] = 0
        
        # Add subject scores
        for subject in subject_cols:
            if subject in subject_scores:
                input_data[subject] = subject_scores[subject]
            else:
                input_data[subject] = df_clean[subject].median()
        
        # Convert to DataFrame and ensure column order
        input_df = pd.DataFrame([input_data])
        
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = df_clean[col].median() if col in df_clean.columns else 0
        
        input_df = input_df[feature_cols]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get top 3 recommendations
        prob_df = pd.DataFrame({
            'field': le_field.classes_,
            'probability': probabilities
        }).sort_values('probability', ascending=False)
        
        recommendations = [(row['field'], row['probability']) 
                         for _, row in prob_df.head(3).iterrows()]
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return [("Error in prediction", 0.0)]

# Main application
def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.processed_data = None

    # Load data only once
    if not st.session_state.data_loaded:
        with st.spinner("Loading dataset..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
        
        if st.session_state.df is None:
            st.stop()
        
        # Prepare data
        with st.spinner("Preparing data and training model..."):
            processed = prepare_data(st.session_state.df)
            if processed[0] is not None:
                df_clean, subject_cols, subject_columns, field_counts, boards, combinations, departments = processed
                subject_mapping = get_combination_subject_mapping(st.session_state.df)
                model, scaler, encoders, feature_cols = train_model(df_clean, subject_cols)
                
                st.session_state.processed_data = {
                    'df_clean': df_clean,
                    'subject_cols': subject_cols,
                    'subject_columns': subject_columns,
                    'field_counts': field_counts,
                    'boards': boards,
                    'combinations': combinations,
                    'departments': departments,
                    'subject_mapping': subject_mapping,
                    'model': model,
                    'scaler': scaler,
                    'encoders': encoders,
                    'feature_cols': feature_cols
                }
            else:
                st.error("Failed to process the data.")
                st.stop()
    
    # Get processed data from session state
    data = st.session_state.processed_data
    if data is None or data['model'] is None:
        st.error("Failed to train the model. Please check your dataset.")
        st.stop()
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(st.session_state.df))
            st.metric("Available Fields", len(data['field_counts']))
        with col2:
            st.metric("Exam Boards", len(data['boards']))
            st.metric("Subjects", len(data['subject_columns']))
        
        # Show field distribution
        st.subheader("Field Distribution")
        fig_dist = px.bar(
            x=data['field_counts'].values[:10], 
            y=data['field_counts'].index[:10],
            orientation='h',
            title="Top 10 Fields by Enrollment"
        )
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Show combination-subject mapping in sidebar
        st.subheader("📋 Curriculum Structure")
        sample_mappings = list(data['subject_mapping'].items())[:3]  # Show first 3 as examples
        for key, subjects in sample_mappings:
            board, combination = key.split('_', 1)
            with st.expander(f"{board} - {combination}"):
                if subjects:
                    st.write("**Subjects:**")
                    for subject in subjects[:8]:  # Show first 8 subjects
                        st.write(f"• {subject}")
                    if len(subjects) > 8:
                        st.write(f"... and {len(subjects) - 8} more")
                else:
                    st.write("No subjects mapped")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Get Your Field Recommendation")
        
        # Step 1: Select examination board
        st.subheader("Step 1: Select Your Examination Board")
        selected_board = st.selectbox(
            "Choose your examination board:",
            data['boards'],
            key="board_select"
        )
        
        if selected_board:
            # Step 2: Select combination
            st.subheader("Step 2: Select Your Combination")
            available_combinations = get_combinations_for_board(data['df_clean'], selected_board)
            
            if available_combinations:
                selected_combination = st.selectbox(
                    "Choose your combination:",
                    available_combinations,
                    key="combination_select"
                )
                
                if selected_combination:
                    # Step 3: Enter subject marks
                    st.subheader("Step 3: Enter Your Subject Marks")
                    
                    available_subjects = get_subjects_for_combination(
                        data['df_clean'], selected_board, selected_combination, data['subject_mapping']
                    )
                    
                    if available_subjects:
                        # Show statistics about the combination
                        combination_stats = data['df_clean'][
                            (data['df_clean']['examinationBoard'] == selected_board) & 
                            (data['df_clean']['combination'] == selected_combination)
                        ]
                        
                        st.success(f"📚 {selected_combination} combination includes {len(available_subjects)} subjects")
                        st.info(f"👥 Based on curriculum analysis of {len(combination_stats)} students")
                        
                        # Show the subjects that will be displayed
                        with st.expander("📖 View all subjects in this combination"):
                            cols_preview = st.columns(3)
                            for i, subject in enumerate(available_subjects):
                                with cols_preview[i % 3]:
                                    st.write(f"• {subject}")
                        
                        subject_scores = {}
                        
                        # Show subjects in a more organized way
                        st.write("**Enter your marks for each subject:**")
                        
                        # Create columns for better layout - adjust based on number of subjects
                        num_cols = min(3, len(available_subjects)) if len(available_subjects) > 6 else 2
                        cols = st.columns(num_cols)
                        
                        for i, subject in enumerate(available_subjects):
                            with cols[i % num_cols]:
                                # Get average score for this subject in this combination for context
                                subject_key = f"{selected_board}_{selected_combination}"
                                combo_data = st.session_state.df[
                                    (st.session_state.df['examinationBoard'] == selected_board) & 
                                    (st.session_state.df['combination'] == selected_combination)
                                ]
                                
                                # Calculate average from actual marks data
                                avg_scores = []
                                for _, row in combo_data.iterrows():
                                    marks_data = None
                                    if isinstance(row['marks'], dict):
                                        marks_data = row['marks']
                                    elif isinstance(row['marks'], str):
                                        try:
                                            marks_data = json.loads(row['marks'])
                                        except:
                                            continue
                                    
                                    if marks_data and subject in marks_data:
                                        score = marks_data[subject]
                                        if pd.notna(score) and score > 0:
                                            avg_scores.append(float(score))
                                
                                if avg_scores:
                                    avg_score_for_subject = np.mean(avg_scores)
                                    help_text = f"Average score: {avg_score_for_subject:.1f} | Students who took this: {len(avg_scores)}"
                                else:
                                    help_text = "Enter your score for this subject"
                                
                                score = st.number_input(
                                    f"**{subject}**",
                                    min_value=0,
                                    max_value=100,
                                    value=75,
                                    step=1,
                                    key=f"subject_{subject}",
                                    help=help_text
                                )
                                subject_scores[subject] = score
                        
                        # Predict button
                        if st.button("🎯 Get Field Recommendations", type="primary"):
                            with st.spinner("Analyzing your academic profile..."):
                                recommendations = make_prediction(
                                    data['model'], data['scaler'], data['encoders'], data['feature_cols'], 
                                    data['df_clean'], data['subject_cols'], selected_board, 
                                    selected_combination, subject_scores
                                )
                                
                                st.session_state['recommendations'] = recommendations
                                st.session_state['avg_score'] = np.mean(list(subject_scores.values()))
                    else:
                        st.warning(f"⚠️ No subjects found for the {selected_combination} combination under {selected_board}.")
                        st.info("This might indicate:")
                        st.write("- Limited data for this specific combination")
                        st.write("- This combination might not be commonly offered")
                        st.write("- Please try selecting a different combination")
                        
                        # Show what combinations are available for reference
                        st.write("**Available combinations for this board:**")
                        for combo in available_combinations:
                            combo_count = len(data['df_clean'][
                                (data['df_clean']['examinationBoard'] == selected_board) & 
                                (data['df_clean']['combination'] == combo)
                            ])
                            st.write(f"• {combo} ({combo_count} students)")
            else:
                st.warning(f"No combinations available for {selected_board} examination board.")
    
    with col2:
        st.header("📈 Your Results")
        
        if 'recommendations' in st.session_state:
            recommendations = st.session_state['recommendations']
            avg_score = st.session_state.get('avg_score', 0)
            
            # Display average score
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Your Average Score</h4>
                <h2>{avg_score:.1f}/100</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations
            st.subheader("🎯 Top Field Recommendations")
            
            for i, (field, confidence) in enumerate(recommendations, 1):
                if field != "Error in prediction":
                    confidence_pct = confidence * 100
                    
                    # Color coding based on confidence
                    if confidence_pct >= 70:
                        color = "#28a745"  # Green
                    elif confidence_pct >= 50:
                        color = "#ffc107"  # Yellow
                    else:
                        color = "#dc3545"  # Red
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 1rem; 
                                border-radius: 8px; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0;">#{i} {field}</h4>
                        <p style="margin: 0; font-size: 1.2em;">
                            Confidence: {confidence_pct:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualization of recommendations
            if len(recommendations) > 1 and recommendations[0][0] != "Error in prediction":
                fields = [rec[0][:30] + "..." if len(rec[0]) > 30 else rec[0] for rec in recommendations]
                confidences = [rec[1] * 100 for rec in recommendations]
                
                fig = px.bar(
                    x=confidences,
                    y=fields,
                    orientation='h',
                    title="Recommendation Confidence Scores",
                    color=confidences,
                    color_continuous_scale="viridis"
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Complete the form above to get your personalized field recommendations!")
    
    # Additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("ℹ️ How it Works")
        st.write("""
        1. Select your examination board
        2. Choose your academic combination
        3. Enter your subject marks
        4. Get AI-powered field recommendations
        """)
    
    with col2:
        st.subheader("🎯 Accuracy")
        st.write("""
        Our recommendation system uses machine learning 
        trained on thousands of student records to provide 
        accurate field predictions based on academic performance.
        """)
    
    with col3:
        st.subheader("📞 Support")
        st.write("""
        Need help? Contact the academic advisory team 
        for personalized guidance and additional 
        information about our programs.
        """)

if __name__ == "__main__":
    main()