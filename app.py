import streamlit as st
import sklearn
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import google.generativeai as genai
import json
import re
genai.configure(api_key='AIzaSyAPu0L2EYVwsgD0AxzlUQRqkLfEqX_NYiw')
nltk.download('punkt')
nltk.download('stopwords')
model=genai.GenerativeModel('gemini-1.0-pro')



# Load a pre-trained transformer model for semantic similarity
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_data
def load_data():
    with open("facultydetails.json", encoding="utf-8") as f:
        return json.load(f)

faculty_data = load_data()

def expand_project_description(description):
    prompt = f"You are an expert project descriptor elaborate the following project description into a detailed overview,include all academia/research keywords, concepts, expand the abbrevations,methods technologies involved, and expected outcomes. The outcome should be a single paragraph containing all the required details with no punctuation and full stops all in single paragraph: '{description}'."
    response = model.generate_content(prompt)
    return response.text

# Define function to find matching faculties

@st.cache_data
def find_matching_faculties(prompt):
    
    expanded_description = expand_project_description(prompt)
    # Extract keywords from the expanded description
    # st.write(expanded_description)
    query_keywords = extract_keywords(expanded_description)
    
    faculty_matches = []
    query_embedding = semantic_model.encode([query_keywords])[0]
    for faculty in faculty_data:
        faculty_keywords = faculty["Keywords"]

        # Calculate semantic similarity between query keywords and faculty keywords
        faculty_embedding = semantic_model.encode([faculty_keywords])[0]
        
        similarity_score = cosine_similarity([query_embedding], [faculty_embedding])[0][0]

        if(similarity_score>=0.25):
            # faculty_matches.append({"Name":faculty["Name"], "Department":faculty["Department"],"Similarity":similarity_score,"Keywords": faculty_keywords})
            faculty_matches.append((faculty["Name"], similarity_score, faculty))
            
    faculty_matches.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
    return faculty_matches[:5]
#     expanded_description = expand_project_description(prompt)
#     query_keywords = extract_keywords(expanded_description)

#   # Create TF-IDF vectorizer (adjust parameters as needed)
#     vectorizer = TfidfVectorizer(max_features=1000)  # Limit vocabulary size (optional)

#   # Transform text data into TF-IDF vectors
#     query_vec = vectorizer.fit_transform([query_keywords])
#     faculty_vecs = vectorizer.transform([faculty["Keywords"] for faculty in faculty_data])

#     faculty_matches = []
#     for i, faculty in enumerate(faculty_data):
#         similarity_score = cosine_similarity(query_vec, faculty_vecs[i])[0][0]
#         faculty_matches.append((faculty["Name"], similarity_score, faculty))
    
    
#     faculty_matches.sort(key=lambda x: x[1], reverse=True)
#     return faculty_matches[:5]
    

@st.cache_data
def extract_keywords(text):
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(clean_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
def validate_and_explain_matches_with_gemini(matches, project_description):
    detailed_matches = []
    for faculty_name, similarity, faculty_info in matches:
        try:
      # Construct a detailed prompt asking for relevance and contribution
            # prompt = f"Explain why this faculty {faculty_name} from the {faculty_info['Department']} is relevant for a project on '{project_description}'. Here are the keywords from their research texts: {', '.join(faculty_info['Keywords'])}. How can they contribute to this project? give one line response only. if the faculty is not even remotely related to the project write 'nothere' in output.be lenient in deciding it and keep the professors which are related and of the same feild, also look at the department and then decide."
            prompt = f"Can you provide a brief explanation of how the research expertise of {faculty_name} from the {faculty_info['Department']} aligns with the project on '{project_description}'? Here are some keywords related to their search {faculty_info['Keywords']}.Limit the response to at most 2 line"

            response = model.generate_content(prompt)
            
            if 'nothere' not in response.text.lower():
                detailed_matches.append((faculty_name, similarity, faculty_info,response.text))
        except Exception as e:
      # Log the error or display a user-friendly message
            continue
     
    return detailed_matches

# Define Streamlit app
def main():
    st.title("Faculty Matching App")

    # Use form to manage input field
    with st.form(key='my_form'):
        # Define a unique key for the text input
        search_key = "search_input"

        # Use session state to manage the input value
        if "input_value" not in st.session_state:
            st.session_state.input_value = ""

        # Set the text input value using the session state
        prompt = st.text_input("Enter your project idea:", value=st.session_state.input_value, key=search_key)

        # Add a placeholder for the clear button
        placeholder = st.empty()

        # Add a button to the form to submit
        submitted = st.form_submit_button("Find Matching Faculties")

    # If form submitted, process the input
    if submitted:
        if prompt:
            matching_faculty_details = find_matching_faculties(prompt)
            # final_details=validate_and_explain_matches_with_gemini(matching_faculty_details, prompt)
            st.markdown("### Top 5 Matching Faculties based on keywords:")
            for faculty_name, similarity, faculty_info in matching_faculty_details:
                st.write(f"**Name:** {faculty_name} (Similarity: {similarity:.4f})")
                st.write(f"**Designation:** {faculty_info['Designation']}")
                st.write(f"**Department:** {faculty_info['Department']}")
                st.write(f"**Email:** {faculty_info['Email']}")
                # st.write(f"**Response:** {response}")
                st.write("---")
            st.write("")  # Add empty line to separate search results from clear button
            st.write("")  # Add empty line for spacing
            if st.button("Clear Search"):
                st.session_state.input_value = ""
                

    # Clear search input

if __name__ == "__main__":
    main()