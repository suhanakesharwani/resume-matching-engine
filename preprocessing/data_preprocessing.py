# Import Necessary Data Modules
from data.candidates import CANDIDATES
from data.jobs import JOBS
from data.skill_aliases import SKILL_ALIASES


# Function to Normalize Text
def normalize(text):
    """
    Normalize the input text by converting it to lowercase,
    splitting it into tokens, and replacing skill aliases.

    Args:
        text (str): Raw skill string containing comma-separated skills.

    Returns:
        list: A list of normalized skills.
    """

    # Convert text to lowercase
    text = text.lower()

    # Split text into tokens and remove whitespace
    tokens = [t.strip() for t in text.split(",")]

    # Initialize list to store normalized skills
    skills = []

    # Iterate over each token
    for token in tokens:

        # Check if token exists in skill aliases
        if token in SKILL_ALIASES:

            # Append canonical skill name
            skills.append(SKILL_ALIASES[token])

    # Return normalized skills
    return skills


# Function to Preprocess Data
def preprocess():
    """
    Preprocess candidate and job data by normalizing skills.

    Returns:
        tuple: Candidate names, normalized resumes,
               job names, and normalized job skills.
    """

    # Initialize lists for candidate data
    candidate_names = []
    resumes = []

    # Process candidate records
    for name, skills in CANDIDATES:
        candidate_names.append(name)
        resumes.append(normalize(skills))

    # Initialize lists for job data
    job_names = []
    jobs = []

    # Process job descriptions
    for jd, (role, skills) in JOBS.items():
        job_names.append(jd)
        jobs.append(normalize(skills))

    # Return preprocessed data
    return candidate_names, resumes, job_names, jobs