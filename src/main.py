from crew import ResearchCrew
import datetime

def run(model='Groq'):
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'user_input': 'What is the first President of Brazil?',
    }

    ResearchCrew(model).create_crew().kickoff(inputs=inputs)

if __name__ == '__main__':
    run('Groq')
    