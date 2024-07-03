import os
from support_functions import load_config
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tools.tools import SearchTools
from langchain_groq import ChatGroq
from crewai_tools import ScrapeWebsiteTool, ScrapeElementFromWebsiteTool, SeleniumScrapingTool
from langchain_openai import ChatOpenAI

@CrewBase
class ResearchCrew:
    def __init__(self, model='Groq'):
        self.agents_config = load_config("config/agents.yaml")
        self.tasks_config = load_config("config/tasks.yaml")
        self.models_config = load_config("config/models.yaml")


        #define LLM model
        if model == 'Groq':
            self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model='mixtral-8x7b-32768')
        elif model == 'Gpt3.5':
            self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model='gpt-3.5-turbo-0125')
            print("Using Gpt3.5 LLM model \n\n")

        # cerating tasks and agents
        self.agents = self.create_agents()
        self.tasks = self.create_tasks()

    def create_agents(self):
        manager_agent = Agent(
            config=self.agents_config["manager_agent"],
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=True,
        )

        researcher_agent = Agent(
            config=self.agents_config["researcher_agent"],
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=True,
        )

        historySpecialist_agent = Agent(
            config=self.agents_config["historySpecialist_agent"],
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

        writer_agent = Agent(
            config=self.agents_config["writer_agent"],
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

        reviser_agent = Agent(
            config=self.agents_config["reviser_agent"],
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )
    
        return [manager_agent, researcher_agent, historySpecialist_agent, writer_agent, reviser_agent]

    def create_tasks(self):
        manager_task = Task(
            config=self.tasks_config["manager_task"],
            agent=self.agents[0],          
        )

        researcher_task = Task(
            config=self.tasks_config["researcher_task"],
            agent=self.agents[1], 
        )

        historySpecialist_task = Task(
            config=self.tasks_config["historySpecialist_task"],
            agent=self.agents[2], 
        )

        writer_task = Task(
            config=self.tasks_config["writer_task"],
            agent=self.agents[3], 
        )

        reviser_task = Task(
            config=self.tasks_config["reviser_task"],
            agent=self.agents[4], 
        )
    
        return [manager_task, researcher_task, historySpecialist_task, writer_task, reviser_task]
    
    def create_crew(self, model='Groq') -> Crew:
        """ Creates the Crew """
        return Crew(
            agents=self.create_agents(), #Automatically created by the @agent decorator
            tasks=self.create_tasks(), #Automatically created by the @task decorator
            process=Process.hierarchical,
            manager_llm=self.llm,
            verbose=True,
        )
    