import sys
import time
import streamlit as st
from crewai import Agent, Task, Crew, Process
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
import os
import re
from crewai_tools import ScrapeWebsiteTool, WebsiteSearchTool
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools

openai_api_key = st.secrets["OPENAI_API_KEY"]

Search_tools = [SearchTools.search_internet, BrowserTools.scrape_and_summarize_website, ScrapeWebsiteTool(), WebsiteSearchTool()]

if openai_api_key:
    try:
        # Criar uma instância do modelo ChatOpenAI
        llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            api_key=openai_api_key
        )
        st.success("A chave da API do OpenAI está configurada e o LLM está pronto para usar!")    
    except Exception as e:
        st.error(f"Ocorreu um erro ao configurar o LLM: {e}")
else:
    st.warning("Por favor, insira uma chave de API do OpenAI válida para continuar.")

# duckduckgo_search = DuckDuckGoSearchRun()

#to keep track of tasks performed by agents
task_values = []
# def create_crewai_setup(product_name):
#     # Definir número máximo de iterações
#     max_iter = 70

#     # Definição dos Agentes
#     product_code_collector = Agent(
#         role='Coletor de Código Universal do Produto',
#         goal=f'Encontrar o código universal do produto usando o nome "{product_name}".',
#         backstory='Especialista em rastrear códigos de produtos, esse agente utiliza o nome do produto para buscar o código universal em diferentes fontes de dados.',
#         llm=llm,
#         max_iter=max_iter,
#         allow_delegation=True,
#         tools=Search_tools,  # Ferramenta de busca na web para coletar dados
#         verbose=True
#     )
    
#     product_code_verifier = Agent(
#         role='Verificador de Código Universal',
#         goal=f'Confirmar e validar o código universal encontrado para o produto "{product_name}", garantindo que seja preciso e atualizado.',
#         backstory='Com experiência em verificação de dados, este agente cruza informações de várias fontes para garantir a precisão do código encontrado.',
#         llm=llm,
#         max_iter=max_iter,
#         allow_delegation=False,
#         tools=Search_tools,
#         verbose=True
#     )

#     # Definição das Tarefas
#     product_code_collection_task = Task(
#         description=f"""
#         Pesquisar o código universal do produto com base no nome "{product_name}". 
#         Use fontes de dados online para identificar o código correto.
#         """,
#         expected_output=f"Código universal encontrado para o produto {product_name}.",
#         agent=product_code_collector
#     )

#     product_code_verification_task = Task(
#         description=f"""
#         Verificar e validar o código universal encontrado para o produto "{product_name}". 
#         Garantir que o código está correto e atualizado, comparando com múltiplas fontes.
#         """,
#         expected_output=f"Código universal validado e preciso para o produto {product_name}.",
#         agent=product_code_verifier
#     )

#     # Configuração da Equipe (Crew)
#     crew = Crew(
#         agents=[product_code_collector, product_code_verifier],
#         tasks=[product_code_collection_task, product_code_verification_task],
#         process=Process.sequential,  # Processo sequencial para execução das tarefas
#         manager_llm=llm,  # Modelo LLM para gerenciar a equipe
#         verbose=True
#     )

#     # Iniciar a execução das tarefas
#     crew_result = crew.kickoff()
#     return crew_result


def create_crewai_setup(product_name, llm):

    max_iter = 50

    # Definir Agentes
    price_scraper = Agent(
        role="Especialista em Busca de Preços",
        goal=f"""Realizar buscas em sites de e-commerce e marketplaces para coletar preços de {product_name} no mercado brasileiro em reais""",
        backstory=f"""Especialista em coleta de dados online, focado em encontrar o maior número de preços disponíveis em diversas plataformas.""",
        verbose=True,
        allow_delegation=False,
        tools=[ScrapeWebsiteTool(), WebsiteSearchTool()],
        llm=llm,
        max_iter=max_iter
    )

    specification_analyzer = Agent(
        role="Especialista em Análise de Especificações",
        goal=f"""Analisar as especificações técnicas e variações de {product_name} nos diferentes sites para garantir a compatibilidade e consistência das ofertas.""",
        backstory=f"""Especialista em análise de especificações de produtos, com foco em verificar as diferenças e similaridades entre ofertas.""",
        verbose=True,
        allow_delegation=False,
        tools=[ScrapeWebsiteTool(), WebsiteSearchTool()],
        llm=llm,
        max_iter=max_iter
    )

    price_analyzer = Agent(
        role="Analista de Preços",
        goal=f"""Comparar os preços coletados para determinar o menor, o maior e a média de preços para {product_name}, incluindo uma análise das variações e descontos.""",
        backstory=f"""Especialista em análise de preços, com experiência em identificar variações de preços, descontos e padrões de flutuação.""",
        verbose=True,
        allow_delegation=False,
        tools=[ ScrapeWebsiteTool(), WebsiteSearchTool()],
        llm=llm,
        max_iter=max_iter
    )

    review_analyzer = Agent(
        role="Especialista em Análise de Avaliações de Produtos",
        goal=f"""Analisar avaliações e classificações de clientes para {product_name}, identificando padrões de satisfação e insatisfação entre as ofertas.""",
        backstory=f"""Especialista em análise de feedback de clientes, focado em entender a qualidade percebida e os principais pontos positivos e negativos.""",
        verbose=True,
        allow_delegation=False,
        tools=[ScrapeWebsiteTool(), WebsiteSearchTool()],
        llm=llm,
        max_iter=max_iter
    )

    # Definir Tarefas
    task1 = Task(
        description=f"""Buscar e coletar os preços de {product_name} em diversos sites de e-commerce e marketplaces brasileiros em reais""",
        expected_output="Lista de preços coletados com links para cada oferta.",
        agent=price_scraper
    )

    task2 = Task(
        description=f"""Analisar as especificações técnicas dos produtos encontrados para verificar se são compatíveis ou se há variações significativas.""",
        expected_output="Relatório de compatibilidade de especificações com observações sobre variações encontradas.",
        agent=specification_analyzer
    )

    task3 = Task(
        description=f"""Comparar os preços coletados, determinando o menor, o maior e a média de preços para {product_name}.""",
        expected_output="Relatório com análise de preço mínimo, máximo, e médio, com observações sobre descontos e variações.",
        agent=price_analyzer
    )

    task4 = Task(
        description=f"""Analisar as avaliações e classificações de clientes para identificar padrões de satisfação ou problemas comuns com o {product_name}.""",
        expected_output="Relatório de análise de feedback com insights sobre a qualidade percebida e problemas recorrentes.",
        agent=review_analyzer
    )

    # Criar e Executar a Equipe
    price_comparison_crew = Crew(
        agents=[price_scraper, specification_analyzer, price_analyzer, review_analyzer],
        tasks=[task1, task2, task3, task4],
        verbose=True,
        process=Process.sequential,  # Alterado para Process.sequential
        manager_llm=llm
    )

    crew_result = price_comparison_crew.kickoff()
    return crew_result



#display the console processing on streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Market Research Analyst" in cleaned_data:
            # Apply different color 
            cleaned_data = cleaned_data.replace("Market Research Analyst", f":{self.colors[self.color_index]}[Market Research Analyst]")
        if "Business Development Consultant" in cleaned_data:
            cleaned_data = cleaned_data.replace("Business Development Consultant", f":{self.colors[self.color_index]}[Business Development Consultant]")
        if "Technology Expert" in cleaned_data:
            cleaned_data = cleaned_data.replace("Technology Expert", f":{self.colors[self.color_index]}[Technology Expert]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

# Streamlit interface
def run_crewai_app():

    # with st.expander("About the Team:"):
    #     st.subheader("Diagram")
    #     left_co, cent_co,last_co = st.columns(3)
    #     with cent_co:
    #         st.image("my_img.png")

    #     st.subheader("Market Research Analyst")
    #     st.text("""       
    #     Role = Market Research Analyst
    #     Goal = Analyze the market demand for {product_name} and suggest marketing strategies
    #     Backstory = Expert at understanding market demand, target audience, 
    #                 and competition for products like {product_name}. 
    #                 Skilled in developing marketing strategies 
    #                 to reach a wide audience.
    #     Task = Analyze the market demand for {product_name}. Current month is Jan 2024.
    #            Write a report on the ideal customer profile and marketing 
    #            strategies to reach the widest possible audience. 
    #            Include at least 10 bullet points addressing key marketing areas. """)
        
    #     st.subheader("Technology Expert")
    #     st.text("""       
    #     Role = Technology Expert
    #     Goal = Assess technological feasibilities and requirements for producing high-quality {product_name}
    #     Backstory = Visionary in current and emerging technological trends, 
    #                 especially in products like {product_name}. 
    #                 Identifies which technologies are best suited 
    #                 for different business models. 
    #     Task = Assess the technological aspects of manufacturing 
    #            high-quality {product_name}. Write a report detailing necessary 
    #            technologies and manufacturing approaches. 
    #            Include at least 10 bullet points on key technological areas.""")

    #     st.subheader("Business Development Consultant")
    #     st.text("""       
    #     Role = Business Development Consultant 
    #     Goal= Evaluate the business model for {product_name}
    #           focusing on scalability and revenue streams
    #     Backstory = Seasoned in shaping business strategies for products like {product_name}. 
    #                 Understands scalability and potential 
    #                 revenue streams to ensure long-term sustainability.
    #     Task = Summarize the market and technological reports 
    #            and evaluate the business model for {product_name}. 
    #            Write a report on the scalability and revenue streams 
    #            for the product. Include at least 10 bullet points 
    #            on key business areas. Give Business Plan, 
    #            Goals and Timeline for the product launch. Current month is Jan 2024. """)
    
    product_name = st.text_input("Enter a product name to analyze the market and business strategy.")

    if st.button("Run Analysis"):
        # Placeholder for stopwatch
        stopwatch_placeholder = st.empty()
        
        # Start the stopwatch
        start_time = time.time()
        with st.expander("Processing!"):
            sys.stdout = StreamToExpander(st)
            with st.spinner("Generating Results"):
                crew_result = create_crewai_setup(product_name=product_name,llm=llm)

        # Stop the stopwatch
        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")

        st.header("Tasks:")
        st.table({"Tasks" : task_values})

        st.header("Results:")
        st.markdown(crew_result)

if __name__ == "__main__":
    run_crewai_app()
