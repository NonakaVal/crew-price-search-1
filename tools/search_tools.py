##################################################################################################################
##################################################################################################################
## Tool de Pesquisa para crewai
##################################################################################################################
##################################################################################################################


import json
import os
import requests
from langchain.tools import tool
import streamlit as st



api_key = st.secrets['SERPER_API_KEY']

class SearchTools():

  @tool("Search internet")
  def search_internet(query):
    """Useful to search the internet about a given topic and return relevant
    results."""
    return SearchTools.search(query)
  
  @tool("Search instagram")
  def search_instagram(query):
    """Useful to search for instagram post about a given topic and return relevant
    results."""
    query = f"site:instagram.com {query}"
    return SearchTools.search(query)

  def search(query, n_results=15):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': api_key,
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    stirng = []
    for result in results[:n_results]:
      try:
        stirng.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    content = '\n'.join(stirng)
    return f"\nSearch result: {content}\n"



# class SearchTools():
  

#   @tool("Search the internet")
#   def search_internet(query):
#     """Useful to search the internet
#     about a a given topic and return relevant results"""
#     serper_api_key = secret_key
#     # os.getenv('SERPER_API_KEY')

#     top_result_to_return = 5
#     url = "https://google.serper.dev/search"
#     payload = json.dumps({"q": query})
#     headers = {
#       'X-API-KEY': serper_api_key,
#       'Content-Type': 'application/json'
# }
#     response = requests.request("POST", url, headers=headers, data=payload)
#     # check if there is an organic key
#     if 'organic' not in response.json():
#       return "Sorry, I couldn't find anything about that, there could be an error with you serper api key."
#     else:
#       results = response.json()['organic']
#       string = []
#       for result in results[:top_result_to_return]:
#         try:
#           string.append('\n'.join([
#               f"Title: {result['title']}", f"Link: {result['link']}",
#               f"Snippet: {result['snippet']}", "\n-----------------"
#           ]))
#         except KeyError:
#           next

#       return '\n'.join(string)
