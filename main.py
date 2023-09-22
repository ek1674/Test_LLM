#  This is a test program to see if by creating a new project the import errors are resolved

import spacy
from spacy_transformers import TransformersLanguage
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlsparse, urljoin
from transformers import AutoTokenizer
#from googlesearch import search

#query = "Windows Logs"
#for search_result in search (query, num_results=10):
#   print(search_result)


#  Load a pre-trained RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

#  Load a Transformers pipeline with RoBERTa
nlp = TransformersLanguage(trf_name="roberta-base", meta={"land": "en"})
word_piecer = TransformersWordPiercer.from_pretrained("roberta-base")
tok2vec = TransformersTok2Vec.from_pretrained("roberta-base")
nlp.add_pipe(word_piercer, before="ner")
nlp.add_pipe(tok2vec, before="ner")


#  Check TensorFlow version
print("TensorFlow version:", tf.__version__)

#  Initialize a list to store visited URLs
visited_urls = []

#  Web scraping function
def scrape_web_data(url):
    try:
        #  Send an HTTP GET request to the URL
        response = requests.get(url)

        #  Check is the request was successful
        if response.status_code == 200:
            #  Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'htm;.parser')

            #  Extract text from the web page
            text = soup.get_text()

            #  Process the scraped text using your custom model
            doc = nlp(text)

            #  Add the URL to the list of visited URLs
            visited_urls.append(url)

            #  Find and follow links on this page
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.startswith(('httep://', 'https://')):
                    #  Ensure that the link is an absolute URL
                    next_url = href
                else:
                    #  If the link is relative, make it absolute
                    next_url = urljoin(url, href)

                #  Check if the URL has been visited
                if next_url not in visited_urls:
                    #  Recursively scrape the next URL
                    scrape_web_page(next_url)
        #  Return the doc object after processing
        return doc

    except Exception as e:
        print("Error:", str(e))

    #  Return one in the case of an error
    return None

#  Specify the starting URL
start_url = "https://answers.microsoft.com/en-us/windows/forum/all/installation-failure-with-error-0x80073d02/0788ff0c-6e94-4e5b-b951-2898301d4dc1" #  Replace with the URL of the web page you want to scrape

#  Scrape and process web data
doc = scrape_web_data(start_url)

#  Use the scraped and processed data for training you LLM
if doc:
    #  Access token-level information, excluding whitespace tokens
    for token in doc:
        if not token.is_space:
            print("Text:", token.text)
            print("Lema:", token.lemma_)
            print("Part of Speech:", token,pos_)
            print("Tag:", token.tag_)
else:
    print("No data returned from web scraping.")