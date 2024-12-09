#import vectorize as vtr
import query as qry
#import nltk

"""
nltk.download('punkt_tab')
print("Download done")

vtr_obj = vtr.Vectorize()

vtr_obj.chunker("https://www.webmd.com/lung/understanding-bronchitis-basics")
vtr_obj.store()
"""

query = ""
response = None
qry_obj = qry.Query()
qry_obj.retrieval_chain()


while (query is not 'q'):
    query = input("User: ")
    response = qry_obj.response(query=query)
    print(f"\n Bot: {response}")