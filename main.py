#import vectorize as vtr
import query as qry
"""
#import nltk

#nltk.download('punkt_tab')
#print("Download done")

vtr_obj = vtr.Vectorize()

vtr_obj.chunker("https://www.webmd.com/lung/understanding-bronchitis-basics")
vtr_obj.store()
#print(vtr_obj.test("bronchitis"))

"""

query = " "
response = None
qry_obj = qry.Query()
qry_obj.retrival_chain()


while (query[0] != 'q'):
    #qry_obj.indexs()
    query = input("\nUser: ")
    if (query[0] == 'q'):
        break
    response = qry_obj.response(query=query)
    print(f"\n Bot: {response}")