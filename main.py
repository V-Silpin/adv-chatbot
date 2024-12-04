import vectorize as vtr
import nltk
nltk.download('punkt_tab')
print("Download done")

vtr_obj = vtr.Vectorize()

vtr_obj.chunker("https://www.webmd.com/lung/understanding-bronchitis-basics")
vtr_obj.store()