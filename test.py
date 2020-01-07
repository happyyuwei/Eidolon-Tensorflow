import logging
import sys

# logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(funcName)s - %(module)s - %(message)s")

# def hello():
#     logging.info("hello")
#     logging.error("error")
from wordcloud import WordCloud

text="hello how are you"
WordCloud = WordCloud().generate(text)
image_produce = WordCloud.to_image()
image_produce.show()


