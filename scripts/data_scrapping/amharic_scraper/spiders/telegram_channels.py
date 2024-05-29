import scrapy
import psycopg2
from psycopg2.extras import RealDictCursor

class AmharicNewsSpider(scrapy.Spider):
    name = 'amharic_news'
    start_urls = ['https://am.al-ain.com/article/u-s-screens-record-3-million-airline-passengers-in-a-day']

    def parse(self, response):
        # Use Scrapy selector to find all paragraph elements
        url = response.url
        title = response.css('title::text').get()
        header = response.css('h1::text').get()
        time = response.css('time::attr(datetime)').get()

        # Extracting paragraphs
        paragraphs = response.css('p::text').getall()

        # Validate the extracted data
        if title and header and paragraphs:
            # Prepare the data to be inserted
            data_to_insert = {
                'url': url,
                'title': title,
                'header': header,
                'time': time,
                'paragraphs': ', '.join(paragraphs)
            }

            # Send the data to a pipeline for further processing
            yield data_to_insert

        # Follow links recursively if they start with 'https://www.bbc.com/amharic/articles/'
        for link in response.css('a::attr(href)').getall():
            if link.startswith('https://am.al-ain.com/article/'):
                yield response.follow(link, callback=self.parse)