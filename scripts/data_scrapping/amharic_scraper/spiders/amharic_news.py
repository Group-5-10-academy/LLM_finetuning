import scrapy
import psycopg2
from psycopg2.extras import RealDictCursor

class AmharicNewsSpider(scrapy.Spider):
    name = 'amharic_news'
    start_urls = ['https://www.bbc.com/amharic/articles/cg33ge4l23vo']

    def __init__(self):
        super().__init__()
        # Establish a connection to the PostgreSQL database
        self.conn = psycopg2.connect(
            dbname='amharic_db',
            user='postgres',
            password='postgres',
            host='localhost',  
            port='5432'
        )
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)

    def close(self, reason):
        # Close the cursor and connection when the spider is closed
        self.cur.close()
        self.conn.close()

    def parse(self, response):
        # Use Scrapy selector to find all paragraph elements
        paragraphs = response.css('p::text').getall()[0]
        title = response.css('title::text').getall()[0]  # Get the first title element
        header = response.css('h1::text').getall()[0]  # Get the first h1 element

        # Prepare the data to be inserted
        data_to_insert = {
            'title': title,
            'header': header,
            'p_tags': paragraphs
        }

        # Insert the data into the database
        self.cur.execute("""
            INSERT INTO amharic_data (title, header1, paragraphs)
            VALUES (%(title)s, %(header)s, %(p_tags)s)
        """, data_to_insert)

        # Commit the transaction
        self.conn.commit()

        self.log(f'Scraped item: {data_to_insert}')  # Log the item to the console
