import psycopg2
from itemadapter import ItemAdapter

class AmharicScraperPipeline:
    def open_spider(self, spider):
        try:
            self.connection = psycopg2.connect(database="amharic_db", user="postgres", password="postgres", host="db", port="5432")
            self.cursor = self.connection.cursor()
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            # Log the error and continue with the next item instead of stopping the spider
            # raise

    def close_spider(self, spider):
        # Check if the cursor and connection were successfully created
        if hasattr(self, 'cursor') and self.cursor is not None:
            self.cursor.close()
        if hasattr(self, 'connection') and self.connection is not None:
            self.connection.close()

    def process_item(self, item, spider):
        item_dict = ItemAdapter(item).asdict()
        try:
            # Use parameterized queries to avoid SQL injection
            sql = """INSERT INTO news_articles_raw (url, title, header, publish_time, content) VALUES (%(url)s, %(title)s, %(header)s, %(time)s, %(paragraphs)s)"""
            self.cursor.execute(sql, {
                'url': item_dict.get('url'),
                'title': item_dict.get('title'),
                'header': item_dict.get('header1'),  # Assuming 'header1' corresponds to 'header' in the database
                'time': item_dict.get('publish_time'),  # Assuming 'publish_time' corresponds to 'time' in the database
                'paragraphs': item_dict.get('paragraphs')
            })
            self.connection.commit()
        except Exception as e:
            print(f"Failed to insert item into database: {e}")