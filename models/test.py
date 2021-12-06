from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', con=engine)
print(df.head())

# engine = create_engine('sqlite:///' + database_filepath)
# table_name = os.path.basename(database_filepath).split('.')[0]
# df = pd.read_sql_table(table_name,con=engine)