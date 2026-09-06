"""
load_to_sqlite.py
Takes the cleaned DataFrame produced by datalayer.py's run_pipeline()
and persists it into a SQLite database for query-based retrieval.
"""
import sqlite3
from data.datalayer import run_pipeline
import pandas as pd


def load_to_sqlite(final_df, db_path="ohlcv.db", table_name="ohlcv"):
    """
    Input : cleaned, validated DataFrame (indexed by real_date, ticker)
    Output: SQLite database file with the data loaded into `table_name`
    """
    # 1. Open a connection to the SQLite database file
    #    (sqlite3.connect creates the file if it doesn't exist yet)
    conn = sqlite3.connect(db_path)
    

    # 2. final_df has a MultiIndex (real_date, ticker) — SQLite needs flat
    #    columns, not a pandas index. Turn the index back into columns.
    df_to_load = final_df.reset_index()
    
    column = df_to_load.columns
    print(f'column = {column}')

    # 3. Write the DataFrame into a SQLite table.
    #    Think about: what should happen if this table already exists?
    #    (replace it fresh each run, or append to it?)
    df_to_load.to_sql(table_name, conn, if_exists = 'replace', index= False)
    
    
    
    # 4. Create an index on the columns most queries will filter/sort by.
    #    Which columns will your rolling-metric and asset-filtering
    #    queries actually use in WHERE / ORDER BY / PARTITION BY?
    # conn.execute("")
    conn.commit()


    # 5. SQL CHECK
    check_distinct_trading_days = pd.read_sql("""
        SELECT 
            MIN(trading_days) as min_days,
            MAX(trading_days) as max_days,
            COUNT(DISTINCT trading_days) as unique_day_counts
        FROM (
            SELECT ticker, COUNT(*) as trading_days
            FROM ohlcv
            GROUP BY ticker
        )
    """, conn)
    
    df_countnull = pd.read_sql("""
            with temp as (
                select ticker, 
                    count(*) - count(px_last) as px_last_null,
                    count(*) - count(px_volume) as px_volume_null,
                    count(*) - count(px_open) as px_open_null,
                    count(*) - count(px_low) as px_low_null,
                    count(*) - count(px_high) as px_high_null
                from ohlcv
                group by ticker
            )
            select sum(px_last_null) as total_last_null,
                    sum(px_volume_null) as total_volume_null,
                    sum(px_open_null) as total_open_null,
                    sum(px_low_null) as total_low_null,
                    sum(px_high_null) as total_high_null
            from temp
                            """, conn)
    ticker = pd.read_sql("""
                select count(distinct ticker) as total_ticker
                from ohlcv 
                         """, conn)
    
    date_range_confirmation = pd.read_sql("""
                            SELECT 
                                MIN(real_date) as start_date,
                                MAX(real_date) as end_date,
                                COUNT(DISTINCT real_date) as total_trading_days
                            FROM ohlcv       
                                          """,conn)
    
    most_volative_ticker = pd.read_sql("""
                    SELECT ticker,
                        AVG(simple_return) as avg_return,
                        AVG(simple_return * simple_return) - 
                        AVG(simple_return) * AVG(simple_return) as variance
                    FROM ohlcv
                    GROUP BY ticker
                    ORDER BY variance DESC
                    LIMIT 10                         
                                       """, conn)
    print(f'check distinct trading days {check_distinct_trading_days}')
    print(f'surviving ticker {ticker}')
    print(f'date start and date end {date_range_confirmation}')
    print(f'count_null{df_countnull}')
    print(f'most_volative_ticker {most_volative_ticker}')


    # 6. Always close the connection when done
    conn.close()
    
    


if __name__ == "__main__":
    # Run the actual cleaning pipeline first
    final_df, issues = run_pipeline("NDX_Universe_OHLC.xlsx")

    # Then persist the cleaned result to SQLite
    load_to_sqlite(final_df)
