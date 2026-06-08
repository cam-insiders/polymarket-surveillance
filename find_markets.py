"""
Search for markets in your local dataset.
Usage: python find_markets.py "trump" "bitcoin" "election"
"""

import pandas as pd
import sys


def search_markets(keywords):
    """Search markets.csv for keywords"""
    
    df = pd.read_csv("data/markets.csv")
    
    print(f"\nSearching {len(df):,} markets for: {', '.join(keywords)}\n")
    print("="*100)
    
    for keyword in keywords:
        # Search in market_slug
        matches = df[df['market_slug'].str.contains(keyword, case=False, na=False)]
        
        if len(matches) == 0:
            print(f"\n⚠️  No matches for '{keyword}'")
            continue
        
        print(f"\n🔍 Found {len(matches)} markets matching '{keyword}':")
        print("-"*100)
        
        # Sort by volume
        matches = matches.sort_values('volume', ascending=False)
        
        for i, row in matches.head(20).iterrows():
            resolved = "✓ RESOLVED" if pd.notna(row['closedTime']) else "⏳ Active"
            
            print(f"\nMarket ID: {row['id']}")
            print(f"  Slug: {row['market_slug']}")
            print(f"  Volume: ${row['volume']:,.0f}")
            print(f"  Status: {resolved}")
            print(f"  Outcomes: {row['answer1']} vs {row['answer2']}")
            print(f"  Condition ID: {row['condition_id'][:20]}...")
            
            if pd.notna(row['closedTime']):
                print(f"  Closed: {row['closedTime']}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_markets.py <keyword1> <keyword2> ...")
        print("Example: python find_markets.py trump bitcoin election")
        sys.exit(1)
    
    keywords = sys.argv[1:]
    search_markets(keywords)